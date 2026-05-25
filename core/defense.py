import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import io

# ==========================================
# 1. INPUT PREPROCESSING DEFENSES
# ==========================================

def spatial_smoothing(image_tensor: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Applies Spatial Smoothing (average pooling) to remove high-frequency adversarial noise.
    """
    # Create average pooling layer as a smooth spatial filter
    padding = kernel_size // 2
    smoother = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding)
    return smoother(image_tensor)

def jpeg_compression_simulation(image_tensor: torch.Tensor, quality: int = 50) -> torch.Tensor:
    """
    Simulates JPEG compression defense by converting tensor to PIL Image, 
    compressing it, and loading it back to remove high-frequency perturbations.
    """
    # Operates on a single tensor image or batches
    # If batched, loop over images
    if len(image_tensor.shape) == 4:
        compressed_batch = []
        for i in range(image_tensor.size(0)):
            compressed_batch.append(jpeg_compression_single(image_tensor[i], quality))
        return torch.stack(compressed_batch)
    else:
        return jpeg_compression_single(image_tensor, quality)

def jpeg_compression_single(image_tensor: torch.Tensor, quality: int) -> torch.Tensor:
    # Convert [C, H, W] tensor in range [0,1] to PIL
    np_img = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    if np_img.shape[2] == 1: # Grayscale support
        np_img = np_img.squeeze(2)
        pil_img = Image.fromarray(np_img, mode='L')
    else:
        pil_img = Image.fromarray(np_img, mode='RGB')
        
    # Save to memory buffer as JPEG with specified quality
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    
    # Reload and convert back to tensor
    reloaded_pil = Image.open(buffer)
    reloaded_np = np.array(reloaded_pil).astype(np.float32) / 255.0
    if len(reloaded_np.shape) == 2:
        reloaded_np = np.expand_dims(reloaded_np, axis=2)
        
    reloaded_tensor = torch.from_numpy(reloaded_np).permute(2, 0, 1).to(image_tensor.device)
    return reloaded_tensor

def bit_depth_reduction(image_tensor: torch.Tensor, bits: int = 3) -> torch.Tensor:
    """
    Reduces the bit depth of pixel representations (quantization), 
    limiting the small, subtle perturbation capacity.
    Formula: round(x * levels) / levels where levels = 2^bits - 1
    """
    levels = (2 ** bits) - 1
    return torch.round(image_tensor * levels) / levels


# ==========================================
# 2. ADVERSARIAL TRAINING
# ==========================================

def adversarial_training_epoch(model: nn.Module, dataloader: DataLoader, 
                               optimizer: optim.Optimizer, loss_fn: nn.Module, 
                               epsilon: float, device: torch.device, 
                               attack_type: str = "fgsm") -> float:
    """
    Runs a single training epoch where 50% of the inputs are standard images,
    and 50% are replaced with adversarially crafted images (FGSM or PGD).
    """
    model.train()
    running_loss = 0.0
    
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        
        # Determine size of batch split
        batch_size = images.size(0)
        split_idx = batch_size // 2
        
        # Split batch into standard and adversarial training streams
        std_images = images[:split_idx]
        adv_images = images[split_idx:]
        adv_targets = targets[split_idx:]
        
        # Generate online adversarial perturbations for the second half
        if len(adv_images) > 0 and epsilon > 0:
            # Requires grads for attack computation
            adv_images.requires_grad = True
            outputs = model(adv_images)
            loss = loss_fn(outputs, adv_targets)
            model.zero_grad()
            loss.backward()
            
            # Extract gradients and perform FGSM/PGD attack
            data_grad = adv_images.grad.data
            
            # Simple FGSM generation
            sign_grad = data_grad.sign()
            perturbed_images = adv_images + epsilon * sign_grad
            perturbed_images = torch.clamp(perturbed_images, 0, 1).detach()
            
            # Concatenate back into a single composite batch
            x_train = torch.cat([std_images, perturbed_images], dim=0)
        else:
            x_train = images
            
        # Standard backpropagation on training data
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_size
        
    return running_loss / len(dataloader.dataset)
