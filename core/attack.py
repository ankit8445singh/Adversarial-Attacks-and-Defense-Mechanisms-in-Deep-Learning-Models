import torch
import torch.nn as nn

def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) Attack.
    Formula: x_adv = x + epsilon * sign(gradient)
    
    Args:
        image: Original clean image tensor (shape: [B, C, H, W])
        epsilon: Perturbation strength control factor
        data_grad: Gradients of the loss with respect to the input image
        
    Returns:
        perturbed_image: The adversarial image tensor, clamped to standard [0, 1] range.
    """
    # 1. Collect the sign of the input gradients
    sign_data_grad = data_grad.sign()
    
    # 2. Add the perturbation to the original image pixels
    perturbed_image = image + epsilon * sign_data_grad
    
    # 3. Clip the perturbed image pixels to maintain valid normalized range [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

def pgd_attack(model: nn.Module, image: torch.Tensor, target: torch.Tensor, 
               epsilon: float, alpha: float = 0.01, num_iter: int = 10) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) Attack.
    An iterative extension of FGSM that projects gradients back into the epsilon-ball around original input.
    
    Args:
        model: Target PyTorch classification model
        image: Original clean image tensor (shape: [B, C, H, W])
        target: True labels corresponding to the image (shape: [B])
        epsilon: Maximum L-infinity perturbation boundary
        alpha: Step size for each iteration (default: 0.01)
        num_iter: Number of gradient steps to take (default: 10)
        
    Returns:
        perturbed_image: The optimized L-infinity bounded adversarial image.
    """
    # 1. Clone the original image as the baseline reference
    original_image = image.clone().detach()
    
    # 2. Clone the image to start our perturbation path
    perturbed_image = image.clone().detach()
    perturbed_image.requires_grad = True
    
    # Loss function (CrossEntropyLoss is standard for classification)
    loss_fn = nn.CrossEntropyLoss()
    
    # 3. Iterative optimization step
    for _ in range(num_iter):
        # Forward pass on current perturbed image
        output = model(perturbed_image)
        
        # Calculate loss
        loss = loss_fn(output, target)
        
        # Zero all existing gradients
        model.zero_grad()
        
        # Backward pass to find gradients with respect to perturbed_image
        loss.backward()
        
        # Get gradient data
        data_grad = perturbed_image.grad.data
        
        # Iterative update: step in the direction of maximizing loss
        with torch.no_grad():
            perturbed_image = perturbed_image + alpha * data_grad.sign()
            
            # Projection step: constrain perturbation to L-infinity norm ball around original
            eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
            perturbed_image = torch.clamp(original_image + eta, min=0, max=1).detach()
            
        perturbed_image.requires_grad = True
        
    return perturbed_image.detach()
