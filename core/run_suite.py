import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Import local modules
from model import get_model
from attack import fgsm_attack, pgd_attack
from defense import (
    spatial_smoothing, 
    jpeg_compression_simulation, 
    bit_depth_reduction,
    adversarial_training_epoch
)

def generate_synthetic_data(dataset_name: str, num_samples: int = 400):
    """Generates synthetic, structurally-coherent image data if torchvision datasets cannot be downloaded."""
    print(f"Generating synthetic structured data representing '{dataset_name.upper()}' for local offline execution...")
    np.random.seed(42)
    
    if dataset_name.lower() == "mnist":
        channels, height, width = 1, 28, 28
    else:
        channels, height, width = 3, 32, 32
        
    # Generate structured images (e.g. geometric blobs, digits, or CIFAR-like channels)
    data = []
    labels = []
    
    for i in range(num_samples):
        # Create an image canvas
        img = np.zeros((channels, height, width), dtype=np.float32)
        label = i % 10
        
        # Add basic structured geometric patterns matching the label to make it learnable
        # Create a circle/rectangle blob with varying positions depending on label
        center_x = width // 2 + int(5 * np.sin(label))
        center_y = height // 2 + int(5 * np.cos(label))
        radius = 5 + (label % 3)
        
        for c in range(channels):
            # Base gradients
            for y in range(height):
                for x in range(width):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist <= radius:
                        img[c, y, x] = 0.8 + 0.2 * np.sin(c + x)
                    else:
                        img[c, y, x] = 0.1 * np.cos(y)
                        
        # Add a tiny bit of Gaussian noise
        img += np.random.normal(0, 0.05, img.shape).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        
        data.append(img)
        labels.append(label)
        
    tensor_x = torch.tensor(np.array(data))
    tensor_y = torch.tensor(np.array(labels), dtype=torch.long)
    return TensorDataset(tensor_x, tensor_y)


def main():
    print("=========================================================================")
    print("Adversarial Attacks and Defense Mechanisms - Deep Learning Evaluation Suite")
    print("=========================================================================")
    
    # Configuration
    dataset = "mnist"  # or cifar10
    epochs = 5
    batch_size = 64
    learning_rate = 0.001
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing environment: {device}")
    
    # 1. Setup Assets Directory Paths
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_img_dir = os.path.join(workspace_root, "assets", "images")
    os.makedirs(assets_img_dir, exist_ok=True)
    
    # 2. Loading Datasets
    try:
        if dataset.lower() == "mnist":
            print("Attempting to load real MNIST data via Torchvision...")
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            # Use larger subset for high training accuracy (e.g. 5000 train, 1000 test)
            train_dataset = torch.utils.data.Subset(train_dataset, range(5000))
            test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
        else:
            print("Attempting to load real CIFAR-10 data via Torchvision...")
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            train_dataset = torch.utils.data.Subset(train_dataset, range(5000))
            test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
    except Exception as e:
        print(f"Could not download standard datasets: {e}")
        train_dataset = generate_synthetic_data(dataset, num_samples=800)
        test_dataset = generate_synthetic_data(dataset, num_samples=200)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Initialize Models
    print("\nInitializing Standard CNN Model...")
    standard_model = get_model(dataset).to(device)
    optimizer_std = optim.Adam(standard_model.parameters(), lr=learning_rate)
    
    print("Initializing Robust (Adversarially Trained) CNN Model...")
    robust_model = get_model(dataset).to(device)
    optimizer_robust = optim.Adam(robust_model.parameters(), lr=learning_rate)
    
    loss_fn = nn.CrossEntropyLoss()
    
    # 4. Train Standard Model
    print("\n--- Training Standard CNN Model ---")
    for epoch in range(1, epochs + 1):
        standard_model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer_std.zero_grad()
            outputs = standard_model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer_std.step()
            
            epoch_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        acc = correct / total
        avg_loss = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")
        
    # 5. Train Robust Model (Adversarial Training using online FGSM with epsilon=0.15)
    print("\n--- Training Robust CNN Model (Adversarial Training, Epsilon=0.15) ---")
    for epoch in range(1, epochs + 1):
        avg_loss = adversarial_training_epoch(
            model=robust_model, 
            dataloader=train_loader, 
            optimizer=optimizer_robust, 
            loss_fn=loss_fn, 
            epsilon=0.15, 
            device=device
        )
        # Evaluate training epoch accuracy
        robust_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = robust_model(images)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = correct / total
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")

    # 6. Comprehensive Attack & Defense Benchmarking
    print("\n--- Benchmarking Attack & Defense Performance across Epsilon Range ---")
    standard_model.eval()
    robust_model.eval()
    
    results = {
        "metadata": {
            "dataset": dataset,
            "architecture": "SimpleCNN (3 Conv, 2 FC layers)",
            "standard_epochs": epochs,
            "robust_epochs": epochs,
            "adversarial_train_epsilon": 0.15
        },
        "epsilons": epsilons,
        "standard_model": {
            "clean_accuracy": 0.0,
            "adversarial_accuracies": [],
            "smoothing_accuracies": [],
            "jpeg_accuracies": [],
            "bit_depth_accuracies": []
        },
        "robust_model": {
            "clean_accuracy": 0.0,
            "adversarial_accuracies": []
        }
    }
    
    # Helper to count accuracy on clean data
    def evaluate_clean_acc(model):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total

    results["standard_model"]["clean_accuracy"] = evaluate_clean_acc(standard_model)
    results["robust_model"]["clean_accuracy"] = evaluate_clean_acc(robust_model)
    
    # Benchmark each epsilon
    for eps in epsilons:
        print(f"Evaluating epsilon = {eps:.2f}...")
        
        # 1. Standard model under attack (FGSM)
        correct_std = 0
        correct_smooth = 0
        correct_jpeg = 0
        correct_bit = 0
        
        # 2. Robust model under attack (FGSM)
        correct_robust = 0
        
        total = 0
        
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            total += targets.size(0)
            
            # --- Standard Model Attack Generation ---
            images.requires_grad = True
            outputs_std = standard_model(images)
            loss = loss_fn(outputs_std, targets)
            standard_model.zero_grad()
            loss.backward()
            data_grad = images.grad.data
            
            # Generate FGSM perturbed images
            perturbed_std = fgsm_attack(images, eps, data_grad).detach()
            
            # Evaluate Standard Model on Adversarial Images
            with torch.no_grad():
                pred_std = standard_model(perturbed_std).max(1)[1]
                correct_std += pred_std.eq(targets).sum().item()
            
            # Evaluate Preprocessing Defenses on standard model adversarial inputs
            with torch.no_grad():
                # Filter A: Spatial Smoothing
                smoothed = spatial_smoothing(perturbed_std, kernel_size=3)
                pred_smooth = standard_model(smoothed).max(1)[1]
                correct_smooth += pred_smooth.eq(targets).sum().item()
                
                # Filter B: JPEG Compression
                jpegified = jpeg_compression_simulation(perturbed_std, quality=50)
                pred_jpeg = standard_model(jpegified).max(1)[1]
                correct_jpeg += pred_jpeg.eq(targets).sum().item()
                
                # Filter C: Bit depth reduction (to 3 bits)
                quantized = bit_depth_reduction(perturbed_std, bits=3)
                pred_bit = standard_model(quantized).max(1)[1]
                correct_bit += pred_bit.eq(targets).sum().item()
                
            # --- Robust Model Attack Generation ---
            # Create a separate attack explicitly against the robust model
            images_robust = images.clone().detach()
            images_robust.requires_grad = True
            outputs_robust = robust_model(images_robust)
            loss_robust = loss_fn(outputs_robust, targets)
            robust_model.zero_grad()
            loss_robust.backward()
            data_grad_robust = images_robust.grad.data
            
            perturbed_robust = fgsm_attack(images, eps, data_grad_robust).detach()
            
            with torch.no_grad():
                pred_robust = robust_model(perturbed_robust).max(1)[1]
                correct_robust += pred_robust.eq(targets).sum().item()
                
        # Record stats for current epsilon
        results["standard_model"]["adversarial_accuracies"].append(correct_std / total)
        results["standard_model"]["smoothing_accuracies"].append(correct_smooth / total)
        results["standard_model"]["jpeg_accuracies"].append(correct_jpeg / total)
        results["standard_model"]["bit_depth_accuracies"].append(correct_bit / total)
        results["robust_model"]["adversarial_accuracies"].append(correct_robust / total)
        
    # Save statistics results to results.json in workspace root
    results_json_path = os.path.join(workspace_root, "results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nStats exported successfully to: {results_json_path}")
    
    # 7. Generate Visualization Grids for Frontend Display
    print("\nSaving high-fidelity side-by-side adversarial comparison grids...")
    images, targets = next(iter(test_loader))
    images, targets = images.to(device), targets.to(device)
    
    # Choose index 0 as sample
    idx = 0
    sample_img = images[idx:idx+1]
    sample_target = targets[idx:idx+1]
    
    sample_img.requires_grad = True
    out = standard_model(sample_img)
    loss = loss_fn(out, sample_target)
    standard_model.zero_grad()
    loss.backward()
    grad = sample_img.grad.data
    
    eps_vis = 0.15
    sign_grad = grad.sign()
    perturbed_vis = fgsm_attack(sample_img, eps_vis, grad).detach()
    jpeg_vis = jpeg_compression_simulation(perturbed_vis, quality=50).detach()
    
    # Helper to convert tensor image back to PIL and save
    def save_tensor_img(tensor, filename):
        t = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        t = np.clip(t, 0.0, 1.0)
        t = (t * 255).astype(np.uint8)
        if t.shape[2] == 1:
            t = t.squeeze(2)
            img = Image.fromarray(t, mode='L')
        else:
            img = Image.fromarray(t, mode='RGB')
        
        path = os.path.join(assets_img_dir, filename)
        img.save(path)
        print(f"Saved: {path}")

    save_tensor_img(sample_img.detach(), "sample_original.png")
    
    # Save gradient map scaled for visibility
    grad_scaled = (sign_grad.squeeze(0).cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
    grad_scaled = (grad_scaled * 255).astype(np.uint8)
    if grad_scaled.shape[2] == 1:
        grad_scaled = grad_scaled.squeeze(2)
        grad_img = Image.fromarray(grad_scaled, mode='L')
    else:
        grad_img = Image.fromarray(grad_scaled, mode='RGB')
    grad_path = os.path.join(assets_img_dir, "sample_gradient.png")
    grad_img.save(grad_path)
    print(f"Saved: {grad_path}")
    
    save_tensor_img(perturbed_vis, "sample_adversarial.png")
    save_tensor_img(jpeg_vis, "sample_defended.png")
    
    print("\nEvaluation pipeline successfully finished.")
    print("=========================================================================")

if __name__ == "__main__":
    main()
