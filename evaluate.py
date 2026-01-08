import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.models import CAE, UNet, VAE, ResNetAE
from src.metrics import get_metrics_batch
import os

def evaluate_all():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # 2. Check for Apple GPU (MacOS)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    # 3. Fallback to CPU
    else:
        device = torch.device('cpu')
    
    # 1. Load Test Data
    if not os.path.exists("data/splits/test_split.pt"):
        print("Run train.py first to generate test data!")
        return
        
    test_ds = torch.load("data/splits/test_split.pt")
    loader = DataLoader(test_ds, batch_size=8, shuffle=True) # Batch of 8 for visuals
    noisy, clean = next(iter(loader))
    noisy, clean = noisy.to(device), clean.to(device)
    
    # 2. Load Models
    models = {
        'CAE': CAE(),
        'UNet': UNet(),
        'VAE': VAE(),
        'ResNet': ResNetAE()
    }
    
    results = {}
    print(f"{'Model':<10} | {'PSNR':<10} | {'SSIM':<10} | {'EPI':<10}")
    print("-" * 50)
    
    for name, model in models.items():
        path = f"saved_models/{name.lower()}/best_model.pth"
        if not os.path.exists(path):
            print(f"Skipping {name} (Weights not found)")
            continue
            
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        
        with torch.no_grad():
            if name == 'VAE':
                out, _, _ = model(noisy)
            else:
                out = model(noisy)
            
            metrics = get_metrics_batch(clean, out)
            print(f"{name:<10} | {metrics['PSNR']:.2f} dB   | {metrics['SSIM']:.4f}     | {metrics['EPI']:.4f}")
            #save output for visualization
            # os.makedirs("results", exist_ok=True)
            # torch.save(out, f"results/{name}_metrics.pdf")
            
            results[name] = out
            
    # 3. Visualization
    fig, axes = plt.subplots(1, len(results) + 2, figsize=(18, 4))
    
    # Plot Original & Noisy
    axes[0].imshow(clean[0].squeeze().cpu(), cmap='gray'); axes[0].set_title("Ground Truth"); axes[0].axis('off')
    axes[1].imshow(noisy[0].squeeze().cpu(), cmap='gray'); axes[1].set_title("Noisy Input"); axes[1].axis('off')
    
    # Plot Models
    for i, (name, out_tensor) in enumerate(results.items()):
        ax = axes[i+2]
        ax.imshow(out_tensor[0].squeeze().cpu().numpy(), cmap='gray')
        ax.set_title(f"{name}")
        ax.axis('off')
        
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/final_benchmark.png")
    print("\nBenchmark saved to results/final_benchmark.png")

if __name__ == "__main__":
    evaluate_all()