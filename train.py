import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.dataset import MedicalDenoisingDataset
from src.models import CAE, UNet, VAE, ResNetAE

def get_model(name, device):
    if name == 'cae': return CAE().to(device)
    if name == 'unet': return UNet().to(device)
    if name == 'vae': return VAE().to(device)
    if name == 'resnet': return ResNetAE().to(device)
    raise ValueError(f"Unknown model: {name}")

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training {args.model.upper()} on {device} ---")
    
    # 1. Dataset Split (80/10/10)
    full_ds = MedicalDenoisingDataset(args.data_path, noise_factor=args.noise)
    train_size = int(0.8 * len(full_ds))
    val_size = int(0.1 * len(full_ds))
    test_size = len(full_ds) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(full_ds, [train_size, val_size, test_size])
    
    # Save test set for later analysis
    os.makedirs("data/splits", exist_ok=True)
    torch.save(test_ds, "data/splits/test_split.pt")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2. Setup
    model = get_model(args.model, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    save_dir = f"saved_models/{args.model}"
    os.makedirs(save_dir, exist_ok=True)

    # 3. Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            
            if args.model == 'vae':
                recon, mu, logvar = model(noisy)
                mse = criterion(recon, clean)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / noisy.numel()
                loss = mse + 0.001 * kld
            else:
                recon = model(noisy)
                loss = criterion(recon, clean)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 4. Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                if args.model == 'vae':
                    recon, _, _ = model(noisy)
                else:
                    recon = model(noisy)
                val_loss += criterion(recon, clean).item()
        
        avg_train = train_loss/len(train_loader)
        avg_val = val_loss/len(val_loader)
        scheduler.step(avg_val)
        
        print(f"Epoch {epoch+1} | Train: {avg_train:.5f} | Val: {avg_val:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save Best
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print(f"   --> Saved Best Model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=['cae', 'unet', 'vae', 'resnet'])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise", type=float, default=0.3)
    args = parser.parse_args()
    train(args)