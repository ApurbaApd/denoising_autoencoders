import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class MedicalDenoisingDataset(Dataset):
    def __init__(self, root_dir, noise_factor=0.3, img_size=128):
        self.root_dir = root_dir
        self.noise_factor = noise_factor
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.dcm')
        # Filter for valid files only
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                      if f.lower().endswith(valid_exts)]
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img_path = self.files[idx]
            clean = Image.open(img_path).convert('L')
            clean = self.transform(clean)
            
            # Gaussian Noise
            noisy = clean + self.noise_factor * torch.randn(*clean.shape)
            noisy = torch.clamp(noisy, 0., 1.)
            return noisy, clean
        except Exception as e:
            # Return dummy if file is corrupted
            return torch.zeros(1, 128, 128), torch.zeros(1, 128, 128)