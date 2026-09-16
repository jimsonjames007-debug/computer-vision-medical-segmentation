import os
import ssl
import urllib.request
import zipfile
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

DATASET_URL = 'https://datasets.simula.no/downloads/kvasir-seg.zip'
DATASET_ZIP = 'kvasir-seg.zip'
DATASET_DIR = 'Kvasir-SEG'


# -------------------------------------------------------------------------
# 1. Dataset Downloading & Extraction
# -------------------------------------------------------------------------
def download_and_extract_dataset(dataset_dir=DATASET_DIR, dataset_url=DATASET_URL, dataset_zip=DATASET_ZIP):
    """
    Downloads and extracts the Kvasir-SEG dataset if not already present.
    """
    if not os.path.exists(dataset_dir):
        print(f"Downloading Kvasir-SEG dataset from {dataset_url}...")
        ssl._create_default_https_context = ssl._create_unverified_context

        if not os.path.exists(dataset_zip):
            urllib.request.urlretrieve(dataset_url, dataset_zip)

        print("Extracting dataset...")
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall()
        print("Dataset ready.")
    else:
        print("Dataset already present.")


# -------------------------------------------------------------------------
# 2. PyTorch Dataset Definition
# -------------------------------------------------------------------------
class MedicalImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading medical images and corresponding binary masks.
    """
    def __init__(self, image_dir, mask_dir, img_size=(128, 128), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        else:
            default_img_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = default_img_transform(image)

        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask


# -------------------------------------------------------------------------
# 3. U-Net Architecture
# -------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for 2D Medical Image Segmentation.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Contracting Path)
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)

        # Decoder (Expanding Path)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        # Output Layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x3 = self.down2(x2)
        x4 = self.pool(x3)
        x5 = self.down3(x4)

        # Decoder with Skip Connections
        x = self.up1(x5)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)

        return torch.sigmoid(self.out_conv(x))


# -------------------------------------------------------------------------
# 4. Evaluation Metrics
# -------------------------------------------------------------------------
def dice_coeff(pred, target, smooth=1.0, threshold=0.5):
    """Computes Dice Similarity Coefficient (DSC)."""
    if threshold is not None:
        pred = (pred > threshold).float()

    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice


def iou_score(pred, target, smooth=1.0, threshold=0.5):
    """Computes Intersection over Union (IoU) / Jaccard Index."""
    if threshold is not None:
        pred = (pred > threshold).float()

    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    total = pred_flat.sum() + target_flat.sum()
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


# -------------------------------------------------------------------------
# 5. Result Visualization
# -------------------------------------------------------------------------
def visualize_predictions(model, dataloader, device, save_path='segmentation_results.png', num_samples=3):
    """
    Plots and saves comparison of Input Image, Ground Truth Mask, and Predicted Mask.
    """
    model.eval()
    images, masks = next(iter(dataloader))

    images = images.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        outputs = model(images)

    images = images.cpu()
    masks = masks.cpu()
    outputs = outputs.cpu()

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    num_samples = min(num_samples, images.size(0))
    fig, axs = plt.subplots(num_samples, 3, figsize=(10, 3.5 * num_samples))

    if num_samples == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(num_samples):
        img = inv_normalize(images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        gt_mask = masks[i].squeeze().numpy()
        pred_mask = (outputs[i].squeeze() > 0.5).numpy().astype(np.float32)

        axs[i, 0].imshow(img)
        axs[i, 0].set_title("Input Image", fontsize=12, fontweight='bold')
        axs[i, 0].axis("off")

        axs[i, 1].imshow(gt_mask, cmap='gray')
        axs[i, 1].set_title("Ground Truth Mask", fontsize=12, fontweight='bold')
        axs[i, 1].axis("off")

        axs[i, 2].imshow(pred_mask, cmap='gray')
        axs[i, 2].set_title("Predicted Mask (U-Net)", fontsize=12, fontweight='bold')
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to '{save_path}'.")
    plt.close()


# -------------------------------------------------------------------------
# 6. Training Pipeline
# -------------------------------------------------------------------------
def train(args):
    # 1. Download / Verify Dataset
    download_and_extract_dataset(args.data_dir)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"--> Using compute device: {device}")

    # 2. Setup Data Transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_dir = os.path.join(args.data_dir, 'images')
    mask_dir = os.path.join(args.data_dir, 'masks')

    dataset = MedicalImageDataset(img_dir, mask_dir, img_size=(args.img_size, args.img_size), transform=transform)
    print(f"--> Total images available in dataset: {len(dataset)}")

    if args.subset_size and args.subset_size < len(dataset):
        dataset_subset = Subset(dataset, range(args.subset_size))
        print(f"--> Using subset of {args.subset_size} samples.")
    else:
        dataset_subset = dataset
        print(f"--> Using full dataset of {len(dataset)} samples.")

    dataloader = DataLoader(dataset_subset, batch_size=args.batch_size, shuffle=True)

    # 3. Model, Loss, Optimizer
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("\nStarting Training Loop...")
    print("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0

        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice_coeff(outputs, masks).item()
            epoch_iou += iou_score(outputs, masks).item()

        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_iou = epoch_iou / num_batches

        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] | Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")

    print("=" * 60)
    print("Training finished successfully!\n")

    # 4. Save Model Weights
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or '.', exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"--> Model checkpoint saved to: {args.save_model}")

    # 5. Visualization
    print("Generating qualitative visualization...")
    visualize_predictions(model, dataloader, device, save_path=args.output_image)


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on Kvasir-SEG Medical Segmentation Dataset")
    parser.add_argument('--data_dir', type=str, default=DATASET_DIR, help="Path to Kvasir-SEG directory")
    parser.add_argument('--img_size', type=int, default=128, help="Image resize dimension (square)")
    parser.add_argument('--batch_size', type=int, default=8, help="DataLoader batch size")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument('--subset_size', type=int, default=100, help="Number of dataset samples to use (0 for full)")
    parser.add_argument('--save_model', type=str, default="unet_kvasir_seg.pth", help="Path to save trained weights")
    parser.add_argument('--output_image', type=str, default="segmentation_results.png", help="Path to save result visualization plot")
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA even if available")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
