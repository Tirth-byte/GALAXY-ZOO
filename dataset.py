"""
Galaxy Image Classification - Dataset Module
==============================================
Handles downloading, preprocessing, augmentation,
and loading of galaxy image datasets.
"""

import os
import shutil
from pathlib import Path

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config


# ─── Transforms ─────────────────────────────────────────────────────────────────

def get_train_transforms():
    """Augmented transforms for training: flips, rotation, jitter, erasing."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        transforms.RandomErasing(p=0.1),
    ])


def get_val_transforms():
    """Clean transforms for validation / inference."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


# ─── Galaxy Dataset Class ───────────────────────────────────────────────────────

class GalaxyDataset(Dataset):
    """PyTorch Dataset for galaxy images organized in class folders."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image and convert to RGB
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ─── Dataset Download & Organization ────────────────────────────────────────────

def download_dataset():
    """
    Download the Galaxy Zoo dataset from Kaggle using kagglehub.
    Organizes images into class folders under data/train/.

    Returns the path to the downloaded data.
    """
    print("=" * 60)
    print("  📡  Downloading Galaxy Zoo Dataset from Kaggle...")
    print("=" * 60)

    try:
        import kagglehub
        # Download the Galaxy10 DECals dataset (labeled galaxy morphology images)
        dataset_path = kagglehub.dataset_download("muhammadhananasghar/galaxy-zoo-dataset")
        print(f"  ✅  Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"  ⚠️  Kaggle download failed: {e}")
        print("  ℹ️  Checking for local data in data/ directory...")
        return None


def organize_dataset(source_path=None):
    """
    Organize downloaded galaxy images into class-based folder structure.
    If source_path is None, expects data already in data/train/<class_name>/ format.

    Returns True if data is ready to use.
    """
    # Check if data is already organized
    if _is_data_organized():
        class_counts = _count_classes()
        print("  ✅  Dataset already organized!")
        _print_class_summary(class_counts)
        return True

    if source_path is None:
        print("  ❌  No data found. Please either:")
        print("      1. Set up Kaggle credentials (kaggle.json)")
        print("      2. Manually place images in data/train/<class_name>/")
        print()
        print("  Expected folder structure:")
        for idx, name in config.CLASS_LABELS.items():
            folder = name.lower().replace(" ", "_").replace("/", "_")
            print(f"      data/train/{folder}/  →  {name}")
        return False

    # Try to organize from downloaded dataset
    print("  🔄  Organizing dataset into class folders...")
    _organize_from_download(source_path)

    if _is_data_organized():
        class_counts = _count_classes()
        _print_class_summary(class_counts)
        return True

    return False


def _is_data_organized():
    """Check if training data exists in expected folder structure."""
    if not os.path.exists(config.TRAIN_DIR):
        return False
    subdirs = [d for d in os.listdir(config.TRAIN_DIR)
               if os.path.isdir(os.path.join(config.TRAIN_DIR, d))]
    if len(subdirs) < 2:
        return False
    # Check at least some images exist
    total = sum(len([f for f in os.listdir(os.path.join(config.TRAIN_DIR, d))
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))])
                for d in subdirs)
    return total > 10


def _count_classes():
    """Count images per class folder."""
    counts = {}
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for d in sorted(os.listdir(config.TRAIN_DIR)):
        path = os.path.join(config.TRAIN_DIR, d)
        if os.path.isdir(path):
            count = len([f for f in os.listdir(path) if Path(f).suffix.lower() in exts])
            if count > 0:
                counts[d] = count
    return counts


def _print_class_summary(counts):
    """Pretty print class distribution."""
    total = sum(counts.values())
    print(f"\n  📊  Dataset Summary: {total} images across {len(counts)} classes")
    print("  " + "─" * 50)
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / max(counts.values()) * 20)
        pct = count / total * 100
        print(f"    {cls:30s} │ {count:5d} ({pct:5.1f}%) {bar}")
    print("  " + "─" * 50)


def _organize_from_download(source_path):
    """
    Organize images from a downloaded Kaggle dataset.
    Handles multiple possible folder structures.
    """
    source = Path(source_path)

    # Strategy 1: Dataset already has class folders
    for subdir in source.rglob("*"):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            images = [f for f in subdir.iterdir()
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}]
            if len(images) > 5:
                dest = os.path.join(config.TRAIN_DIR, subdir.name)
                os.makedirs(dest, exist_ok=True)
                for img in tqdm(images, desc=f"    Copying {subdir.name}", leave=False):
                    shutil.copy2(str(img), os.path.join(dest, img.name))


# ─── Data Loader Factory ────────────────────────────────────────────────────────

def get_class_folders():
    """Get sorted list of class folder names from training directory."""
    if not os.path.exists(config.TRAIN_DIR):
        return []
    return sorted([d for d in os.listdir(config.TRAIN_DIR)
                   if os.path.isdir(os.path.join(config.TRAIN_DIR, d))])


def prepare_data():
    """
    Prepare training and validation data loaders.

    Returns:
        train_loader, val_loader, class_names, num_classes
    """
    class_names = get_class_folders()
    num_classes = len(class_names)

    if num_classes < 2:
        raise ValueError(
            f"Need at least 2 class folders in {config.TRAIN_DIR}, found {num_classes}.\n"
            f"Run `python dataset.py` first to download and organize the dataset."
        )

    print(f"\n  🏷️  Found {num_classes} classes: {class_names}")

    # Collect all image paths and labels
    image_paths = []
    labels = []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(config.TRAIN_DIR, class_name)
        for fname in os.listdir(class_dir):
            if Path(fname).suffix.lower() in exts:
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)

    print(f"  📁  Total images: {len(image_paths)}")

    # Stratified train/val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=config.VAL_SPLIT,
        stratify=labels,
        random_state=config.RANDOM_SEED,
    )

    print(f"  🔀  Train: {len(train_paths)} | Validation: {len(val_paths)}")

    # Create datasets
    train_dataset = GalaxyDataset(train_paths, train_labels, transform=get_train_transforms())
    val_dataset = GalaxyDataset(val_paths, val_labels, transform=get_val_transforms())

    # Weighted sampler to handle class imbalance
    class_counts = [0] * num_classes
    for label_idx in train_labels:
        class_counts[label_idx] += 1
    class_weights = [1.0 / c if c > 0 else 0 for c in class_counts]
    sample_weights = [class_weights[label_idx] for label_idx in train_labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, class_names, num_classes


# ─── Main: Download & Organize ──────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║    🔭  Galaxy Zoo Dataset Preparation                    ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # Step 1: Try to download
    source = download_dataset()

    # Step 2: Organize into class folders
    success = organize_dataset(source)

    if success:
        print("\n  🎉  Dataset is ready for training!")
        print("  💡  Run `python train.py` to start training.\n")
    else:
        print("\n  ⚠️  Dataset setup incomplete.")
        print("  💡  Please organize your galaxy images manually:")
        print(f"      Place images in: {config.TRAIN_DIR}/<class_name>/")
        print("      Each subfolder = one galaxy class.\n")
