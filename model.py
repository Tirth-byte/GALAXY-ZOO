"""
Galaxy Image Classification - Model Module
============================================
EfficientNet-B0 with custom classification head
for galaxy morphology classification.
"""

import torch
import torch.nn as nn
import timm

import config


class GalaxyClassifier(nn.Module):
    """
    Galaxy Morphology Classifier using EfficientNet-B0 backbone.

    Architecture:
        EfficientNet-B0 (pretrained on ImageNet)
        → Adaptive Average Pool
        → Dropout(0.3)
        → FC(1280, 512) + BatchNorm + ReLU + Dropout(0.2)
        → FC(512, num_classes)
    """

    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True):
        super().__init__()

        # Load pretrained EfficientNet-B0 backbone
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,        # Remove default head
            global_pool="avg",    # Keep global average pooling
        )

        # Freeze early layers (first 4 of 7 blocks) for transfer learning
        self._freeze_early_layers()

        # Get feature dimension from backbone
        feat_dim = self.backbone.num_features  # 1280 for EfficientNet-B0

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

        # Initialize classifier weights
        self._init_classifier()

    def _freeze_early_layers(self):
        """Freeze the first 4 blocks of EfficientNet for transfer learning."""
        # Freeze stem
        for param in self.backbone.conv_stem.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False

        # Freeze first 4 of 7 blocks
        for i, block in enumerate(self.backbone.blocks):
            if i < 4:
                for param in block.parameters():
                    param.requires_grad = False

    def _init_classifier(self):
        """Initialize classifier head weights with Kaiming normal."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass: backbone features → classifier head."""
        features = self.backbone(x)         # (B, 1280)
        logits = self.classifier(features)  # (B, num_classes)
        return logits

    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning (optional second training phase)."""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        """Return count of trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def build_model(num_classes=config.NUM_CLASSES, pretrained=True):
    """
    Factory function to build and return the galaxy classifier model.

    Args:
        num_classes: Number of galaxy morphology classes
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        model: GalaxyClassifier instance
        device: torch device (cuda/mps/cpu)
    """
    # Select best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  🍎  Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("  💻  Using CPU")

    model = GalaxyClassifier(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    trainable, total = model.get_trainable_params()
    print(f"  📐  Model parameters: {trainable:,} trainable / {total:,} total")
    print(f"      ({trainable / total * 100:.1f}% trainable — transfer learning)")

    return model, device


# ─── Main: Model Info ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║    🔭  Galaxy Classifier — Model Architecture            ║")
    print("╚" + "═" * 58 + "╝")
    print()

    model, device = build_model()
    print(f"\n  Model summary:")
    print(f"    Backbone: EfficientNet-B0 (ImageNet pretrained)")
    print(f"    Head: 1280 → 512 → {config.NUM_CLASSES}")
    print(f"    Device: {device}")
    print()
