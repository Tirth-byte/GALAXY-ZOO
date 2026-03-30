"""
Galaxy Image Classification - Configuration
=============================================
Central configuration for all training and prediction parameters.
"""

import os

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
NEW_IMAGES_DIR = os.path.join(BASE_DIR, "new_images")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
TRAINING_HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.png")

# ─── Model Parameters ──────────────────────────────────────────────────────────
IMAGE_SIZE = 224          # EfficientNet-B0 input size
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 15
EARLY_STOP_PATIENCE = 5
DROPOUT_RATE = 0.3

# ─── Dataset ────────────────────────────────────────────────────────────────────
NUM_CLASSES = 7
VAL_SPLIT = 0.2          # 80% train / 20% validation
RANDOM_SEED = 42

# ─── Galaxy Morphology Classes ──────────────────────────────────────────────────
CLASS_LABELS = {
    0: "Completely Round Smooth",
    1: "In-Between Smooth",
    2: "Cigar-Shaped Smooth",
    3: "Edge-On Disk",
    4: "Barred Spiral",
    5: "Unbarred Spiral",
    6: "Irregular / Merger",
}

CLASS_DESCRIPTIONS = {
    0: "Elliptical galaxy with a perfectly spherical shape, no visible disk or spiral structure.",
    1: "Elliptical galaxy, slightly elongated, smooth texture without spiral arms.",
    2: "Highly elongated elliptical galaxy with a cigar-like shape.",
    3: "Disk galaxy viewed edge-on, appearing as a narrow band of light.",
    4: "Spiral galaxy with a prominent central bar structure connecting the spiral arms.",
    5: "Spiral galaxy with arms emerging directly from the central bulge (no bar).",
    6: "Galaxy with irregular shape, often a result of gravitational interaction or merger.",
}

# ─── ImageNet Normalization Stats ───────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Create required directories ───────────────────────────────────────────────
for d in [DATA_DIR, TRAIN_DIR, TEST_DIR, NEW_IMAGES_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)
