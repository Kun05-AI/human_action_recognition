from pathlib import Path


# =========================
# Project directories
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
MODEL_DIR = PROJECT_ROOT / "models"


# =========================
# YOLO26 Pose
# =========================

MODEL_PATH = MODEL_DIR / "yolo26m-pose.pt"

DEVICE = 0
IMAGE_SIZE = 640

PERSON_CLASS_ID = 0

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.50


# =========================
# Visualization
# =========================

BOX_THICKNESS = 2
SKELETON_THICKNESS = 2


# =========================
# Pose Buffer
# =========================

POSE_BUFFER_SIZE = 30