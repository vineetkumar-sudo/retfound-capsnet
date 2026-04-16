import cv2
import numpy as np


def preprocess_fundus(img_path: str, target_size: int = 224) -> np.ndarray:
    """Preprocess a fundus image: green channel extraction, denoising, CLAHE enhancement.

    Args:
        img_path: Path to the fundus image.
        target_size: Output spatial dimension (square).

    Returns:
        Enhanced image as uint8 array with shape (target_size, target_size, 3).
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    green = img[:, :, 1]
    green = cv2.medianBlur(green, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    enhanced = cv2.resize(enhanced, (target_size, target_size))
    return np.stack([enhanced] * 3, axis=-1)
