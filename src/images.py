import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # type:ignore
import numpy as np


def open_image(image_fn: str) -> np.ndarray:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img: np.ndarray = cv2.imread(image_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(f"Read image data type of {img.dtype}")
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def convert_to_hsv(image: np.ndarray) -> np.ndarray:
    assert len(image.shape) == 3
    out = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    out[:, :, 0] /= 360.0
    print(np.max(out[:, :, 0]))
    return out
