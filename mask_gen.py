import os

import cv2
import numpy as np

image_dir_train = "./data/images/train"
image_dir_val = "./data/images/val"
mask_dir_train = "./data/masks/train"
mask_dir_val = "./data/masks/val"

os.makedirs(mask_dir_train, exist_ok=True)
os.makedirs(mask_dir_val, exist_ok=True)


def create_detailed_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    cv2.imwrite(mask_path, mask)


for filename in os.listdir(image_dir_train):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(image_dir_train, filename)
        mask_path = os.path.join(mask_dir_train, filename)
        create_detailed_mask(image_path, mask_path)

for filename in os.listdir(image_dir_val):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(image_dir_val, filename)
        mask_path = os.path.join(mask_dir_val, filename)
        create_detailed_mask(image_path, mask_path)

