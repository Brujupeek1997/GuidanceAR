import os

import cv2
import numpy as np


def find_polygon_points(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img.shape[0] * img.shape[1]
    contours = [c for c in contours if 500 < cv2.contourArea(c) < img_area * 0.9]

    if not contours:
        raise ValueError(f"No contours found in image: {image_path}")

    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.0025 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    img_h, img_w = img.shape[:2]

    polygon_points = []
    for point in approx_polygon:
        x, y = point[0]
        polygon_points.extend([x / img_w, y / img_h])

    return polygon_points


def draw_polygon(image_path, polygon_points, output_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    pixel_points = [(int(x * w), int(y * h)) for x, y in zip(polygon_points[::2], polygon_points[1::2])]

    cv2.polylines(img, [np.array(pixel_points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imwrite(output_path, img)


def process_dataset(base_path):
    for split in ['train', 'val']:
        images_dir = os.path.join(base_path, 'data', 'images', split)
        labels_dir = os.path.join(base_path, 'data', 'labels', split)
        vis_dir = os.path.join(base_path, 'data', 'visualization', split)

        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        for img_file in os.listdir(images_dir):
            if not img_file.endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, img_file.rsplit('.', 1)[0] + '.txt')
            vis_path = os.path.join(vis_dir, f"polygon_{img_file}")

            try:
                polygon_points = find_polygon_points(image_path)

                with open(label_path, 'w') as f:
                    f.write("0 " + " ".join([f"{coord:.6f}" for coord in polygon_points]) + "\n")

                draw_polygon(image_path, polygon_points, vis_path)

                print(f"Processed: {img_file}")
                print("Polygon points:", polygon_points)
                print("-" * 50)

            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")


def main():
    base_dir = os.getcwd()

    try:
        process_dataset(base_dir)
        print("Dataset processing completed successfully!")
        print("\nVisualization images have been saved to the 'data/visualization' directory.")
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")


if __name__ == "__main__":
    main()
