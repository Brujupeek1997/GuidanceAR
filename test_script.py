import os

import cv2
from ultralytics import YOLO


def highlight_screwdriver(model_path, test_images_folder, output_folder):
    model = YOLO(model_path)

    os.makedirs(output_folder, exist_ok=True)

    for image_name in os.listdir(test_images_folder):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(test_images_folder, image_name)
            image = cv2.imread(image_path)

            results = model.predict(source=image, save=False, save_txt=False)

            for result in results:
                mask = result.masks.data[0].cpu().numpy()

                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

                color = (0, 255, 0)
                highlighted_image = image.copy()
                highlighted_image[mask_resized > 0.5] = color

                output_image_path = os.path.join(output_folder, f"highlighted_{image_name}")
                cv2.imwrite(output_image_path, highlighted_image)
                print(f"Saved highlighted image to {output_image_path}")


if __name__ == "__main__":
    model_path = "runs/segment/train4/weights/best.pt"
    test_images_folder = "data/images/test"
    output_folder = "test_results/highlighted"

    highlight_screwdriver(model_path, test_images_folder, output_folder)
