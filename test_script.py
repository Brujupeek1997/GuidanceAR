import os

import cv2
from ultralytics import YOLO


def detect_screwdriver(model_path, test_images_folder, output_folder, screwdriver_class_id=0, confidence_threshold=0.5):
    # Load the YOLO model
    model = YOLO(model_path)

    os.makedirs(output_folder, exist_ok=True)

    for image_name in os.listdir(test_images_folder):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(test_images_folder, image_name)
            image = cv2.imread(image_path)

            results = model.predict(source=image, save=False)

            for result in results:
                boxes = result.boxes.data.cpu().numpy()

                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box

                    if int(cls) == screwdriver_class_id and conf >= confidence_threshold:
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f'Screwdriver: {conf:.2f}'
                        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                    2)

            output_image_path = os.path.join(output_folder, f'detected_{image_name}')
            cv2.imwrite(output_image_path, image)
            print(f'Saved detected image to {output_image_path}')


if __name__ == "__main__":
    model_path = "runs/detect/train16/weights/best.pt"
    test_images_folder = "data/images/test"
    output_folder = "test_results"
    screwdriver_class_id = 0
    confidence_threshold = 0.5

    detect_screwdriver(model_path, test_images_folder, output_folder, screwdriver_class_id, confidence_threshold)
