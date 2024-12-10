import os
import cv2
from ultralytics import YOLO

model_path = "runs/detect/train2/weights/best.pt"
test_folder = "test"
output_folder = "test_results"

os.makedirs(output_folder, exist_ok=True)

model = YOLO(model_path)

image_extensions = (".jpg", ".jpeg", ".png")

for image_name in os.listdir(test_folder):
    if image_name.lower().endswith(image_extensions):
        image_path = os.path.join(test_folder, image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        results = model.predict(image, conf=0.15, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = box.conf[0]

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{model.names[class_id]} {confidence:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, image)
        print(f"Processed and saved: {output_path}")
