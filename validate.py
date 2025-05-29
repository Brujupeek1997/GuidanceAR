import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

model_path = 'runs/detect/train4/weights/best.pt'
model = YOLO(model_path)

test_image_path = 'test/test7.jpg'

results = model(test_image_path)

img = cv2.imread(test_image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detected_classes = set()

for result in results:
    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"{result.names[int(cls)]} {conf:.2f}"

        detected_classes.add(result.names[int(cls)])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

required_classes = {"capac_scaled", "leg1_scaled", "bod_scaled", "body_scale", "flat1_scaled", "ingot_scaled"}
if required_classes.issubset(detected_classes):
    message = "Hey, I identified that you have all pieces to build the following LEGO building: Star Wars Robot"
    print(message)
    (text_width, text_height), baseline = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (10, 10), (10 + text_width, 10 + text_height + baseline), (0, 255, 0), -1)
    cv2.putText(img, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Display the result
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.axis('off')
plt.show()
print(f"Detected classes: {detected_classes}")