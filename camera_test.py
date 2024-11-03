import cv2
from ultralytics import YOLO


def camera_detection(model_path, screwdriver_class_id=0, confidence_threshold=0.5):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        results = model.predict(source=frame, save=False)

        for result in results:
            boxes = result.boxes.data.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2, conf, cls = box

                if int(cls) == screwdriver_class_id and conf >= confidence_threshold:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f'Screwdriver: {conf:.2f}'
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('YOLO Camera Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "runs/detect/train16/weights/best.pt"
    screwdriver_class_id = 0
    confidence_threshold = 0.5

    camera_detection(model_path, screwdriver_class_id, confidence_threshold)
