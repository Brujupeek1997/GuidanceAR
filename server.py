from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS
import os
import uuid

app = Flask(__name__)
CORS(app)

model_path = 'runs/detect/train9/weights/best.pt'
model = YOLO(model_path)

processed_images_folder = 'processed_images'
os.makedirs(processed_images_folder, exist_ok=True)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        nparr = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(frame)

        detections = []

        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                label = f"{result.names[int(cls)]} {conf:.2f}"

                detections.append({
                    'name': result.names[int(cls)],
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': float(conf)
                })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        random_name = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(processed_images_folder, random_name)
        cv2.imwrite(save_path, frame)

        return jsonify({'detections': detections, 'saved_image': save_path})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/ping', methods=['GET'])
def ping():
    try:
        return jsonify({'message': 'Server is running'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    context = ('cert.pem', 'key.pem')
    app.run(host='0.0.0.0', port=20398, debug=True, ssl_context=context)
