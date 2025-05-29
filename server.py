import time

from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS
import os
import uuid

app = Flask(__name__)
CORS(app)


loaded_models = {}

default_model_path = 'train9/weights/best.pt'
loaded_models[default_model_path] = YOLO(default_model_path)

model_mapping = {
    "Hammer": "train20/weights/best.pt",
    "ScrewDriver": "train30/weights/best.pt",
    "wrench": "train31/weights/best.pt",
    # "knife": "train11/weights/best.pt",
    # "tapemeasure": "train10/weights/best.pt",
     "scissors": "train32/weights/best.pt",
}

tools_set = {"Hammer", "ScrewDriver", "wrench", "not yet trained", "tapemeasure", "scissors"}

model_map_values = {
    default_model_path: 0.85,
    "train20/weights/best.pt": 0.82,
    "train30/weights/best.pt": 0.80,
    "train31/weights/best.pt": 0.78,
   # "train11/weights/best.pt": 0.81,
   # "train10/weights/best.pt": 0.77,
    "train32/weights/best.pt": 0.83,
}

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        detection_filter = request.form.get('filter', '').strip()
        print("Detection filter:", detection_filter)

        if detection_filter in tools_set:
            model_path = model_mapping.get(detection_filter, default_model_path)
            if model_path == "not yet trained":
                return jsonify({'error': 'Model for this tool is not yet trained'}), 400
            if model_path not in loaded_models:
                loaded_models[model_path] = YOLO(model_path)
            model_to_use = loaded_models[model_path]
        else:
            model_path = default_model_path
            model_to_use = loaded_models[default_model_path]

        image_file = request.files['image']
        nparr = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        start_time = time.time()
        results = model_to_use(frame)
        inference_time_ms = (time.time() - start_time) * 1000

        detections = []
        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    'name': result.names[int(cls)],
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': float(conf)
                })

        random_name = f"{uuid.uuid4().hex}.jpg"
        processed_images_folder = 'processed_images'
        os.makedirs(processed_images_folder, exist_ok=True)
        save_path = os.path.join(processed_images_folder, random_name)
        cv2.imwrite(save_path, frame)

        map_value = model_map_values.get(model_path, 0.0)

        response = jsonify({
            'detections': detections,
            'saved_image': save_path,
            'map': map_value
        })
        response.headers['X-Process-Time'] = str(round(inference_time_ms, 2))
        return response


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
    app.run(host='0.0.0.0', port=20398, debug=True, ssl_context=context, threaded=True)
