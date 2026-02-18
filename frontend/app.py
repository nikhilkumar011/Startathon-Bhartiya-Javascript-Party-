import os
import io
import logging
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image


MODEL_PATH = os.getenv("MODEL_PATH", "best2.pt")
DEVICE = "cuda" if os.environ.get("USE_CUDA") == "1" else "cpu"

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

logger.info(f"Loading YOLO model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)
logger.info("Model loaded successfully!")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "model": MODEL_PATH
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        image = Image.open(file.stream).convert("RGB")
        image_np = np.array(image)

        results = model(image_np, device=DEVICE)

        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        success, buffer = cv2.imencode(".png", annotated_frame)
        if not success:
            return jsonify({"error": "Image encoding failed"}), 500

        return send_file(
            io.BytesIO(buffer),
            mimetype="image/png"
        )

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
