from flask import Flask, render_template, request, jsonify
from detector import SentimentDetector
from sentiment_model.structs import EMOTION_DICT, EMOTION_DICT_TR
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Singleton detector - model bir kez yüklenir
detector = SentimentDetector()


def numpy_to_base64(img_np, fmt=".jpg"):
    """Numpy array'i base64 string'e çevirir."""
    _, buffer = cv2.imencode(fmt, img_np)
    return base64.b64encode(buffer).decode("utf-8")


def analyze_image(img_np):
    """Verilen numpy görüntüyü analiz eder ve sonuçları döndürür."""
    try:
        predicted_class, confidence, cropped_face, face_xywh, annotated_image = detector.apply(img_np)
    except Exception as e:
        return {"error": str(e)}, 400

    # Orijinal görüntüyü base64'e çevir
    original_b64 = numpy_to_base64(img_np)

    # Annotated image BGR->RGB dönüşümü (detector zaten yapmış olabilir)
    annotated_b64 = numpy_to_base64(annotated_image)

    # Cropped face
    cropped_b64 = numpy_to_base64(cropped_face)

    return {
        "original_image": original_b64,
        "annotated_image": annotated_b64,
        "cropped_face": cropped_b64,
        "predicted_class": predicted_class,
        "label_en": EMOTION_DICT.get(predicted_class, "Unknown"),
        "label_tr": EMOTION_DICT_TR.get(predicted_class, "Bilinmiyor"),
        "confidence": round(confidence, 4),
    }


# ── Routes ──────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("image.html")


@app.route("/sentiment_analysis/image")
def image_page():
    return render_template("image.html")


@app.route("/sentiment_analysis/webcam")
def webcam_page():
    return render_template("webcam.html")


# ── API Endpoints ───────────────────────────────────────


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Yüklenen görsel dosyasını analiz eder."""
    if "image" not in request.files:
        return jsonify({"error": "Görsel dosyası bulunamadı"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_np is None:
        return jsonify({"error": "Görsel okunamadı"}), 400

    result = analyze_image(img_np)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]

    return jsonify(result)


@app.route("/api/analyze-frame", methods=["POST"])
def api_analyze_frame():
    """Webcam'den gelen base64 frame'i analiz eder."""
    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "Frame verisi bulunamadı"}), 400

    # Base64 decode
    try:
        frame_data = data["frame"]
        # data:image/jpeg;base64, prefix'ini temizle
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]

        img_bytes = base64.b64decode(frame_data)
        img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"error": "Frame decode edilemedi"}), 400

    if img_np is None:
        return jsonify({"error": "Frame okunamadı"}), 400

    result = analyze_image(img_np)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
