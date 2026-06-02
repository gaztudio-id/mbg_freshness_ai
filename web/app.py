import os
import base64
import cv2
import numpy as np
import tensorflow as tf
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, render_template, url_for

app = Flask(__name__)

# Memuat Model global
model_path = 'mbg_freshness_model.h5'
if not os.path.exists(model_path):
    # Jika dijalankan dari dalam folder 'web', cari model di folder induk (root)
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mbg_freshness_model.h5')

try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model berhasil dimuat dari: {model_path}")
except Exception as e:
    print(f"Error memuat model dari {model_path}: {e}")
    model = None

# Template HTML Gabungan (Clean & Minimalist UI)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    return jsonify({"status": "ready" if model is not None else "error"})

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500
        
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image data"}), 400
            
        # Decode base64
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocessing model
        img_resized = cv2.resize(frame, (224, 224))
        # Pastikan format RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_scaled = img_rgb / 255.0
        img_expanded = np.expand_dims(img_scaled, axis=0)
        
        # Prediksi
        prediction = model.predict(img_expanded, verbose=0)[0][0]
        prediction_val = float(prediction)
        
        segar_prob = (1.0 - prediction_val) * 100
        tidak_prob = prediction_val * 100
        
        label = "segar" if prediction_val < 0.5 else "tidak_segar"
        
        # Logika Kelayakan Piecewise Linear (Ghaswul Fikri Fadhillah)
        if prediction_val < 0.5:
            kelayakan = 100.0 - (prediction_val * 60)
            kelayakan_status = "LAYAK KONSUMSI (Sangat Segar & Aman)"
        elif prediction_val < 0.7:
            kelayakan = 70.0 - ((prediction_val - 0.5) * 100)
            kelayakan_status = "LAYAK KONSUMSI (Batas Aman - Konsumsi Segera!)"
        else:
            kelayakan = 50.0 - (((prediction_val - 0.7) / 0.3) * 50)
            kelayakan_status = "TIDAK LAYAK KONSUMSI (Busuk/Rusak - Jangan Diolah!)"
            
        kelayakan = max(0.0, min(100.0, kelayakan))
        
        return jsonify({
            "success": True,
            "label": label,
            "segar": segar_prob,
            "tidak": tidak_prob,
            "kelayakan": kelayakan,
            "kelayakan_status": kelayakan_status
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("Memulai server web MBG AI...")
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
