import tempfile
import os
# Use a pipeline as a high-level helper
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS
# from pyngrok import ngrok, conf

pipe = pipeline("automatic-speech-recognition", model="Oriserve/Whisper-Hindi2Hinglish-Prime")




app = Flask(__name__)
CORS(app)

# ✅ Set ngrok auth token
# conf.get_default().auth_token = "your-ngrok-auth-token"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)

    try:
        result = pipe(tmp_path)
        os.remove(tmp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔥 Start ngrok tunnel and run app
# public_url = ngrok.connect(5000)
# print("🔥 Public API URL:", public_url)
# app.run(port=5000)

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "Hello from Flask on Render!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
