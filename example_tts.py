from flask import Flask, request, send_file, jsonify
import torchaudio as ta
import torch
from datetime import datetime
from io import BytesIO
from chatterbox.tts import ChatterboxTTS

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)

@app.route('/generate', methods=['POST'])
def generate_audio():
    try:
        data = request.json
        text = data.get('text')
        prompt = data.get('prompt')  # Optional custom voice

        if not text:
            return jsonify({'error': 'Text is required'}), 400

        print(f"Generating TTS for: {text}")

        # Generate audio
        wav = model.generate(text, audio_prompt_path=prompt) if prompt else model.generate(text)

        # Save to in-memory buffer
        buffer = BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)

        return send_file(buffer, mimetype='audio/wav', as_attachment=True, download_name="tts.wav")
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/")
def home():
    return "Chatterbox TTS is live!"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
