from flask import Flask, request, send_file, jsonify
import torchaudio as ta
import torch
import os
from chatterbox.tts import ChatterboxTTS
from datetime import datetime

app = Flask(__name__)

# Automatically select device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)

@app.route('/generate', methods=['POST'])
def generate_audio():
    try:
        data = request.json
        text = data.get('text')
        prompt = data.get('prompt')  # Optional path to custom voice file

        if not text:
            return jsonify({'error': 'Text is required'}), 400

        # Generate audio
        if prompt:
            wav = model.generate(text, audio_prompt_path=prompt)
        else:
            wav = model.generate(text)

        # Create a unique filename
        output_path = f"output_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        ta.save(output_path, wav, model.sr)

        return send_file(output_path, mimetype='audio/wav', as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/")
def hello():
    return "Chatterbox TTS is live!"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

