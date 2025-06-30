from flask import Flask, request, jsonify
import os
from model import load_model, preprocess_audio, analyze_audio

app = Flask(__name__)
model = load_model()

@app.route('/analyze', methods=['POST'])
def analyze():
    results = {}
    for key in ['audio0.webm', 'audio1.webm', 'audio2.webm']:
        if key in request.files:
            path = f'temp_{key}.webm'
            request.files[key].save(path)
            audio_tensor = preprocess_audio(path)
            result = analyze_audio(model, audio_tensor)
            os.remove(path)
            persona = key.split('.')[0]
            results[persona] = result

    overall = {
        "confidence": round(sum(r["confidence"] for r in results.values()) / 3, 2),
        "clarity": round(sum(r["clarity"] for r in results.values()) / 3, 2),
        "emotion": max(set([r["emotion"] for r in results.values()]), key=[r["emotion"] for r in results.values()].count),
        "filler": max(set([r["filler"] for r in results.values()]), key=[r["filler"] for r in results.values()].count),
    }

    return jsonify({
        "leo": results.get("audio0", {}),
        "evelyn": results.get("audio1", {}),
        "claire": results.get("audio2", {}),
        "overall": overall
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

