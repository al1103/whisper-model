from flask import Flask, request, jsonify
import whisper
import os
import uuid
import pronouncing
import eng_to_ipa  # Add this library for IPA phonetic transcription

app = Flask(__name__)
model = whisper.load_model("base")  # hoặc "tiny", "small" cho nhanh hơn

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    # Tạo tên file tạm duy nhất
    temp_filename = f"temp_audio_{uuid.uuid4().hex}.wav"
    file_path = os.path.join("./", temp_filename)
    file.save(file_path)

    try:
        result = model.transcribe(file_path)
        text = result.get("text", "")
        
        # Analyze pronunciation
        words = text.strip().split()
        pronunciation_data = []
        
        for word in words:
            # Remove punctuation for better matching
            clean_word = ''.join(c for c in word.lower() if c.isalpha())
            if clean_word:
                # Get IPA transcription
                ipa_transcription = eng_to_ipa.convert(clean_word)
                
                # Still use pronouncing for syllables and rhymes
                phonetic = pronouncing.phones_for_word(clean_word)
                syllable_count = pronouncing.syllable_count(phonetic[0]) if phonetic else 0
                
                pronunciation_data.append({
                    "word": word,
                    "phonetic": ipa_transcription,  # IPA format
                    "syllables": syllable_count,
                    "rhymes": pronouncing.rhymes(clean_word)[:5] if phonetic else []
                })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({
        "text": text,
        "pronunciation": pronunciation_data
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)