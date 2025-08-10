from flask import Flask, request, jsonify
import whisper
import os
import uuid
import pronouncing
import eng_to_ipa  # Add this library for IPA phonetic transcription
import Levenshtein  # Replace difflib with Levenshtein distance
from difflib import SequenceMatcher  # Add this import
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import functools

app = Flask(__name__)

# Load model once at startup - use smaller model for speed
model = whisper.load_model("tiny")  # Changed from "small" to "tiny" for faster processing

# Cache for IPA conversions to avoid repeated calculations
@functools.lru_cache(maxsize=1000)
def get_word_ipa_cached(word):
    """Get IPA transcription for a word with caching"""
    clean_word = ''.join(c for c in word.lower() if c.isalpha())
    if clean_word:
        return eng_to_ipa.convert(clean_word)
    return ""

def get_word_ipa(word):
    """Get IPA transcription for a word"""
    return get_word_ipa_cached(word)

# Optimized Levenshtein distance calculation
def fast_levenshtein_distance(s1, s2):
    """Optimized Levenshtein distance calculation"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    if len(s1) == 0:
        return len(s2)

    # Use only two rows instead of full matrix
    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def calculate_pronunciation_score_optimized(reference_ipa, spoken_ipa):
    """Optimized pronunciation accuracy score calculation"""
    if not reference_ipa or not spoken_ipa:
        return 0

    # Quick exact match check
    if reference_ipa.lower() == spoken_ipa.lower():
        return 100

    # Use optimized distance calculation
    distance = fast_levenshtein_distance(reference_ipa.lower(), spoken_ipa.lower())
    max_length = max(len(reference_ipa), len(spoken_ipa))

    if max_length == 0:
        return 100

    similarity = (1 - distance / max_length) * 100
    return round(max(0, similarity), 2)

def calculate_word_pronunciation_score_optimized(reference_ipa, spoken_ipa):
    """Optimized word-level pronunciation score calculation"""
    if not reference_ipa or not spoken_ipa:
        return 0

    # Quick exact match check
    ref_clean = reference_ipa.replace(" ", "").lower()
    spoken_clean = spoken_ipa.replace(" ", "").lower()

    if ref_clean == spoken_clean:
        return 100

    # Use optimized distance calculation
    distance = fast_levenshtein_distance(ref_clean, spoken_clean)
    max_length = max(len(ref_clean), len(spoken_clean))

    if max_length == 0:
        return 100

    # Simplified bonus calculation for speed
    basic_similarity = (1 - distance / max_length) * 100

    # Quick bonus checks
    bonus = 0
    if abs(len(ref_clean) - len(spoken_clean)) <= 1:
        bonus += 5
    if ref_clean and spoken_clean and ref_clean[0] == spoken_clean[0]:
        bonus += 3
    if ref_clean and spoken_clean and ref_clean[-1] == spoken_clean[-1]:
        bonus += 3

    final_score = min(100, basic_similarity + bonus)
    return round(max(0, final_score), 2)

def process_words_parallel(words):
    """Process multiple words in parallel for IPA conversion"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(get_word_ipa, words))

def get_file_extension(filename):
    """Get file extension from filename"""
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

def save_temp_audio_file(file):
    """Save uploaded file with appropriate extension"""
    file_ext = get_file_extension(file.filename)
    supported_formats = ['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg']

    if file_ext not in supported_formats:
        file_ext = 'wav'

    temp_filename = f"temp_audio_{uuid.uuid4().hex}.{file_ext}"
    file_path = os.path.join("./", temp_filename)
    file.save(file_path)
    return file_path

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = save_temp_audio_file(file)

    try:
        # Use faster transcription settings
        result = model.transcribe(
            file_path,
            fp16=False,  # Disable fp16 for CPU compatibility
            language="en",  # Specify language for faster processing
            task="transcribe"
        )
        text = result.get("text", "")

        # Parallel processing for pronunciation analysis
        words = text.strip().split()
        if words:
            ipa_transcriptions = process_words_parallel(words)

            pronunciation_data = []
            for i, word in enumerate(words):
                clean_word = ''.join(c for c in word.lower() if c.isalpha())
                if clean_word and i < len(ipa_transcriptions):
                    # Skip syllable counting for speed - can be added back if needed
                    pronunciation_data.append({
                        "word": word,
                        "phonetic": ipa_transcriptions[i],
                        "syllables": len([c for c in ipa_transcriptions[i] if c in 'aeiouəɪʊɔɑɛæʌɜ'])  # Quick syllable estimate
                    })
        else:
            pronunciation_data = []

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({
        "text": text,
        "pronunciation": pronunciation_data
    })

@app.route("/compare-pronunciation", methods=["POST"])
def compare_pronunciation():
    """Optimized pronunciation comparison"""
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    reference_text = request.form.get("reference_text")
    if not reference_text:
        return jsonify({"error": "No reference text provided"}), 400

    file = request.files["file"]
    reference_text = reference_text.strip()
    file_path = save_temp_audio_file(file)

    try:
        # Fast transcription with optimizations
        result = model.transcribe(
            file_path,
            fp16=False,
            language="en",
            task="transcribe",
            beam_size=1,  # Faster beam search
            best_of=1     # Single candidate
        )
        spoken_text = result.get("text", "").strip()

        # Parallel IPA processing
        reference_words = reference_text.split()
        spoken_words = spoken_text.split()

        # Process both sets of words in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_ref = executor.submit(process_words_parallel, reference_words)
            future_spoken = executor.submit(process_words_parallel, spoken_words)

            reference_ipa_words = future_ref.result()
            spoken_ipa_words = future_spoken.result()

        reference_full_ipa = " ".join(reference_ipa_words)
        spoken_full_ipa = " ".join(spoken_ipa_words)

        # Use optimized scoring
        overall_score = calculate_pronunciation_score_optimized(reference_full_ipa, spoken_full_ipa)

        # Quick word-level analysis (simplified)
        word_analysis = []
        min_len = min(len(reference_ipa_words), len(spoken_ipa_words))

        for i in range(min_len):
            score = calculate_word_pronunciation_score_optimized(
                reference_ipa_words[i], spoken_ipa_words[i]
            )
            word_analysis.append({
                "reference": reference_ipa_words[i],
                "spoken": spoken_ipa_words[i],
                "score": score,
                "status": "matched"
            })

        # Handle missing/extra words quickly
        if len(reference_ipa_words) > len(spoken_ipa_words):
            for i in range(len(spoken_ipa_words), len(reference_ipa_words)):
                word_analysis.append({
                    "reference": reference_ipa_words[i],
                    "spoken": "[missing]",
                    "score": 0,
                    "status": "missing"
                })
        elif len(spoken_ipa_words) > len(reference_ipa_words):
            for i in range(len(reference_ipa_words), len(spoken_ipa_words)):
                word_analysis.append({
                    "reference": "[extra]",
                    "spoken": spoken_ipa_words[i],
                    "score": 25,
                    "status": "extra"
                })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({
        "reference_text": reference_text,
        "spoken_text": spoken_text,
        "reference_ipa": reference_full_ipa,
        "spoken_ipa": spoken_full_ipa,
        "overall_score": overall_score,
        "accuracy_level": "Excellent" if overall_score >= 90 else "Good" if overall_score >= 75 else "Fair" if overall_score >= 60 else "Needs Improvement",
        "word_analysis": word_analysis
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
