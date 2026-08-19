import os

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify
)

from werkzeug.utils import secure_filename
from predict import predict_emotion
from pydub import AudioSegment

# ==========================================
# Tell pydub where FFmpeg is
# ==========================================

AudioSegment.converter = r"C:\Users\HI\Downloads\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\HI\Downloads\ffmpeg-8.1.2-essentials_build\bin\ffprobe.exe"

# ==========================================
# Flask Configuration
# ==========================================

app = Flask(__name__)

app.secret_key = "emotion_ai_secret_key"

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {"wav"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# Helper Function
# ==========================================

def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# Upload WAV Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "audio" not in request.files:
        flash("No audio selected.")
        return redirect(url_for("home"))

    file = request.files["audio"]

    if file.filename == "":
        flash("Please choose a WAV file.")
        return redirect(url_for("home"))

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        emotion, confidence = predict_emotion(filepath)

        return render_template(
            "result.html",
            emotion=emotion,
            confidence=round(confidence * 100, 2),
            filename=filename
        )

    flash("Only WAV files are supported.")

    return redirect(url_for("home"))


# ==========================================
# Live Recording Route
# ==========================================

@app.route("/record", methods=["POST"])
def record():

    try:

        if "audio" not in request.files:
            return jsonify({"error": "No audio received"}), 400

        audio = request.files["audio"]

        webm_path = os.path.join(app.config["UPLOAD_FOLDER"], "recorded.webm")
        wav_path = os.path.join(app.config["UPLOAD_FOLDER"], "recorded.wav")

        audio.save(webm_path)

        from pydub import AudioSegment

        AudioSegment.converter = r"C:\Users\HI\Downloads\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe"
        AudioSegment.ffprobe = r"C:\Users\HI\Downloads\ffmpeg-8.1.2-essentials_build\bin\ffprobe.exe"

        sound = AudioSegment.from_file(webm_path, format="webm")
        sound.export(wav_path, format="wav")

        emotion, confidence = predict_emotion(wav_path)

        return jsonify({
            "emotion": emotion,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500
        # ==========================================
# Run Flask Application
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )