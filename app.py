import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from model_utils import VALID_IMAGE_EXTENSIONS, predict_external_cataract, prepare_upload_image

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in VALID_IMAGE_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", error=None, result=None, image_url=None)


@app.route("/predict", methods=["POST"])
def predict_route():
    file = request.files.get("image")
    if not file or not file.filename:
        return render_template("index.html", error="Please choose a JPG or PNG eye photo.", result=None, image_url=None)

    if not allowed_file(file.filename):
        return render_template("index.html", error="Only JPG, JPEG, and PNG images are supported.", result=None, image_url=None)

    safe_name = secure_filename(file.filename)
    file_id = uuid.uuid4().hex
    raw_upload_path = UPLOAD_FOLDER / f"{file_id}_raw{Path(safe_name).suffix.lower()}"
    final_upload_path = UPLOAD_FOLDER / f"{file_id}.jpg"

    file.save(raw_upload_path)

    try:
        prepare_upload_image(raw_upload_path, final_upload_path)
        result = predict_external_cataract(final_upload_path)
        image_url = f"/static/uploads/{final_upload_path.name}"
        return render_template("index.html", error=None, result=result, image_url=image_url)
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Could not analyze that image: {exc}",
            result=None,
            image_url=None,
        )
    finally:
        if raw_upload_path.exists():
            raw_upload_path.unlink()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8888)), debug=False)
