from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import cv2
import numpy as np
import os
from keras.models import load_model
import mimetypes

app = Flask(__name__)

# Load the trained model
model = load_model(r'4_classification_model.h5')

# Ensure uploads directory exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess video (resize, and ensure a fixed number of frames)
def preprocess_video(video_path, num_frames=20, frame_size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)

    # If not enough frames, pad with black frames
    while len(frames) < num_frames:
        frames.append(np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8))

    cap.release()
    return np.array(frames)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        video = request.files['video']
        video_filename = video.filename
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video.save(video_path)

        # Preprocess the video frames
        frames = preprocess_video(video_path, num_frames=20)
        frames = np.expand_dims(frames, axis=0)

        # Get prediction from the model
        prediction = model.predict(frames)[0]
        prediction_text = "Fake" if prediction > 0.5 else "Real"
        assurance = prediction[0] * 100 if prediction_text == "Fake" else (1 - prediction[0]) * 100

        # Build the video URL using url_for
        video_url = url_for('uploaded_file', filename=video_filename)

        # Render the result and pass the video URL for playback
        return render_template(
            "prediction.html",
            prediction_text=f"Video is {prediction_text}.",
            assurance=assurance,
            video_url=video_url,  # Use url_for to correctly build URL
            prediction_class="success" if prediction_text == "Real" else "danger",
            prediction_result=prediction_text.lower(),
            prediction_confidence=assurance
        )
    else:
        return render_template("prediction.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Dynamically get the MIME type of the video file
    mime_type, _ = mimetypes.guess_type(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype=mime_type)

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == "POST":
        return redirect(url_for('prediction'))
    else:
        return render_template("prediction.html")

if __name__ == '__main__':
    app.debug = True
    app.run()
