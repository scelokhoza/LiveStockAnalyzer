from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
import os
import cv2
from tensorflow.keras.models import load_model
from analyzer import AnalyzeLiveStock
import numpy as np



app = Flask(__name__)


MODEL_PATH = 'cow_health_model.keras'
DEFAULT_VIDEO_PATH = 'static/assets/videos/sickcow.mp4'


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"message": "No video file provided"}), 400

    video_file = request.files['video']
    video_path = os.path.join('uploads', video_file.filename)
    video_file.save(video_path)

    analyzer = AnalyzeLiveStock(MODEL_PATH)
    result = analyzer.analyze_video(video_path)
    return jsonify(result)


@app.route('/default_video', methods=['GET'])
def get_default_video():
    return send_file(DEFAULT_VIDEO_PATH, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)