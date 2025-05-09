from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import cv2
import os
import uuid
from werkzeug.utils import secure_filename
from utils import app as face_app, get_all_embeddings, insert_embedding, create_database
import torch
import json

create_database('face_image')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Hàm kiểm tra định dạng ảnh
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Hàm tính cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
            file.save(filepath)

            img = cv2.imread(filepath)
            faces = face_app.get(img)

            if len(faces) == 0:
                return render_template('index.html', result="Không tìm thấy khuôn mặt nào.")

            # Trích xuất embedding từ khuôn mặt đầu tiên
            face = faces[0]
            embedding = face.embedding

            # So sánh với tất cả các embedding đã lưu
            known_faces = get_all_embeddings()
            if not known_faces:
                return render_template('index.html', result="Chưa có dữ liệu trong database.")

            similarities = [(name, cosine_similarity(embedding, emb)) for name, emb in known_faces]
            best_match = max(similarities, key=lambda x: x[1])
            
            if best_match[1] >= 0.6:  # ngưỡng có thể điều chỉnh
                result = f"Khuôn mặt giống nhất: {best_match[0]} (similarity: {best_match[1]:.2f})"
            else:
                result = "Không tìm thấy khuôn mặt khớp đủ gần."

            return render_template('index.html', result=result)

        return render_template('index.html', result="Vui lòng chọn một ảnh hợp lệ.")

    return render_template('index.html')
