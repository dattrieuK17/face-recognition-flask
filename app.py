from flask import Flask, request, render_template
import cv2
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename
from utils import get_all_embeddings, cosine_similarity
from insightface.app import FaceAnalysis
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    output_img_path = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return render_template('index.html', result="Vui lòng chọn ảnh.")

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)

        img = cv2.imread(filepath)
        faces = face_app.get(img)

        if not faces:
            return render_template('index.html', result="Không tìm thấy khuôn mặt nào.", input_image=unique_name)

        known_faces = get_all_embeddings()
        for face in faces:
            emb = face.embedding
            name = "Stranger"
            max_sim = 0

            if known_faces:
                sims = [(person, cosine_similarity(emb, known_emb)) for person, known_emb in known_faces]
                name, max_sim = max(sims, key=lambda x: x[1])
                if max_sim < 0.5:
                    name = "Unknown"

            # Vẽ bounding box + name
            x1, y1, x2, y2 = map(int, face.bbox)
            # --- Font và kích thước text ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, thickness)

            # --- Tọa độ khung chữ ---
            rect_x1 = x1
            rect_y1 = y1 - text_height - 10 if y1 - text_height - 10 > 0 else y1 + 10
            rect_x2 = x1 + text_width + 10
            rect_y2 = rect_y1 + text_height + 10

            # --- Vẽ nền mờ cho label ---
            overlay = img.copy()
            cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # --- Vẽ chữ label ---
            cv2.putText(img, name, (rect_x1 + 5, rect_y2 - 5), font, font_scale, (0, 0, 0), thickness)

            # --- Vẽ bounding box ---
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Lưu ảnh kết quả
        output_name = "result_" + unique_name
        output_img_path = os.path.join(app.config['UPLOAD_FOLDER'], output_name)
        cv2.imwrite(output_img_path, img)

        return render_template('index.html', result="Xử lý xong!", input_image=unique_name, output_image=output_name)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False)
