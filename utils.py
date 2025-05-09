import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import sqlite3
import json
from tqdm import tqdm
import torch

# Khởi tạo FaceAnalysis app
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'])
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

def insert_embedding(person_name, embedding_vector):
    conn = sqlite3.connect('face_embeddings.db')
    cursor = conn.cursor()

    embedding_json = json.dumps(embedding_vector.tolist())
    cursor.execute('''
        INSERT INTO embeddings (person_name, embedding)
        VALUES (?, ?)
    ''', (person_name, embedding_json))

    conn.commit()
    conn.close()


def get_all_embeddings():
    conn = sqlite3.connect('face_embeddings.db')
    cursor = conn.cursor()

    cursor.execute('SELECT person_name, embedding FROM embeddings')
    results = cursor.fetchall()

    conn.close()
    return [(name, np.array(json.loads(embed))) for name, embed in results]

def create_database(image_root):
    conn = sqlite3.connect('face_embeddings.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    ''')
    conn.commit()

    for folder_name in os.listdir(image_root):
        person_dir = os.path.join(image_root, folder_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)

            img = cv2.imread(img_path)
            if img is None:
                print(f"Lỗi đọc ảnh: {img_path}")
                continue

            faces = app.get(img)
            if len(faces) == 0:
                print(f"Không tìm thấy khuôn mặt trong ảnh: {img_path}")
                continue

            face = faces[0]
            embedding = face.embedding
            insert_embedding(folder_name, embedding)
            print(f"Đã lưu embedding cho {img_name} (người: {folder_name})")

    conn.close()

# Hàm tính cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

if not os.path.exists('face_embeddings.db'):
    create_database('face_image')