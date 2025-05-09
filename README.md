
# Face Recognition Flask

Ứng dụng web nhận diện khuôn mặt thời gian thực, xây dựng với Flask và OpenCV. Hệ thống cho phép tải lên ảnh, trích xuất embedding khuôn mặt và so sánh với cơ sở dữ liệu đã lưu trữ.

## 🧠 Mô tả dự án

Ứng dụng sử dụng Flask làm backend và OpenCV để xử lý hình ảnh. Người dùng có thể tải lên ảnh, hệ thống sẽ phát hiện khuôn mặt, trích xuất embedding và so sánh với cơ sở dữ liệu để nhận diện.

## 🛠️ Công nghệ sử dụng

- Python 3.x
- Flask
- Insightface
- OpenCV
- SQLite (qua `face_embeddings.db`)

## 📁 Cấu trúc thư mục

```
face-recognition-flask/
├── app.py                 # Flask application
├── utils.py               # Hàm hỗ trợ xử lý ảnh và embedding
├── face_embeddings.db     # Cơ sở dữ liệu lưu trữ embedding
├── requirements.txt       # Danh sách thư viện cần thiết
├── templates/             # Giao diện HTML
└── uploads/               # Thư mục lưu ảnh tải lên
```

## 🚀 Cài đặt và chạy ứng dụng

1. **Clone repository:**

   ```bash
   git clone https://github.com/dattrieuK17/face-recognition-flask.git
   cd face-recognition-flask
   ```

2. **Tạo môi trường ảo và cài đặt dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Trên Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **(Tùy chọn) Tổ chức dữ liệu khuôn mặt để thử nghiệm:**

   Tạo thư mục `face_image/` chứa các thư mục con theo tên, mỗi thư mục chứa các ảnh khuôn mặt:

   ```
   face_image/
   ├── {your_name_1}/
   │   ├── abc.jpg
   │   └── xyz.jpg
   ├── {your_name_2}/
   │   ├── jkl.jpg
   │   └── qweqwe.jpg
   ```


4. **Chạy ứng dụng:**

   ```bash
   python app.py
   ```

55. **Truy cập ứng dụng:**

   Mở trình duyệt và truy cập `http://localhost:5000`.

## 📸 Chức năng chính

- Tải lên ảnh để nhận diện khuôn mặt.
- Trích xuất embedding từ ảnh tải lên.
- So sánh embedding với cơ sở dữ liệu để nhận diện.
- Hiển thị kết quả nhận diện trên giao diện web.

## 📌 Ghi chú

- Cơ sở dữ liệu `face_embeddings.db` chứa embedding của các khuôn mặt đã biết.
- Thư mục `uploads/` lưu trữ ảnh người dùng tải lên.

