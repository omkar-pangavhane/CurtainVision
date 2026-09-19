# 🎭 CurtainVision

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-v11-FF6F00?style=for-the-badge&logo=ultralytics&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**An AI-powered Computer Vision system for real-time curtain detection and analysis, built with FastAPI and YOLO11.**

[Features](#-features) • [Tech Stack](#-tech-stack) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-endpoints) • [Author](#-author)

</div>

---

## 📖 Overview

**CurtainVision** is a modern, scalable, and production-ready computer vision application that leverages the power of **YOLO11** for real-time curtain detection and object analysis. The backend is powered by **FastAPI**, offering a blazing-fast REST API with JWT authentication, caching, and modular service architecture.

Whether you're building a smart home automation system or an AI-powered surveillance solution, CurtainVision provides a robust foundation for your vision-based needs.

---

## ✨ Features

- 🎯 **Real-time Object Detection** — Powered by YOLO11 for high-accuracy curtain and object detection
- ⚡ **FastAPI Backend** — Asynchronous, high-performance REST API
- 🔐 **JWT Authentication** — Secure user authentication and authorization
- 🚀 **Intelligent Caching** — Optimized response times with built-in caching layer
- 🧩 **Modular Architecture** — Clean separation of concerns (services, routes, utils)
- 📊 **Analysis Engine** — Advanced image/video analysis pipeline
- 🐳 **Easy Deployment** — Docker-ready and cloud-friendly
- 📝 **Auto-generated API Docs** — Swagger UI & ReDoc out of the box

---

## 🛠️ Tech Stack

| Category      | Technology                          |
| ------------- | ----------------------------------- |
| **Language**  | Python 3.10+                        |
| **Backend**   | FastAPI, Uvicorn                    |
| **AI/ML**     | Ultralytics YOLO11, OpenCV, NumPy   |
| **Auth**      | JWT (python-jose), Passlib (bcrypt) |
| **Database**  | SQLite / PostgreSQL (SQLAlchemy)    |
| **Caching**   | In-memory / Redis                   |
| **Dev Tools** | VS Code, Git, Pydantic              |

---

## 📁 Project Structure

```
CurtainVision/
├── app/
│   ├── __init__.py
│   ├── analysis.py           # Image/video analysis logic
│   ├── auth.py               # Authentication routes
│   ├── auth_utils.py         # JWT & password utilities
│   ├── cache.py              # Caching layer
│   ├── config.py             # App configuration
│   ├── curtain_service.py    # Core curtain detection service
│   ├── database.py           # DB connection & models
│   └── dependencies.py       # FastAPI dependencies
├── models/                   # ML models & weights
├── .vscode/
│   └── launch.json           # VS Code debug configuration
├── .gitignore
├── requirements.txt
├── yolo11n.pt                # YOLO11 nano weights
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/omkar-pangavhane/CurtainVision.git
cd CurtainVision
```

### 2️⃣ Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file in the project root:

```env
SECRET_KEY=your_super_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
DATABASE_URL=sqlite:///./curtainvision.db
```

---

## ▶️ Usage

### Run with Uvicorn

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Run with VS Code Debugger

Press **F5** (uses `.vscode/launch.json` configuration).

### Access the Application

| Service        | URL                         |
| -------------- | --------------------------- |
| **API Root**   | http://127.0.0.1:8000       |
| **Swagger UI** | http://127.0.0.1:8000/docs  |
| **ReDoc**      | http://127.0.0.1:8000/redoc |

---

## 🔌 API Endpoints

| Method | Endpoint          | Description                 | Auth Required |
| ------ | ----------------- | --------------------------- | ------------- |
| `POST` | `/auth/register`  | Register a new user         | ❌            |
| `POST` | `/auth/login`     | Login and get JWT token     | ❌            |
| `GET`  | `/auth/me`        | Get current user info       | ✅            |
| `POST` | `/analyze/image`  | Analyze an uploaded image   | ✅            |
| `POST` | `/analyze/video`  | Analyze an uploaded video   | ✅            |
| `GET`  | `/curtain/detect` | Real-time curtain detection | ✅            |
| `GET`  | `/health`         | Health check endpoint       | ❌            |

> 📌 Full interactive docs available at `/docs`.

---

## 🧪 Example Request

```bash
curl -X POST "http://127.0.0.1:8000/analyze/image" \
  -H "Authorization: Bearer <YOUR_JWT_TOKEN>" \
  -F "file=@sample.jpg"
```

**Response:**

```json
{
  "status": "success",
  "detections": [
    {
      "class": "curtain",
      "confidence": 0.94,
      "bbox": [120, 45, 340, 560]
    }
  ],
  "processing_time_ms": 87
}
```

---

## 🐳 Docker (Optional)

```bash
docker build -t curtainvision .
docker run -p 8000:8000 curtainvision
```

---

## 🤝 Contributing

Contributions are what make the open-source community amazing! Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👤 Author

<div align="center">

**Omkar Pangavhane**

[![GitHub](https://img.shields.io/badge/GitHub-omkar--pangavhane-181717?style=for-the-badge&logo=github)](https://github.com/omkar-pangavhane)

⭐ **If you like this project, please give it a star!** ⭐

</div>

---

<div align="center">
Made with ❤️ using FastAPI & YOLO11
</div>
