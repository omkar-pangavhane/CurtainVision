# 🎭 CurtainVision — AI Curtain Try-On

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-v11-FF6F00?style=for-the-badge&logo=ultralytics&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

### AI-Powered Curtain Detection and Virtual Try-On

**Detect curtains, analyze windows, and explore curtain styles using computer vision and AI.**

[Features](#-features) • [Tech Stack](#️-tech-stack) • [Installation](#️-installation) • [Usage](#️-usage) • [API](#-api-endpoints) • [Author](#-author)

</div>

---

## 📸 Project Preview

<div align="center">

<img src="https://ixacurtains.com/cdn/shop/articles/Street-Facing_Windows_curtains.webp?v=1783592530&width=1100" alt="Curtains in a modern living room" width="85%"/>

### Transform Your Interior with AI

Explore curtain styles and room designs as visual references for the CurtainVision project.

</div>

## 📖 Overview

**CurtainVision** is an AI-powered computer vision project designed for curtain detection and analysis. It uses Python, YOLO11, OpenCV, and FastAPI to provide a foundation for image analysis and curtain-related applications.

The project is designed to support a virtual curtain try-on workflow, where users can upload a room image, identify the window or curtain region, and explore how different curtain styles could look in the space.

The backend provides REST API endpoints, with support for authentication, caching, and a modular service architecture.

### 🎯 Project Goal

Build an AI-powered application that can:

- Detect curtains and relevant objects in images.
- Analyze uploaded room images.
- Support a future virtual curtain replacement workflow.
- Expose computer vision functionality through FastAPI.
- Provide a scalable foundation for a web-based curtain visualization application.

---

## ✨ Features

- 🎯 **Object Detection** — YOLO11-based object detection pipeline.
- 🪟 **Curtain and Window Analysis** — Analyze room images for curtain-related visual features.
- 🎨 **Virtual Curtain Try-On Concept** — Designed to support visualization of different curtain styles in a room.
- ⚡ **FastAPI Backend** — REST API development using FastAPI and Uvicorn.
- 🔐 **JWT Authentication** — Token-based authentication and authorization.
- 🚀 **Caching Layer** — Support for in-memory or Redis caching.
- 🧩 **Modular Architecture** — Separate authentication, database, service, and analysis modules.
- 📊 **Image and Video Analysis** — Analysis pipeline for uploaded media.
- 📝 **Interactive API Documentation** — Swagger UI and ReDoc.
- 🐳 **Deployment Ready** — Docker-based deployment support can be added through container configuration.

---

## 🖼️ Curtain Inspiration Gallery

The following images are visual references for interior styles and curtain designs. They are not screenshots of actual CurtainVision model predictions.

<div align="center">

<table>
<tr>
<td align="center" width="50%">

### Modern Neutral Curtains

<a href="https://ixacurtains.com/blogs/news/do-blackout-curtains-keep-heat-out">
<img src="https://ixacurtains.com/cdn/shop/articles/Street-Facing_Windows_curtains.webp?v=1783592530&width=1100" alt="Modern living room with neutral curtains" width="100%"/>
</a>

*Neutral curtain styling for contemporary interiors.*

</td>
<td align="center" width="50%">

### Floor-to-Ceiling Curtains

<a href="https://www.bestblinds.co.nz/curtain-pleat-styles">
<img src="https://www.bestblinds.co.nz/pleats/hero-curtain-pleats.webp" alt="Floor-to-ceiling curtains in a living room" width="100%"/>
</a>

*Curtain folds and window styling inspiration.*

</td>
</tr>
<tr>
<td align="center" width="50%">

### Elegant Window Styling

<a href="https://www.patelblinds.com.au/curtains-point-cook/">
<img src="https://www.patelblinds.com.au/wp-content/uploads/2024/10/curtns_img.jpg" alt="Elegant cream curtains framing glass doors" width="100%"/>
</a>

*Light-colored curtains for bright living spaces.*

</td>
<td align="center" width="50%">

### Interior Design Inspiration

<a href="https://ashtory.ru/shtory-v-gostinuyu">
<img src="https://static.tildacdn.com/tild6164-6336-4564-a363-356433653638/ashtory_gostinaya_er.jpeg" alt="Elegant living room with soft beige curtains" width="100%"/>
</a>

*Modern room design and curtain combinations.*

</td>
</tr>
</table>

</div>

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Programming Language** | Python 3.10+ |
| **Backend Framework** | FastAPI, Uvicorn |
| **Computer Vision** | Ultralytics YOLO11, OpenCV |
| **Numerical Processing** | NumPy |
| **Authentication** | JWT (`python-jose`) |
| **Password Security** | Passlib, bcrypt |
| **Database** | SQLite / PostgreSQL, SQLAlchemy |
| **Caching** | In-memory caching / Redis |
| **API Validation** | Pydantic |
| **Development Tools** | VS Code, Git, GitHub |
| **Deployment** | Docker (optional) |

---

## 📁 Project Structure

```text
CurtainVision/
├── app/
│   ├── __init__.py
│   ├── analysis.py           # Image/video analysis logic
│   ├── auth.py               # Authentication routes
│   ├── auth_utils.py         # JWT and password utilities
│   ├── cache.py              # Caching layer
│   ├── config.py             # Application configuration
│   ├── curtain_service.py    # Core curtain detection service
│   ├── database.py           # Database connections and models
│   └── dependencies.py       # FastAPI dependencies
├── models/                   # ML models and weights
├── .vscode/
│   └── launch.json           # VS Code debug configuration
├── .gitignore
├── requirements.txt
├── yolo11n.pt                # YOLO11 nano weights
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/omkar-pangavhane/CurtainVision.git
cd CurtainVision
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root.

```env
SECRET_KEY=your_super_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
DATABASE_URL=sqlite:///./curtainvision.db
```

Replace the example secret key with a securely generated value.

### 5. Run the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Ensure that `app/main.py` exists and initializes your FastAPI application.

---

## ▶️ Usage

Once the application is running, use the following URLs.

| Service | URL |
|---|---|
| **API Root** | http://127.0.0.1:8000 |
| **Swagger UI** | http://127.0.0.1:8000/docs |
| **ReDoc** | http://127.0.0.1:8000/redoc |

### API Documentation

FastAPI automatically provides interactive API documentation through Swagger UI and ReDoc, based on the application's registered routes and schemas.

---

## 🔌 API Endpoints

The following table describes the planned or documented endpoint layout. Actual availability depends on the routes implemented in the application.

| Method | Endpoint | Description | Authentication |
|---|---|---|---|
| `POST` | `/auth/register` | Register a new user | No |
| `POST` | `/auth/login` | Authenticate and receive a JWT token | No |
| `GET` | `/auth/me` | Retrieve current user information | Yes |
| `POST` | `/analyze/image` | Analyze an uploaded image | Yes |
| `POST` | `/analyze/video` | Analyze an uploaded video | Yes |
| `GET` | `/curtain/detect` | Run curtain detection | Yes |
| `GET` | `/health` | Check application health | No |

📌 Use `/docs` to inspect the endpoints actually registered by your running application.

---

## 🧪 Example API Request

Upload an image for analysis using `curl`.

```bash
curl -X POST "http://127.0.0.1:8000/analyze/image" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@sample.jpg"
```

### Example Response

The following is an illustrative response format. Actual output depends on your implemented API and detection model.

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

## 🧠 How the Curtain Try-On Workflow Can Work

A complete virtual try-on pipeline can be organized as follows:

```text
        Upload Room Image
                |
                v
       Image Preprocessing
                |
                v
      Window / Curtain Detection
                |
                v
      Region or Mask Extraction
                |
                v
        Select Curtain Style
                |
                v
    Virtual Curtain Visualization
                |
                v
         Display Final Image
```

### Main Components

| Component | Purpose |
|---|---|
| **Image Upload** | Accept an image of a room from the user. |
| **Object Detection** | Locate relevant objects and regions. |
| **Segmentation / Masking** | Isolate the window or curtain area when implemented. |
| **Curtain Selection** | Choose a curtain style, fabric, or color. |
| **Image Compositing / AI Editing** | Render the new curtain while respecting the window area. |
| **FastAPI** | Expose the pipeline through API endpoints. |

**Implementation note:** Detecting a window or curtain is not the same as performing virtual try-on. Realistic curtain replacement requires an additional image compositing, inpainting, or generative image-editing stage. A pretrained YOLO11 nano checkpoint may also need a custom-trained model or another suitable detector to recognize curtain-specific classes.

---

## 🐳 Docker Deployment

Docker support requires a `Dockerfile` in the repository root.

Once configured, build and run the container:

```bash
docker build -t curtainvision .
docker run -p 8000:8000 curtainvision
```

Open the API documentation at:

```text
http://127.0.0.1:8000/docs
```

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.

   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. Commit your changes.

   ```bash
   git commit -m "Add AmazingFeature"
   ```

4. Push the branch.

   ```bash
   git push origin feature/AmazingFeature
   ```

5. Open a Pull Request.

---

## 📄 License

This project is distributed under the **MIT License**. See the `LICENSE` file in the repository for the applicable terms.

The embedded curtain photographs belong to their respective source websites and are provided here as externally hosted visual references. Their use is subject to the respective owners' terms.

---

## 👤 Author

<div align="center">

### Omkar Pangavhane

[![GitHub](https://img.shields.io/badge/GitHub-omkar--pangavhane-181717?style=for-the-badge&logo=github)](https://github.com/omkar-pangavhane)

⭐ **If you like this project, please give it a star!** ⭐

</div>

---

<div align="center">

**Made with ❤️ using Python, FastAPI, and YOLO11**

</div>
