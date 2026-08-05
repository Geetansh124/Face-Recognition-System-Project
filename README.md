# 🎬 Bollywood Celebrity Face Recognition System

> **Which Bollywood celebrity do you look like?** Upload your photo and find your Bollywood doppelgänger using deep learning!

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.3.1-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.4.3-red?style=for-the-badge&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?style=for-the-badge&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-Latest-green?style=for-the-badge&logo=opencv)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Pipeline](#-pipeline)
- [Dependencies](#-dependencies)
- [Model Details](#-model-details)
- [Notes & Troubleshooting](#-notes--troubleshooting)

---

## 🧠 Overview

The **Bollywood Celebrity Face Recognition System** is a machine learning–powered web application that identifies which Bollywood celebrity a user resembles the most. By leveraging the **VGGFace (ResNet50)** deep learning model for facial feature extraction and **MTCNN** for face detection, the system compares facial embeddings using **cosine similarity** to find the closest match from a dataset of **99 top Bollywood celebrities**.

The application is served through a clean, interactive **Streamlit** web interface — simply upload a photo and get your celebrity match instantly.

---

## ✨ Features

- 🔍 **Automatic Face Detection** — Uses MTCNN to detect and crop faces from uploaded images
- 🧬 **Deep Feature Extraction** — Generates 2048-dimensional face embeddings via VGGFace ResNet50
- 📊 **Cosine Similarity Matching** — Ranks all celebrities by face similarity score
- 🖼️ **Side-by-Side Display** — Shows your photo alongside the best-matched celebrity
- 💾 **Pre-computed Embeddings** — Fast inference using pre-saved feature vectors (`embedding.pkl`)
- 🌐 **Interactive Web UI** — Clean Streamlit-based interface, no coding required to use

---

## 🛠️ Tech Stack

| Category            | Technology                           |
|---------------------|---------------------------------------|
| Language            | Python 3.7+                          |
| Deep Learning       | TensorFlow 2.3.1, Keras 2.4.3        |
| Face Model          | VGGFace (ResNet50) — `keras-vggface` |
| Face Detection      | MTCNN 0.1.0                          |
| Similarity Metric   | Cosine Similarity (scikit-learn)     |
| Image Processing    | OpenCV, Pillow (PIL)                 |
| Web Framework       | Streamlit                            |
| Data Serialization  | Pickle                               |
| Dataset Downloader  | `bing-image-downloader`              |
| Progress Tracking   | tqdm                                 |

---

## 📁 Project Structure

```
Bollywood celebrities pr/
│
├── README.md                    # Project documentation
│
└── face-recognizer/
    ├── app.py                   # 🌐 Streamlit web application (main entry point)
    ├── feature_extractor.py     # 🧬 Script to generate & save face embeddings
    ├── data_downloader.py       # 📥 Script to download celebrity images via Bing
    ├── test.py                  # 🧪 Standalone test script (CLI-based)
    │
    ├── data/                    # 📂 Celebrity image dataset (99 celebrities)
    │   ├── Aamir_Khan/
    │   ├── Aishwarya_Rai/
    │   ├── Amitabh_Bachchan/
    │   ├── Deepika_Padukone/
    │   ├── Shah_Rukh_Khan/
    │   └── ... (99 celebrities total)
    │
    ├── uploads/                 # 📤 Temporary storage for user-uploaded images
    ├── embedding.pkl            # 💾 Pre-computed face feature vectors
    └── filenames.pkl            # 💾 List of all celebrity image file paths
```

---

## 🔬 How It Works

The system follows a **two-phase pipeline**:

### Phase 1 — Offline: Feature Extraction (Pre-processing)

```
Celebrity Images  →  VGGFace ResNet50  →  2048-dim Embeddings  →  embedding.pkl
```

1. All celebrity images from the `data/` directory are loaded
2. Each image is passed through the **VGGFace ResNet50** model (pre-trained on MS-Celeb-1M)
3. The model outputs a **2048-dimensional feature vector** (facial embedding) per image
4. All embeddings are serialized and saved to `embedding.pkl`
5. Corresponding file paths are saved to `filenames.pkl`

### Phase 2 — Online: Real-Time Matching (Inference)

```
User Photo  →  MTCNN (Face Detection)  →  Face Crop  →  VGGFace  →  Cosine Similarity  →  Best Match
```

1. **User uploads** a photo via the Streamlit UI
2. **MTCNN** detects and extracts the face bounding box from the image
3. The detected face is **resized to 224×224** pixels
4. **VGGFace ResNet50** generates a 2048-dim embedding for the user's face
5. **Cosine Similarity** is computed between the user's embedding and all pre-saved celebrity embeddings
6. The celebrity with the **highest cosine similarity score** is returned as the match
7. Both the uploaded image and the matched celebrity photo are displayed side-by-side

---

## 🗃️ Dataset

The dataset contains images of **99 popular Bollywood celebrities**, sourced using the Bing Image Downloader.

<details>
<summary><strong>Click to view full list of celebrities (99 total)</strong></summary>

| # | Celebrity | # | Celebrity | # | Celebrity |
|---|-----------|---|-----------|---|-----------|
| 1 | Aamir Khan | 34 | Jacqueline Fernandez | 67 | Salman Khan |
| 2 | Abhay Deol | 35 | John Abraham | 68 | Sanjay Dutt |
| 3 | Abhishek Bachchan | 36 | Juhi Chawla | 69 | Sara Ali Khan |
| 4 | Aftab Shivdasani | 37 | Kajal Aggarwal | 70 | Shah Rukh Khan |
| 5 | Aishwarya Rai | 38 | Kajol | 71 | Shahid Kapoor |
| 6 | Akshay Kumar | 39 | Kangana Ranaut | 72 | Shilpa Shetty |
| 7 | Akshaye Khanna | 40 | Kareena Kapoor | 73 | Shraddha Kapoor |
| 8 | Alia Bhatt | 41 | Karisma Kapoor | 74 | Shreyas Talpade |
| 9 | Ameesha Patel | 42 | Kartik Aaryan | 75 | Shruti Haasan |
| 10 | Amitabh Bachchan | 43 | Katrina Kaif | 76 | Sidharth Malhotra |
| 11 | Amrita Rao | 44 | Kiara Advani | 77 | Sonakshi Sinha |
| 12 | Amy Jackson | 45 | Kriti Kharbanda | 78 | Sonam Kapoor |
| 13 | Anil Kapoor | 46 | Kriti Sanon | 79 | Suniel Shetty |
| 14 | Anushka Sharma | 47 | Kunal Khemu | 80 | Sunny Deol |
| 15 | Anushka Shetty | 48 | Lara Dutta | 81 | Sushant Singh Rajput |
| 16 | Arjun Kapoor | 49 | Madhuri Dixit | 82 | Taapsee Pannu |
| 17 | Arjun Rampal | 50 | Manoj Bajpayee | 83 | Tabu |
| 18 | Arshad Warsi | 51 | Mrunal Thakur | 84 | Tamannaah Bhatia |
| 19 | Asin | 52 | Nana Patekar | 85 | Tiger Shroff |
| 20 | Ayushmann Khurrana | 53 | Nargis Fakhri | 86 | Tusshar Kapoor |
| 21 | Bhumi Pednekar | 54 | Naseeruddin Shah | 87 | Uday Chopra |
| 22 | Bipasha Basu | 55 | Nushrat Bharucha | 88 | Vaani Kapoor |
| 23 | Bobby Deol | 56 | Paresh Rawal | 89 | Varun Dhawan |
| 24 | Deepika Padukone | 57 | Parineeti Chopra | 90 | Vicky Kaushal |
| 25 | Disha Patani | 58 | Pooja Hegde | 91 | Vidya Balan |
| 26 | Emraan Hashmi | 59 | Prabhas | 92 | Vivek Oberoi |
| 27 | Esha Gupta | 60 | Prachi Desai | 93 | Yami Gautam |
| 28 | Farhan Akhtar | 61 | Preity Zinta | 94 | Zareen Khan |
| 29 | Govinda | 62 | Priyanka Chopra | 95 | R Madhavan |
| 30 | Hrithik Roshan | 63 | Rajkummar Rao | 96 | Richa Chadda |
| 31 | Huma Qureshi | 64 | Ranbir Kapoor | 97 | Riteish Deshmukh |
| 32 | Ileana DCruz | 65 | Randeep Hooda | 98 | Saif Ali Khan |
| 33 | Irrfan Khan | 66 | Rani Mukerji | 99 | Ranveer Singh |

</details>

---

## ⚙️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/bollywood-face-recognizer.git
cd bollywood-face-recognizer/face-recognizer
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install streamlit
pip install mtcnn==0.1.0
pip install tensorflow==2.3.1
pip install keras==2.4.3
pip install keras-vggface==0.6
pip install keras_applications==1.0.8
pip install opencv-python
pip install Pillow
pip install scikit-learn
pip install tqdm
pip install bing-image-downloader
```

---

## 🚀 Usage

### Step 1: (Optional) Download Additional Celebrity Images

Edit `data_downloader.py` to specify the celebrity name and desired image count, then run:

```bash
python data_downloader.py
```

### Step 2: Generate Feature Embeddings

Run the feature extractor to generate `embedding.pkl` and `filenames.pkl` from the `data/` folder:

```bash
python feature_extractor.py
```

> ⏳ This may take several minutes depending on the size of your dataset and hardware.

### Step 3: Launch the Web Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### Step 4: Use the App

1. Click **"Choose an image"** to upload your photo
2. Wait for the model to detect your face and extract features
3. View the matched Bollywood celebrity side-by-side with your uploaded image!

---

### Running the CLI Test (Optional)

To test the core pipeline without the Streamlit UI:

```bash
# Place a test image at sample/satya.jpg
python test.py
```

The matched celebrity image will pop up in a separate OpenCV window.

---

## 🔄 Pipeline

```
[Download Images] ──▶ [Organize into data/] ──▶ [Run feature_extractor.py]
                                                         │
                                              [Save embedding.pkl + filenames.pkl]
                                                         │
                                              [Launch app.py via Streamlit]
                                                         │
                                              [User Uploads Photo]
                                                         │
                                              [MTCNN Face Detection → Crop]
                                                         │
                                              [Resize to 224×224]
                                                         │
                                              [VGGFace ResNet50 → 2048-dim Embedding]
                                                         │
                                              [Cosine Similarity vs all celebrities]
                                                         │
                                              [Best Match → Display Results]
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | 2.3.1 | Deep learning backend |
| `keras` | 2.4.3 | High-level neural network API |
| `keras-vggface` | 0.6 | Pre-trained VGGFace models |
| `keras_applications` | 1.0.8 | Keras application utilities |
| `mtcnn` | 0.1.0 | Multi-task Cascaded CNN face detector |
| `opencv-python` | Latest | Image loading and processing |
| `Pillow` | Latest | Image resizing and array conversion |
| `scikit-learn` | Latest | Cosine similarity computation |
| `streamlit` | Latest | Web application framework |
| `numpy` | Latest | Numerical array operations |
| `tqdm` | Latest | Progress bars during extraction |
| `bing-image-downloader` | Latest | Downloading training images |
| `pickle` | Built-in | Serializing embeddings to disk |

---

## 📊 Model Details

### VGGFace ResNet50

- **Architecture:** ResNet50 (adapted for face recognition)
- **Pre-training:** Trained on **MS-Celeb-1M** — a large-scale face recognition dataset
- **Input Shape:** `(224, 224, 3)` — RGB images
- **Output:** `2048-dimensional` feature vector (facial embedding)
- **Pooling:** Global Average Pooling (`avg`) applied after the last convolutional block
- **Top Layers:** Excluded (`include_top=False`) — used purely as a feature extractor

### MTCNN (Face Detector)

- **Full Name:** Multi-task Cascaded Convolutional Neural Network
- **Function:** Detects face bounding boxes in arbitrary images
- **Output Used:** Bounding box `[x, y, width, height]` from the first detected face
- **Robustness:** Works across different lighting conditions, angles, and scales

### Cosine Similarity

Given a query embedding **q** and a celebrity embedding **c**, the similarity is computed as:

```
similarity(q, c) = (q · c) / (||q|| × ||c||)
```

- Score ranges from **-1** (opposite) to **1** (identical)
- The celebrity with the **highest score** is selected as the match

---

## ⚠️ Notes & Troubleshooting

- **Only one face per upload** — The system uses the **first detected face** in the image. Ensure the uploaded photo has a clear, single, front-facing face for best results.
- **`uploads/` directory** — Make sure an `uploads/` folder exists in the `face-recognizer/` directory before running the app, or it will throw a file-not-found error.
- **Version compatibility** — `keras-vggface` is only compatible with specific older versions of TensorFlow/Keras. Stick to `tensorflow==2.3.1` and `keras==2.4.3` as listed.
- **GPU acceleration** — While not required, a CUDA-compatible GPU significantly speeds up both embedding generation and inference.
- **Embedding regeneration** — If you add new celebrities to the `data/` folder, re-run `feature_extractor.py` to regenerate `embedding.pkl` and `filenames.pkl`.
- **`st.beta_columns` deprecation** — In newer versions of Streamlit, replace `st.beta_columns(2)` with `st.columns(2)` in `app.py`.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

Built with ❤️ for Bollywood fans and ML enthusiasts
