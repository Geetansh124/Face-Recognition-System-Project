<div align="center">

<img src="https://shieldcn.dev/header/gradient.svg?title=Bollywood%20Face%20Similarity%20Demo&subtitle=VGGFace%20embeddings%20%2B%20MTCNN%20%2B%20cosine%20similarity&accent=%23a855f7&mode=dark" alt="Face recognition project banner" width="100%" />

![Python](https://shieldcn.dev/badge/python-machine-learning-3776AB.svg?logo=python&variant=secondary)
![Streamlit](https://shieldcn.dev/badge/interface-streamlit-FF4B4B.svg?logo=streamlit&variant=secondary)
![Stars](https://shieldcn.dev/github/stars/Geetansh124/Face-Recognition-System-Project.svg?variant=outline)

**An educational Streamlit demo that compares a face embedding with a collection of Bollywood celebrity images.**

[Source](https://github.com/Geetansh124/Face-Recognition-System-Project) · [Issues](https://github.com/Geetansh124/Face-Recognition-System-Project/issues)

</div>

> **Use responsibly:** This project processes face images, which are sensitive biometric data. Use only images you have permission to process, avoid retaining uploads, and do not use similarity scores for identity verification, access decisions, or other high-impact purposes.

## How the demo works

The app uses MTCNN to detect a face, VGGFace ResNet50 to create an embedding, and cosine similarity to rank the stored celebrity-image embeddings. The included code expects `embedding.pkl` and `filenames.pkl` alongside the app; regenerate or provide these artifacts before running if they are not present in your checkout.

~~~mermaid
flowchart LR
  I[User-selected image] --> D[MTCNN face detection]
  D --> E[VGGFace embedding]
  E --> C[Cosine similarity]
  DB[(Stored image embeddings)] --> C
  C --> R[Closest sample match]
~~~

## Repository layout

| Path | Role |
| --- | --- |
| `face-recognizer/app.py` | Streamlit interface and matching flow |
| `face-recognizer/feature_extractor.py` | Embedding-generation script |
| `face-recognizer/data_downloader.py` | Image collection helper |
| `face-recognizer/data/` | Image folders used by the pipeline |
| `face-recognizer/test.py` | Standalone test/demo script |

## Setup

The project uses an older `keras-vggface` stack, so dependency compatibility depends on the Python, TensorFlow, Keras, and operating-system versions. Create an isolated environment and install compatible versions based on the script comments and package documentation. The README does not claim a currently tested modern dependency lockfile.

~~~bash
git clone https://github.com/Geetansh124/Face-Recognition-System-Project.git
cd Face-Recognition-System-Project/face-recognizer
python -m venv .venv
~~~

Activate the environment, install the compatible packages, ensure the expected embedding files exist, then launch:

~~~bash
streamlit run app.py
~~~

## Dataset, privacy, and limitations

- Image downloads must comply with source-site terms, copyright, and the rights of the people depicted.
- Similarity is not proof of identity or an objective measure of resemblance. Results can vary with pose, lighting, image quality, and dataset coverage.
- Avoid storing uploaded photos; do not expose the app publicly without reviewing upload handling, retention, and security.
- Use synthetic or consented test images and clearly disclose the educational nature of the demo.
