# AgroScan – Plant Disease Recognition System

AgroScan is a deep learning web application that detects plant diseases from leaf images. Upload a photo of a leaf and the app identifies the crop and disease, explains the cause, and suggests a cure. It was built as a Final Year Project in Software Engineering to support early agricultural decision-making.

## Features

- **Disease detection** across **39 classes** covering 14 crops (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato), plus a "no leaf" background class
- **Cause and cure** information for every predicted disease
- **Drag-and-drop upload** with a clean, responsive Bootstrap UI
- **Scan history** with pagination, so you can review previous analyses
- About and Contact pages

## Tech Stack

| Layer    | Technology                                             |
|----------|--------------------------------------------------------|
| Model    | TensorFlow / Keras, EfficientNetB4 (transfer learning) |
| Backend  | Python, Flask                                          |
| Frontend | HTML, CSS, Bootstrap 5, Jinja2 templates               |
| Training | Google Colab, Jupyter Notebook                         |

## Model

- **Architecture:** EfficientNetB4 pre-trained on ImageNet, with a global average pooling layer, dropout (0.2) and a 39-class dense output layer
- **Training:** 6 epochs with the base model frozen, then 10 epochs of fine-tuning from layer 100 onwards
- **Input size:** 160 × 160 RGB
- **Dataset:** [Plant leaf diseases dataset with augmentation](https://data.mendeley.com/datasets/tywbtsjrjv/1) (Mendeley Data, about 61,000 images), split 80 / 10 / 10 into train, validation and test sets
- **Test accuracy:** **95.2%**

The full training pipeline is in [`notebooks/Plant_Disease_Model_Training.ipynb`](notebooks/Plant_Disease_Model_Training.ipynb).

## Project Structure

```
Agro-Scan-FYP/
├── app.py                  # Flask application (routes, prediction, history)
├── plant_disease.json      # Disease names with causes and cures
├── requirements.txt        # Python dependencies
├── models/
│   └── plant_disease_recog_model_pwp.keras   # Trained model (Git LFS)
├── notebooks/
│   └── Plant_Disease_Model_Training.ipynb    # Model training notebook
├── sample_data/            # Sample leaf images for testing
├── static/                 # CSS, JS and images
├── templates/              # HTML pages (home, result, history, about, contact)
└── uploadimages/           # Uploaded images are saved here at runtime
```

## Getting Started

### Prerequisites

- Python 3.9 – 3.12
- [Git LFS](https://git-lfs.com/), which is needed to download the model file

### Installation

```bash
git lfs install
git clone https://github.com/waleedbokhari786/Agro-Scan-FYP.git
cd Agro-Scan-FYP
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser, upload a leaf image (try one from `sample_data/`) and view the result.

## How It Works

1. The user uploads a leaf image on the home page.
2. The image is saved to `uploadimages/` and resized to 160 × 160.
3. The EfficientNetB4 model predicts the most likely of the 39 classes.
4. The matching cause and cure are loaded from `plant_disease.json`.
5. The result is saved to the scan history (`scans.json`) and shown on the result page.

## Supported Classes

<details>
<summary>Show all 39 classes</summary>

| Crop | Classes |
|------|---------|
| Apple | Apple scab, Black rot, Cedar apple rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery mildew, Healthy |
| Corn | Cercospora / Gray leaf spot, Common rust, Northern leaf blight, Healthy |
| Grape | Black rot, Esca (Black measles), Leaf blight, Healthy |
| Orange | Huanglongbing (Citrus greening) |
| Peach | Bacterial spot, Healthy |
| Bell Pepper | Bacterial spot, Healthy |
| Potato | Early blight, Late blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery mildew |
| Strawberry | Leaf scorch, Healthy |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy |
| Other | Background without leaves |

</details>

## Author

Developed by [@waleedbokhari786](https://github.com/waleedbokhari786) as a Final Year Project.
