# Image Based Plant Disease Detection 🌱

A deep learning web application for automated plant disease detection using CNN and MobileNet algorithms.

## 📋 Overview

This project implements a comprehensive plant disease detection system that can identify 51 different plant species and disease conditions from leaf images. The system achieves high accuracy rates:
- **CNN Model**: 92.76% accuracy
- **MobileNet Model**: 92.18% accuracy

## 🏗️ Project Structure

```
plant-disease-detection/
├── backend/                 # Model training and data
│   ├── code.ipynb          # Jupyter notebook with model training code
│   ├── cnn_model.h5        # Trained CNN model (not included - too large)
│   ├── mobilenet_v2_classifier_model.h5  # MobileNet model (not included - too large)
│   ├── train/              # Training dataset (not included - too large)
│   └── test/               # Test dataset (not included - too large)
│
├── frontend/               # Flask web application
│   ├── app.py             # Main Flask application
│   ├── db.sql             # Database schema
│   ├── DL-models/         # Production models
│   ├── templates/         # HTML templates
│   │   ├── index.html
│   │   ├── login.html
│   │   ├── register.html
│   │   ├── home.html
│   │   ├── prediction.html
│   │   ├── about.html
│   │   ├── algorithm.html
│   │   └── graph.html
│   └── static/            # Static assets
│       ├── css/           # Stylesheets
│       ├── js/            # JavaScript files
│       ├── img/           # Images
│       └── vendor/        # Third-party libraries
│
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🚀 Features

- **User Authentication**: Login and registration system
- **Image Upload**: Upload plant leaf images for disease detection
- **Real-time Prediction**: Instant disease classification
- **Disease Information**: Detailed causes and remedies for each disease
- **Multiple Models**: Compare CNN and MobileNet algorithms
- **Performance Metrics**: View accuracy graphs and model comparison

## 🛠️ Technology Stack

### Backend
- Python 3.x
- TensorFlow/Keras
- Jupyter Notebook
- MobileNet V2
- Custom CNN

### Frontend
- Flask
- MySQL
- HTML/CSS/JavaScript
- Bootstrap

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- MySQL Server
- pip package manager

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
```

2. **Install Python dependencies**
```bash
pip install flask tensorflow keras mysql-connector-python pandas numpy scikit-learn joblib werkzeug
```

3. **Set up MySQL Database**
   - Create a database named `plant`
   - Run `db.sql` to create the required tables
```bash
mysql -u root -p plant < frontend/db.sql
```

4. **Update Database Credentials**
   - Open `frontend/app.py`
   - Update lines 35-40 with your MySQL credentials:
```python
mydb = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    port="3306",
    database='plant'
)
```

5. **Add Model Files**
   - Download or train the model files
   - Place `mobilenet_v2_classifier_model.h5` in `frontend/DL-models/`

6. **Run the Application**
```bash
cd frontend
python app.py
```

7. **Access the Web App**
   - Open browser and go to `http://localhost:5000`

## 📊 Dataset

The project uses a dataset with 51 plant disease categories including:
- Apple diseases (Apple scab, Black rot, Cedar apple rust, Healthy)
- Tomato diseases (Bacterial spot, Early blight, Late blight, Leaf mold, etc.)
- Potato diseases (Early blight, Late blight, Healthy)
- Corn, Cherry, Grape, Peach, Bell Pepper, Strawberry diseases
- Eggplant conditions (various pests and nutrient deficiencies)
- Other plant conditions

## 🎯 Usage

1. **Register/Login**: Create an account or login
2. **Upload Image**: Go to prediction page and upload a plant leaf image
3. **Get Results**: View disease classification, causes, and remedies
4. **Compare Models**: Check algorithm page for model performance comparison

## 📝 Important Notes

⚠️ **Security**: Before pushing to GitHub, **REMOVE** your database password from `app.py` (currently visible in line 38). Use environment variables instead.

⚠️ **Large Files**: Model files (.h5) and datasets are NOT included in this repository due to size limitations. You need to:
- Train your own models using `backend/code.ipynb`, OR
- Download pre-trained models separately
- Use Git LFS for large model files if needed

## 🔬 Model Training

To train your own models:
1. Open `backend/code.ipynb` in Jupyter Notebook
2. Ensure training/test datasets are in place
3. Run all cells to train both CNN and MobileNet models
4. Models will be saved in the backend folder

## 📄 License

This project is for educational purposes.

## 👤 Authors

[Your Name]
[Your Student ID]

## 🙏 Acknowledgments

- Open-source deep learning community
- TensorFlow and Keras documentation
- PlantVillage dataset contributors

---

**Note**: This is an academic project for plant disease detection using deep learning algorithms.

