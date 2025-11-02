# Plant Disease Detection - Frontend Setup

## Python Version
Use **Python 3.10.8**

## Required Packages

Install the following packages:
```bash
pip install tensorflow==2.15.0
pip install scikit-learn==1.5.2
pip install flask
pip install mysql-connector-python
pip install pandas numpy
pip install joblib
pip install werkzeug
```

## Quick Install Command
```bash
pip install -r requirements.txt
```

## Database Setup
1. Install MySQL Server
2. Create database: `CREATE DATABASE plant;`
3. Run `db.sql` to create tables
4. Update MySQL credentials in `app.py` (lines 35-41)

## Running the Application
```bash
cd frontend
python app.py
```

Then open: http://localhost:5000

## Notes
- Ensure model file `DL-models/mobilenet_v2_classifier_model.h5` is present
- Create `static/uploads/` directory if it doesn't exist
- Make sure MySQL server is running
