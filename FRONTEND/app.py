from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file
import mysql.connector, os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from joblib import load
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
import joblib
from tensorflow.keras.applications import MobileNetV2 



app = Flask(__name__)

# TODO: Update with your MySQL credentials before running
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # Add your MySQL password here
    port="3306",
    database='plant'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Password not matched!")
    return render_template('register.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])
        
        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')



# Flask route for displaying predictions
@app.route('/view_data', methods=['GET', 'POST'])
def view_data():
    if request.method == 'POST':
        # Get the uploaded image
        myfile = request.files['image']
        if myfile:
            fn = myfile.filename
            mypath = os.path.join('static/uploads/', fn)
            myfile.save(mypath)

            # Make the prediction using the SVM model
            predicted_class = predict_image_class(mypath, svm_model, label_encoder)
            # Assuming 'results' function gets detailed info about the prediction
            prediction, causes, remedies, organic, inorganic = results(predicted_class)

            # Render the results on the page
            return render_template('view_data.html', 
                                   prediction=predicted_class, 
                                   causes=causes, 
                                   remedies=remedies, 
                                   organic=organic, 
                                   inorganic=inorganic, 
                                   file_path=mypath)
    return render_template('view_data.html')



@app.route('/algorithm',methods=['GET','POST'])
def algorithm():
    global x_train, x_test, y_train, y_test,df
    msg = None
    if request.method == 'POST':
        algorithm = request.form['algorithm']
        
           
        
        if algorithm == "CNN":
            model_name = "CNN"
            accuracy = 92.76

        elif algorithm == "MobileNet":
            model_name = "MobileNet"
            accuracy = 92.18

        
        msg = f"Accuracy of {model_name} is {accuracy}%"
    return render_template('algorithm.html', accuracy = msg)




# Load the saved classification model

model = tf.keras.models.load_model(r'DL-models\mobilenet_v2_classifier_model.h5')
from tensorflow.keras.applications import MobileNet
# Reload the MobileNet feature extractor
base_model = MobileNet(weights='imagenet', include_top=False)
feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
               'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
               'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Downy mildew', 
               'Eggplant - Epilachna Beetle', 'Eggplant - Flea Beetle', 'Eggplant - Healthy', 'Eggplant - Jassid', 
               'Eggplant - Mite', 'Eggplant - Mite and Epilachna Beetle', 'Eggplant - Nitrogen Deficiency', 
               'Eggplant - Nitrogen and Potassium Deficiency', 'Eggplant - Potassium Deficiency', 'Fresh Leaf', 
               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Gray mold', 'Leaf scars', 
               'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 
               'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
               'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
               'Tomato___Septoria_leaf_spot', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 
               'Tomato___healthy', 'cordana', 'healthy', 'pestalotiopsis', 'sigatoka']



disease_info = {
    'Apple___Apple_scab': {'cause': 'Caused by the fungus Venturia inaequalis.', 'remedy': 'Apply fungicides and prune affected areas.'},
    'Apple___Black_rot': {'cause': 'Caused by the fungus Botryosphaeria obtusa.', 'remedy': 'Remove infected parts and apply fungicides.'},
    'Apple___Cedar_apple_rust': {'cause': 'Caused by the fungus Gymnosporangium juniperi-virginianae.', 'remedy': 'Remove nearby cedar trees and use fungicides.'},
    'Apple___healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'Background_without_leaves': {'cause': 'No plant material detected.', 'remedy': 'Ensure image contains plant leaves for analysis.'},
    'Blueberry___healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'Cherry___Powdery_mildew': {'cause': 'Caused by the fungus Podosphaera clandestina.', 'remedy': 'Apply sulfur-based fungicides and improve air circulation.'},
    'Cherry___healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'Corn___Common_rust': {'cause': 'Caused by the fungus Puccinia sorghi.', 'remedy': 'Apply fungicides and plant resistant varieties.'},
    'Corn___Northern_Leaf_Blight': {'cause': 'Caused by the fungus Exserohilum turcicum.', 'remedy': 'Use resistant hybrids and apply fungicides.'},
    'Corn___healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'Downy mildew': {'cause': 'Caused by various oomycete pathogens like Peronospora.', 'remedy': 'Improve air circulation and apply copper-based fungicides.'},
    'Eggplant - Epilachna Beetle': {'cause': 'Caused by infestation of Epilachna beetles.', 'remedy': 'Use insecticidal soap or introduce natural predators.'},
    'Eggplant - Flea Beetle': {'cause': 'Caused by flea beetle feeding damage.', 'remedy': 'Apply row covers or use organic insecticides.'},
    'Eggplant - Healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'Eggplant - Jassid': {'cause': 'Caused by jassid (leafhopper) infestation.', 'remedy': 'Use insecticidal sprays or sticky traps.'},
    'Eggplant - Mite': {'cause': 'Caused by spider mite infestation.', 'remedy': 'Apply miticides or increase humidity.'},
    'Eggplant - Mite and Epilachna Beetle': {'cause': 'Caused by combined mite and beetle damage.', 'remedy': 'Use miticides and insecticides together.'},
    'Eggplant - Nitrogen Deficiency': {'cause': 'Caused by insufficient nitrogen in soil.', 'remedy': 'Apply nitrogen-rich fertilizer.'},
    'Eggplant - Nitrogen and Potassium Deficiency': {'cause': 'Caused by lack of nitrogen and potassium.', 'remedy': 'Use a balanced NPK fertilizer.'},
    'Eggplant - Potassium Deficiency': {'cause': 'Caused by insufficient potassium in soil.', 'remedy': 'Apply potassium-rich fertilizer.'},
    'Fresh Leaf': {'cause': 'No specific disease identified.', 'remedy': 'Monitor plant health regularly.'},
    'Grape___Black_rot': {'cause': 'Caused by the fungus Guignardia bidwellii.', 'remedy': 'Prune affected areas and apply fungicides.'},
    'Grape___Esca_(Black_Measles)': {'cause': 'Caused by fungal pathogens like Phaeomoniella.', 'remedy': 'Remove infected vines and improve vineyard hygiene.'},
    'Grape___healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'Gray mold': {'cause': 'Caused by the fungus Botrytis cinerea.', 'remedy': 'Reduce humidity and apply fungicides.'},
    'Leaf scars': {'cause': 'Caused by physical damage or natural leaf drop.', 'remedy': 'No treatment needed unless infection occurs.'},
    'Peach___Bacterial_spot': {'cause': 'Caused by Xanthomonas arboricola bacteria.', 'remedy': 'Remove affected parts and apply copper sprays.'},
    'Peach___healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'Pepper,_bell___Bacterial_spot': {'cause': 'Caused by Xanthomonas campestris bacteria.', 'remedy': 'Avoid overhead watering and use copper-based sprays.'},
    'Pepper,_bell___healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'Potato___Early_blight': {'cause': 'Caused by the fungus Alternaria solani.', 'remedy': 'Apply fungicides and rotate crops.'},
    'Potato___Late_blight': {'cause': 'Caused by the oomycete Phytophthora infestans.', 'remedy': 'Use resistant varieties and apply fungicides.'},
    'Potato___healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'Raspberry___healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'Soybean___healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'Squash___Powdery_mildew': {'cause': 'Caused by the fungus Podosphaera xanthii.', 'remedy': 'Apply sulfur or potassium bicarbonate sprays.'},
    'Strawberry___Leaf_scorch': {'cause': 'Caused by the fungus Diplocarpon earlianum.', 'remedy': 'Remove affected leaves and apply fungicides.'},
    'Strawberry___healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'Tomato___Bacterial_spot': {'cause': 'Caused by Xanthomonas bacteria.', 'remedy': 'Remove affected plants and avoid overhead watering.'},
    'Tomato___Early_blight': {'cause': 'Caused by the fungus Alternaria solani.', 'remedy': 'Apply fungicides and improve air circulation.'},
    'Tomato___Late_blight': {'cause': 'Caused by the oomycete Phytophthora infestans.', 'remedy': 'Use resistant varieties and apply fungicides.'},
    'Tomato___Leaf_Mold': {'cause': 'Caused by the fungus Fulvia fulva.', 'remedy': 'Increase ventilation and apply fungicides.'},
    'Tomato___Septoria_leaf_spot': {'cause': 'Caused by the fungus Septoria lycopersici.', 'remedy': 'Remove infected leaves and apply fungicides.'},
    'Tomato___Target_Spot': {'cause': 'Caused by the fungus Corynespora cassiicola.', 'remedy': 'Apply fungicides and reduce leaf wetness.'},
    'Tomato___Tomato_mosaic_virus': {'cause': 'Caused by the Tomato mosaic virus.', 'remedy': 'Remove infected plants and disinfect tools.'},
    'Tomato___healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'cordana': {'cause': 'Caused by the fungus Cordana musae.', 'remedy': 'Apply fungicides and remove affected leaves.'},
    'healthy': {'cause': 'No disease present.', 'remedy': 'Maintain regular care and monitoring.'},
    'pestalotiopsis': {'cause': 'Caused by the fungus Pestalotiopsis spp.', 'remedy': 'Prune affected areas and apply fungicides.'},
    'sigatoka': {'cause': 'Caused by the fungus Mycosphaerella fijiensis.', 'remedy': 'Apply fungicides and improve canopy airflow.'}
}

def make_prediction(model, image_path):
    """
    Preprocess the image and make a prediction using the trained model.
    """
    # Load the image and preprocess it
    img = load_img(image_path, target_size=(224, 224))  # Resize to MobileNet input size
    img_array = img_to_array(img)  # Convert to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for MobileNet

    # Extract features using the MobileNet feature extractor
    features = feature_extractor.predict(img_array)

    # Predict the class using the trained classification model
    predictions = model.predict(features)
    predicted_class_idx = np.argmax(predictions)  # Get index of the highest probability
    predicted_class = class_names[predicted_class_idx]  # Map index to class name
    return predicted_class

@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        myfile = request.files['file']  # Get the uploaded file
        fn = myfile.filename  # Extract filename
        mypath = os.path.join('static', 'img', fn)  # Save path
        myfile.save(mypath)  # Save the file to the server

        # Make prediction
        predicted_class = make_prediction(model, mypath)

        cause = disease_info.get(predicted_class, {}).get('cause', 'No cause information available.')
        remedy = disease_info.get(predicted_class, {}).get('remedy', 'No remedy information available.')
     

        # Return result to the template
        return render_template('prediction.html', path=mypath, prediction=predicted_class,cause=cause,remedy=remedy)

    return render_template('prediction.html')







@app.route('/graph')
def graph():
    return render_template('graph.html')




if __name__ == '__main__':
    app.run(debug = True)