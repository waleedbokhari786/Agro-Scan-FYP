from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import tensorflow as tf
import os
from datetime import datetime
import math

app = Flask(__name__)
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

with open("plant_disease.json",'r') as file:
    plant_disease = json.load(file)

HISTORY_FILE = "scans.json"

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_to_history(record):
    history = load_history()
    history.insert(0, record) # Prepend new record
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/',methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

def extract_features(image):
    image = tf.keras.utils.load_img(image,target_size=(160,160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

@app.route('/upload/',methods = ['POST','GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        if image:
            filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
            save_path = f"uploadimages/{filename}"
            image.save(save_path)
            
            prediction = model_predict(f'./{save_path}')
            
            scan_id = uuid.uuid4().hex
            scan_record = {
                'id': scan_id,
                'imagepath': f'/{save_path}',
                'prediction': prediction,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            save_to_history(scan_record)

            return redirect(url_for('result_page', scan_id=scan_id))
    
    return redirect('/')

@app.route('/result/<scan_id>')
def result_page(scan_id):
    history = load_history()
    
    # Find current scan
    current_scan = next((item for item in history if item['id'] == scan_id), None)
    if not current_scan:
        return redirect('/')

    # Pagination logic
    page = request.args.get('page', 1, type=int)
    per_page = 4
    total_items = len(history)
    total_pages = math.ceil(total_items / per_page)
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_history = history[start_idx:end_idx]

    return render_template('result.html', 
                           scan=current_scan, 
                           history=paginated_history,
                           page=page,
                           total_pages=total_pages,
                           scan_id=scan_id)

@app.route('/latest')
def latest_analysis():
    history = load_history()
    
    # Pagination logic
    page = request.args.get('page', 1, type=int)
    per_page = 8  # Show more items per page on the history page
    total_items = len(history)
    total_pages = math.ceil(total_items / per_page)
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_history = history[start_idx:end_idx]

    return render_template('history.html', 
                           history=paginated_history,
                           page=page,
                           total_pages=total_pages)
        
if __name__ == "__main__":
    app.run(debug=True)