from flask import Flask, redirect, render_template, request, flash, jsonify, session
from flask_session import Session
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import webbrowser
import threading
from werkzeug.utils import secure_filename
import cv2
import sqlite3
from datetime import datetime
import json
from geopy.geocoders import Nominatim
import requests

app = Flask(__name__)
app.secret_key = 'secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load Models
try:
    model_resnet = load_model('Resnet_leaf.h5')
    model_inceptionv3 = load_model('Inceptionv3_leaf.h5')
    models_loaded = True
except:
    models_loaded = False
    print("Models could not be loaded. Running in demo mode.")

# Mapping of class names
class_names = ['Health', 'Bacterial leaf blight', 'Brown spot', 'Leaf smut']

# Treatment mapping
treatment_mapping = {
    'Health': 'No treatment needed. Maintain good agricultural practices.',
    'Bacterial leaf blight': 'Use copper-based bactericides like Blitox-50 and ensure proper water drainage. Recommended dosage: 2g/liter of water.',
    'Brown spot': 'Apply fungicides like Hexaconazole (Contaf Plus) at 1ml/liter or Tricyclazole. Avoid high humidity conditions.',
    'Leaf smut': 'Use systemic fungicides like Carbendazim (Bavistin) at 1g/liter or Propiconazole.'
}

# Prevention tips
prevention_tips = {
    'Bacterial leaf blight': [
        'Use disease-free seeds from certified sources',
        'Avoid excessive nitrogen fertilization',
        'Practice proper water management - avoid flooding',
        'Remove infected plant debris after harvest',
        'Rotate crops with pulses or oilseeds'
    ],
    'Brown spot': [
        'Use resistant varieties like Swarna, IR64',
        'Maintain proper plant spacing (20x15 cm)',
        'Avoid water stress through regular irrigation',
        'Apply balanced fertilization with emphasis on silicon',
        'Remove infected plant material and burn it'
    ],
    'Leaf smut': [
        'Use certified disease-free seeds',
        'Practice crop rotation with non-rice crops for 2 seasons',
        'Avoid planting in poorly drained soils',
        'Apply appropriate fungicides preventively at tillering stage',
        'Remove and destroy infected plants'
    ],
    'General': [
        'Regularly monitor your crops every week',
        'Maintain field sanitation by removing weeds',
        'Use resistant varieties when available',
        'Practice proper irrigation techniques - avoid water stagnation',
        'Follow recommended fertilization practices based on soil testing'
    ]
}

# Indian Government schemes and resources
govt_schemes = [
    {
        'name': 'Pradhan Mantri Fasal Bima Yojana (PMFBY)',
        'description': 'Crop insurance scheme to protect against natural calamities',
        'link': 'https://pmfby.gov.in/',
        'eligibility': 'All farmers including sharecroppers and tenant farmers'
    },
    {
        'name': 'Krishi Sinchai Yojana',
        'description': 'Micro irrigation fund to improve water use efficiency',
        'link': 'https://pmksy.gov.in/',
        'eligibility': 'Individual farmers, farmer groups, NGOs'
    },
    {
        'name': 'Soil Health Card Scheme',
        'description': 'Provides soil health information to farmers',
        'link': 'https://soilhealth.dac.gov.in/',
        'eligibility': 'All farmers can avail free soil testing'
    },
    {
        'name': 'National Mission on Sustainable Agriculture',
        'description': 'Promotes sustainable agriculture practices',
        'link': 'https://nmsa.dac.gov.in/',
        'eligibility': 'Farmers, farmer producer organizations'
    },
    {
        'name': 'Paramparagat Krishi Vikas Yojana (PKVY)',
        'description': 'Promotes organic farming practices',
        'link': 'https://pgsindia-ncof.gov.in/PKVY/index.aspx',
        'eligibility': 'Farmers willing to practice organic farming'
    },
    {
        'name': 'PM-KISAN Scheme',
        'description': 'Direct income support of ₹6,000 per year to farmers',
        'link': 'https://pmkisan.gov.in/',
        'eligibility': 'Small and marginal landholder farmer families'
    }
]

# Local resources - now with regional data
local_resources = {
    'agricultural_experts': [
        {'name': 'Dr. Rajesh Kumar', 'specialization': 'Plant Pathology', 'contact': '+91 9876543210', 'region': 'North India'},
        {'name': 'Dr. Priya Singh', 'specialization': 'Rice Diseases', 'contact': '+91 9765432109', 'region': 'East India'},
        {'name': 'Dr. S. M. Patel', 'specialization': 'Agricultural Extension', 'contact': '+91 9654321098', 'region': 'West India'},
        {'name': 'Dr. K. Venkatesh', 'specialization': 'Soil Health', 'contact': '+91 9543210987', 'region': 'South India'},
        {'name': 'Dr. A. Sharma', 'specialization': 'Crop Protection', 'contact': '+91 9432109876', 'region': 'Central India'},
        {'name': 'Dr. M. Das', 'specialization': 'Rice Cultivation', 'contact': '+91 9321098765', 'region': 'North East India'}
    ],
    'supply_stores': [
        {'name': 'Krishi Seva Kendra', 'items': 'Fungicides, Fertilizers, Seeds', 'location': 'Main Market', 'region': 'North India'},
        {'name': 'Farm Solutions', 'items': 'Organic Pesticides, Tools', 'location': 'Industrial Area', 'region': 'East India'},
        {'name': 'IFFCO Bazar', 'items': 'All agricultural inputs', 'location': 'Multiple locations', 'region': 'All India'},
        {'name': 'Agro Inputs Center', 'items': 'Seeds, Fertilizers, Equipment', 'location': 'Agricultural Mandi', 'region': 'West India'},
        {'name': 'Kisan Supply Store', 'items': 'Pesticides, Tools, Seeds', 'location': 'City Center', 'region': 'Central India'},
        {'name': 'North East Agro', 'items': 'Organic inputs, Traditional varieties', 'location': 'Guwahati', 'region': 'North East India'}
    ],
    'extension_services': [
        {'name': 'Agricultural Extension Office', 'services': 'Soil Testing, Advisory, Training', 'contact': '0551-2345678', 'region': 'North India'},
        {'name': 'KVK (Krishi Vigyan Kendra)', 'services': 'Farmers training, Demonstration', 'contact': '0661-3456789', 'region': 'East India'},
        {'name': 'ATMA (Agricultural Technology Management Agency)', 'services': 'Technology dissemination', 'contact': '0771-4567890', 'region': 'South India'},
        {'name': 'State Agricultural University', 'services': 'Research, Advisory, Training', 'contact': '0881-5678901', 'region': 'West India'},
        {'name': 'Krishi Vigyan Kendra', 'services': 'Training, Demonstration, Advisory', 'contact': '0991-6789012', 'region': 'Central India'},
        {'name': 'ICAR Research Complex', 'services': 'Research, Technology transfer', 'contact': '0110-7890123', 'region': 'North East India'}
    ]
}

# Regional crop calendars for different Indian states (expanded list)
crop_calendars = {
    'Punjab': {
        'Kharif Season (June-October)': ['Rice', 'Cotton', 'Sugarcane', 'Maize'],
        'Rabi Season (November-April)': ['Wheat', 'Mustard', 'Barley', 'Potato']
    },
    'Haryana': {
        'Kharif Season (June-October)': ['Rice', 'Cotton', 'Pearl Millet', 'Sorghum'],
        'Rabi Season (November-April)': ['Wheat', 'Mustard', 'Chickpea', 'Barley']
    },
    'Uttar Pradesh': {
        'Kharif Season (June-October)': ['Rice', 'Maize', 'Sugarcane', 'Pigeon Pea'],
        'Rabi Season (November-April)': ['Wheat', 'Mustard', 'Potato', 'Lentil']
    },
    'Bihar': {
        'Kharif Season (June-October)': ['Rice', 'Maize', 'Pigeon Pea', 'Green Gram'],
        'Rabi Season (November-April)': ['Wheat', 'Mustard', 'Lentil', 'Peas']
    },
    'West Bengal': {
        'Aus (April-August)': ['Rice', 'Jute', 'Pulses', 'Oilseeds'],
        'Aman (July-December)': ['Rice', 'Oilseeds', 'Vegetables', 'Potato'],
        'Boro (November-May)': ['Rice', 'Potato', 'Onion', 'Tomato']
    },
    'Odisha': {
        'Kharif Season (June-October)': ['Rice', 'Pulses', 'Oilseeds', 'Maize'],
        'Rabi Season (November-April)': ['Wheat', 'Mustard', 'Vegetables', 'Pulses']
    },
    'Andhra Pradesh': {
        'Kharif Season (June-October)': ['Rice', 'Cotton', 'Groundnut', 'Chilli'],
        'Rabi Season (November-April)': ['Rice', 'Maize', 'Sunflower', 'Pulses']
    },
    'Telangana': {
        'Kharif Season (June-October)': ['Rice', 'Cotton', 'Maize', 'Soybean'],
        'Rabi Season (November-April)': ['Rice', 'Chickpea', 'Sunflower', 'Vegetables']
    },
    'Tamil Nadu': {
        'Kuruvai (June-September)': ['Rice', 'Cotton', 'Pulses', 'Oilseeds'],
        'Thaladi (September-January)': ['Rice', 'Oilseeds', 'Sugarcane', 'Banana'],
        'Navarai (January-April)': ['Rice', 'Vegetables', 'Flowers', 'Pulses']
    },
    'Kerala': {
        'Autumn (April-August)': ['Rice', 'Rubber', 'Coconut', 'Spices'],
        'Winter (September-December)': ['Rice', 'Banana', 'Tapioca', 'Vegetables'],
        'Summer (January-March)': ['Rice', 'Pulses', 'Oilseeds', 'Fruits']
    },
    'Karnataka': {
        'Kharif Season (June-October)': ['Rice', 'Cotton', 'Groundnut', 'Sugarcane'],
        'Rabi Season (November-April)': ['Rice', 'Maize', 'Sunflower', 'Pulses']
    },
    'Maharashtra': {
        'Kharif Season (June-October)': ['Rice', 'Cotton', 'Soybean', 'Pulses'],
        'Rabi Season (October-March)': ['Wheat', 'Chickpea', 'Sunflower', 'Vegetables']
    },
    'Gujarat': {
        'Kharif Season (June-October)': ['Rice', 'Cotton', 'Groundnut', 'Castor'],
        'Rabi Season (November-April)': ['Wheat', 'Mustard', 'Cumin', 'Isabgol']
    },
    'Rajasthan': {
        'Kharif Season (June-October)': ['Rice', 'Cotton', 'Pearl Millet', 'Pulses'],
        'Rabi Season (November-April)': ['Wheat', 'Mustard', 'Barley', 'Cumin']
    },
    'Madhya Pradesh': {
        'Kharif Season (June-October)': ['Rice', 'Soybean', 'Pulses', 'Maize'],
        'Rabi Season (November-April)': ['Wheat', 'Chickpea', 'Mustard', 'Lentil']
    },
    'Chhattisgarh': {
        'Kharif Season (June-October)': ['Rice', 'Pulses', 'Oilseeds', 'Maize'],
        'Rabi Season (November-April)': ['Wheat', 'Chickpea', 'Vegetables', 'Lentil']
    },
    'Jharkhand': {
        'Kharif Season (June-October)': ['Rice', 'Maize', 'Pulses', 'Oilseeds'],
        'Rabi Season (November-April)': ['Wheat', 'Mustard', 'Vegetables', 'Pulses']
    },
    'Assam': {
        'Autumn (April-August)': ['Rice', 'Jute', 'Pulses', 'Oilseeds'],
        'Winter (September-December)': ['Rice', 'Mustard', 'Potato', 'Vegetables'],
        'Summer (January-March)': ['Rice', 'Pulses', 'Oilseeds', 'Fruits']
    },
    'North Eastern States': {
        'Spring (March-June)': ['Rice', 'Maize', 'Vegetables', 'Pulses'],
        'Autumn (July-October)': ['Rice', 'Millet', 'Oilseeds', 'Spices'],
        'Winter (November-February)': ['Rice', 'Mustard', 'Potato', 'Peas']
    }
}

# Weather API integration (example using OpenWeatherMap)
def get_weather_data(region):
    # Map regions to cities for weather data
    region_city_map = {
        'North India': 'Delhi',
        'South India': 'Chennai',
        'East India': 'Kolkata',
        'West India': 'Mumbai',
        'Central India': 'Bhopal',
        'North East India': 'Guwahati'
    }
    
    city = region_city_map.get(region, 'Delhi')
    
    # In a real implementation, you would use an actual API key
    # api_key = "your_openweathermap_api_key"
    # base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city},in&appid={api_key}&units=metric"
    
    # For demo purposes, returning mock data based on region
    mock_weather_data = {
        'North India': {'temperature': 28, 'humidity': 65, 'description': 'Partly cloudy', 'rainfall': '10% chance'},
        'South India': {'temperature': 32, 'humidity': 75, 'description': 'Mostly sunny', 'rainfall': '20% chance'},
        'East India': {'temperature': 30, 'humidity': 80, 'description': 'Humid', 'rainfall': '30% chance'},
        'West India': {'temperature': 31, 'humidity': 70, 'description': 'Sunny', 'rainfall': '5% chance'},
        'Central India': {'temperature': 29, 'humidity': 60, 'description': 'Clear skies', 'rainfall': '0% chance'},
        'North East India': {'temperature': 27, 'humidity': 85, 'description': 'Cloudy', 'rainfall': '40% chance'}
    }
    
    return mock_weather_data.get(region, {'temperature': 28, 'humidity': 75, 'description': 'Partly cloudy', 'rainfall': '10% chance'})

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_leaf_image(image_path):
    """Basic check to verify if the uploaded image is likely a leaf"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range of green color in HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate percentage of green pixels
        green_percentage = np.sum(mask > 0) / (img.shape[0] * img.shape[1])
        
        # If more than 10% of the image is green, consider it a leaf
        return green_percentage > 0.1
    except:
        return False

def predict_image(image_path, model, target_size=(64, 64)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(np.array(prediction))
    return prediction, predicted_class_index

def init_db():
    """Initialize database for storing user queries and history"""
    conn = sqlite3.connect('farmer_queries.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS queries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  phone TEXT,
                  location TEXT,
                  query TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if not models_loaded:
        flash('Models are not loaded. Running in demonstration mode.', 'warning')
    
    models = {
        'ResNet50': model_resnet if models_loaded else None,
        'InceptionV3': model_inceptionv3 if models_loaded else None
    }
    
    predictions = {}
    predicted_treatment = None
    image_path = None
    prevention_advice = []
    is_leaf = True
    weather_data = None
    user_region = session.get('region', 'North India')  # Default region

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected. Please choose an image to upload.', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('static', 'uploads', filename)
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.save(file_path)
            
            # Check if the uploaded image is likely a leaf
            is_leaf = is_leaf_image(file_path)
            
            if not is_leaf:
                flash('The uploaded image does not appear to be a leaf or lacks clarity. Please upload a clear image of a rice leaf for accurate diagnosis.', 'warning')
            else:
                image_path = file_path
                
                if models_loaded:
                    for model_name, model in models.items():
                        try:
                            if model_name == 'InceptionV3':
                                prediction, predicted_class_index = predict_image(file_path, model, (75, 75))
                            else:
                                prediction, predicted_class_index = predict_image(file_path, model)
                            
                            predicted_class = class_names[predicted_class_index]
                            predictions[model_name] = {
                                'class': predicted_class,
                                'confidence': float(prediction[0][predicted_class_index])
                            }
                        except Exception as e:
                            flash(f'Error processing image with {model_name}: {str(e)}', 'error')
                
                    if predictions:
                        top_model = max(predictions.items(), key=lambda x: x[1]['confidence'])
                        top_prediction_class = top_model[1]['class']
                        predicted_treatment = treatment_mapping.get(top_prediction_class, 'No treatment available.')
                        
                        if top_prediction_class != 'Health':
                            prevention_advice = prevention_tips.get(top_prediction_class, []) + prevention_tips['General']
                else:
                    # Demo mode with mock predictions
                    predictions = {
                        'ResNet50': {'class': 'Bacterial leaf blight', 'confidence': 0.87},
                        'InceptionV3': {'class': 'Bacterial leaf blight', 'confidence': 0.92}
                    }
                    predicted_treatment = treatment_mapping['Bacterial leaf blight']
                    prevention_advice = prevention_tips['Bacterial leaf blight'] + prevention_tips['General']
        else:
            flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP).', 'error')
    
    # Get weather data based on user's region
    if user_region:
        weather_data = get_weather_data(user_region)

    return render_template(
        'index.html',
        predictions=predictions,
        image_path=image_path,
        predicted_treatment=predicted_treatment,
        prevention_advice=prevention_advice,
        is_leaf=is_leaf,
        govt_schemes=govt_schemes,
        local_resources=local_resources,
        crop_calendars=crop_calendars,
        weather_data=weather_data,
        user_region=user_region
    )

@app.route('/set_region', methods=['POST'])
def set_region():
    region = request.form.get('region')
    if region:
        session['region'] = region
        flash(f'Region set to {region}', 'success')
    return redirect('/')

@app.route('/submit_query', methods=['POST'])
def submit_query():
    name = request.form.get('name')
    phone = request.form.get('phone')
    location = request.form.get('location')
    query = request.form.get('query')
    
    # Store query in database
    conn = sqlite3.connect('farmer_queries.db')
    c = conn.cursor()
    c.execute("INSERT INTO queries (name, phone, location, query) VALUES (?, ?, ?, ?)",
              (name, phone, location, query))
    conn.commit()
    conn.close()
    
    flash('Your query has been submitted. An expert will contact you soon.', 'success')
    return redirect('/')

@app.route('/get_prevention_tips')
def get_prevention_tips():
    disease = request.args.get('disease', 'General')
    tips = prevention_tips.get(disease, []) + prevention_tips['General']
    return jsonify({'disease': disease, 'tips': tips})

@app.route('/get_region_resources')
def get_region_resources():
    region = request.args.get('region', 'North India')
    
    # Filter resources by region
    filtered_resources = {
        'agricultural_experts': [expert for expert in local_resources['agricultural_experts'] if expert['region'] == region],
        'supply_stores': [store for store in local_resources['supply_stores'] if store['region'] == region or store['region'] == 'All India'],
        'extension_services': [service for service in local_resources['extension_services'] if service['region'] == region]
    }
    
    return jsonify(filtered_resources)

if __name__ == '__main__':
    init_db()  # Initialize database
    port = 5000  # choose your port

    def open_browser():
        webbrowser.open_new(f"http://127.0.0.1:{port}/")

    # Run browser in a separate thread so it doesn't block Flask
    threading.Timer(1, open_browser).start()

    # Start Flask
    app.run(debug=True, port=port)