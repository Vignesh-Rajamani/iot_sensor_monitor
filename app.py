from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import json
from datetime import datetime
import threading
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# In-memory data storage (replace with database in production)
sensor_data = []
anomalies = []
alerts = []

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC = os.getenv('MQTT_TOPIC', 'sensor/data')

# Anomaly Detection Model
clf = IsolationForest(contamination=0.05, random_state=42)

# MQTT Client Setup
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        process_sensor_data(data)
    except Exception as e:
        print(f"Error processing MQTT message: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

def start_mqtt():
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_forever()

# Data Processing
def process_sensor_data(data):
    # Add timestamp if not present
    if 'timestamp' not in data:
        data['timestamp'] = datetime.now().isoformat()
    
    # Store data
    sensor_data.append(data)
    
    # Check for anomalies
    check_anomalies(data)
    
    # Emit to dashboard
    socketio.emit('new_data', data)
    
    # Keep only recent data (e.g., last 1000 entries)
    if len(sensor_data) > 1000:
        sensor_data.pop(0)

def check_anomalies(data):
    try:
        # Convert data to features (simplified example)
        features = np.array([[data.get('temperature', 0), 
                            data.get('humidity', 0), 
                            data.get('pressure', 0)]])
        
        # Predict anomaly
        prediction = clf.predict(features)
        
        if prediction[0] == -1:
            anomaly = {
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'type': 'anomaly_detected'
            }
            anomalies.append(anomaly)
            alerts.append(anomaly)
            socketio.emit('new_alert', anomaly)
            print(f"Anomaly detected: {data}")
    except Exception as e:
        print(f"Error in anomaly detection: {e}")

# API Endpoints
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/data', methods=['GET', 'POST'])
def handle_data():
    if request.method == 'POST':
        if request.content_type == 'application/json':
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        process_sensor_data(data)
        return jsonify({"status": "success"})
    
    # GET request - return recent data
    return jsonify(sensor_data[-100:])

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                process_sensor_data(row.to_dict())
            return jsonify({"status": "success", "records_processed": len(df)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file format"}), 400

@app.route('/api/anomalies')
def get_anomalies():
    return jsonify(anomalies[-50:])

@app.route('/api/alerts')
def get_alerts():
    return jsonify(alerts[-20:])

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send initial data
    socketio.emit('init_data', {'sensor_data': sensor_data[-100:], 'anomalies': anomalies[-20:]})

if __name__ == '__main__':
    # Start MQTT client in a separate thread
    mqtt_thread = threading.Thread(target=start_mqtt)
    mqtt_thread.daemon = True
    mqtt_thread.start()
    
    # Start Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)