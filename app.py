from flask import Flask, render_template, request, url_for
import torch
import os
import shutil
import sys
import pathlib
from PIL import Image

# Ensure YOLOv5 is in the system path
app = Flask(__name__)

if os.name == 'nt':  # Fix for Windows compatibility
    pathlib.PosixPath = pathlib.WindowsPath

YOLO_PATH = r'C:\Users\Deepak\OneDrive\Desktop\UOW\Projects\ChestAbnormality\yolov5-master'
sys.path.insert(0, YOLO_PATH)

# Import detect from YOLOv5
import detect  # Ensure you have detect.py in your YOLOv5 directory

# Load your custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file provided", 400

    file = request.files['file']
    img_bytes = file.read()

    # Save the uploaded image
    input_image_path = 'input_image.jpg'
    with open(input_image_path, 'wb') as f:
        f.write(img_bytes)

    # Clear the result folder
    result_folder = 'static/result'
    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)
    os.makedirs(result_folder)

    # Run the detection using detect.py function with additional parameters
    # Calling detect.py with line_thickness
    detect.run(
        weights='best.pt',            # Model weights
        source=input_image_path,      # Input image path
        imgsz=(1024, 1024),           # Image size
        conf_thres=0.25,              # Confidence threshold
        iou_thres=0.45,               # IOU threshold
        save_txt=False,               # Don't save txt files
        save_conf=False,              # Don't save confidence scores in txt
        save_crop=False,              # Don't crop detected objects
        project=result_folder,        # Where to save the results
        name='',                      # No name, overwrite previous results
        exist_ok=True,                # Overwrite the folder if exists
        line_thickness=1              # Set the bounding box line thickness
    )


    # Fetch the results
    output_image_path = os.path.join(result_folder, 'input_image.jpg')

    # Extract the predictions
    predictions = []
    results = model(input_image_path)  # Run model on image
    for pred in results.xyxy[0]:       # Iterate through each detected object
        class_id = int(pred[5])        # Class ID
        confidence = float(pred[4])    # Confidence score
        label = model.names[class_id]  # Get class label from class ID
        predictions.append({"label": label, "confidence": confidence})

    return render_template('result.html', output_image=output_image_path, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
