from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import eventlet
import eventlet.wsgi
import os
import tempfile
import tensorflow as tf
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')
UPLOAD_FOLDER = 'static/uploaded_files'
TEMP_UPLOAD_FOLDER = 'temp_uploads'  # Temporary upload folder
MODEL_FOLDER = 'models'  # Folder to save uploaded models

# Create the necessary folders if they don't exist
os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_UPLOAD_FOLDER'] = TEMP_UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 500  

# Set environment variable for temp directory
os.environ['TMPDIR'] = app.config['TEMP_UPLOAD_FOLDER']

# Load the default pre-trained model for segmentation
model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'model.h5'))

# Load YOLO for object detection
net = cv2.dnn.readNet(os.path.join(MODEL_FOLDER, "yolov4-tiny.weights"), os.path.join(MODEL_FOLDER, "yolov4-tiny.cfg"))
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open(os.path.join(MODEL_FOLDER, "coco.names"), "r") as f:
    classes = [line.strip() for line in f.readlines()]
# Map YOLO categories to Font Awesome icons
category_icon_mapping = {
    "person": "fas fa-user",
    "bicycle": "fas fa-bicycle",
    "car": "fas fa-car",
    "motorcycle": "fas fa-motorcycle",
    "airplane": "fas fa-plane",
    "bus": "fas fa-bus",
    "train": "fas fa-train",
    "truck": "fas fa-truck",
    "boat": "fas fa-ship",
    "traffic light": "fas fa-traffic-light",
    "fire hydrant": "fas fa-fire-extinguisher",
    "stop sign": "fas fa-stop-circle",
    "parking meter": "fas fa-parking",
    "bench": "fas fa-chair",
    "bird": "fas fa-dove",
    "cat": "fas fa-cat",
    "dog": "fas fa-dog",
    "horse": "fas fa-horse",
    "sheep": "fas fa-sheep",
    "cow": "fas fa-cow",
    "elephant": "fas fa-elephant",
    "bear": "fas fa-bear",
    "zebra": "fas fa-zebra",
    "giraffe": "fas fa-giraffe",
    "backpack": "fas fa-backpack",
    "umbrella": "fas fa-umbrella",
    "handbag": "fas fa-handbag",
    "tie": "fas fa-tie",
    "suitcase": "fas fa-suitcase",
    "frisbee": "fas fa-compact-disc",
    "skis": "fas fa-skiing",
    "snowboard": "fas fa-snowboarding",
    "sports ball": "fas fa-basketball-ball",
    "kite": "fas fa-fighter-jet",
    "baseball bat": "fas fa-baseball-bat",
    "baseball glove": "fas fa-baseball-glove",
    "skateboard": "fas fa-skating",
    "surfboard": "fas fa-surfing",
    "tennis racket": "fas fa-table-tennis",
    "bottle": "fas fa-wine-bottle",
    "wine glass": "fas fa-wine-glass",
    "cup": "fas fa-coffee",
    "fork": "fas fa-utensil-fork",
    "knife": "fas fa-utensil-knife",
    "spoon": "fas fa-utensil-spoon",
    "bowl": "fas fa-bowl",
    "banana": "fas fa-banana",
    "apple": "fas fa-apple-alt",
    "sandwich": "fas fa-sandwich",
    "orange": "fas fa-orange",
    "broccoli": "fas fa-broccoli",
    "carrot": "fas fa-carrot",
    "hot dog": "fas fa-hotdog",
    "pizza": "fas fa-pizza",
    "donut": "fas fa-donut",
    "cake": "fas fa-cake",
    "chair": "fas fa-chair",
    "couch": "fas fa-couch",
    "potted plant": "fas fa-seedling",
    "bed": "fas fa-bed",
    "dining table": "fas fa-table",
    "toilet": "fas fa-toilet",
    "tv": "fas fa-tv",
    "laptop": "fas fa-laptop",
    "mouse": "fas fa-mouse",
    "remote": "fas fa-tv-retro",
    "keyboard": "fas fa-keyboard",
    "cell phone": "fas fa-mobile-alt",
    "microwave": "fas fa-microphone",
    "oven": "fas fa-oven",
    "toaster": "fas fa-toaster",
    "sink": "fas fa-sink",
    "refrigerator": "fas fa-refrigerator",
    "book": "fas fa-book",
    "clock": "fas fa-clock",
    "vase": "fas fa-vase",
    "scissors": "fas fa-scissors",
    "teddy bear": "fas fa-bear",
    "hair drier": "fas fa-wind",
    "toothbrush": "fas fa-tooth",
    # Segmentation categories
    "void": "fas fa-question",
    "flat": "fas fa-square",
    "construction": "fas fa-building",
    "object": "fas fa-cube",
    "nature": "fas fa-leaf",
    "sky": "fas fa-cloud",
    "human": "fas fa-user",
    "vehicle": "fas fa-car"
}

# Function to get the Font Awesome class for a given category
def get_icon(category):
    return category_icon_mapping.get(category, "fas fa-question")  # Default to a question mark if not found

# Add get_icon to the template context
@app.context_processor
def utility_processor():
    return dict(get_icon=get_icon)

def detect_objects(image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = []
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            detected_objects.append((label, confidence))
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image, detected_objects

def process_image(image_path):
    image = cv2.imread(image_path)
    processed_image, detected_objects = detect_objects(image)
    result_image_filename = os.path.basename(image_path) + ".jpg"
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_image_filename)
    cv2.imwrite(result_image_path, processed_image)
    print(f"Processed image saved at: {result_image_path}")  # Debugging line
    num_objects = len(detected_objects)
    return result_image_filename, detected_objects, num_objects

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return "Error processing video.", []

    processed_frame, detected_objects = detect_objects(frame)
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(video_path) + ".jpg")
    cv2.imwrite(result_image_path, processed_frame)
    print(f"Processed frame saved at: {result_image_path}")  # Debugging line
    cap.release()
    return result_image_path, detected_objects

import plotly.express as px
import plotly.io as pio
import json

def segment_image(image_path, model_path=None):
    global model
    if model_path:
        model = tf.keras.models.load_model(model_path)
    
    original_image = cv2.imread(image_path)
    processed_image = original_image.copy()
    
    resized_image = cv2.resize(original_image, (128, 128))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    resized_image = np.expand_dims(resized_image, axis=0) / 255.0
    
    predictions = model.predict(resized_image)
    pred_masks = np.argmax(predictions, axis=-1)  # Enlever [0] ici
    predicted_confidences = np.max(predictions, axis=-1)  # Extraire les scores de confiance

    # Gestion de la dimension du lot
    pred_masks = np.squeeze(pred_masks, axis=0)  # Supprimer la dimension du lot si présente
    predicted_confidences = np.squeeze(predicted_confidences, axis=0)  # Supprimer la dimension du lot si présente
    
    pred_masks = cv2.resize(pred_masks, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    predicted_confidences = cv2.resize(predicted_confidences, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    label_id_to_category_id = {
        0: 'void',
        1: 'flat',
        2: 'construction',
        3: 'object',
        4: 'nature',
        5: 'sky',
        6: 'human',
        7: 'vehicle'
    }
    
    detected_categories = []
    category_counts = {}
    color_mapping = {}

    # Utiliser l'analyse des composants connectés pour compter les instances de chaque catégorie
    for category in np.unique(pred_masks):
        mask = pred_masks == category
        num_labels, labels_im = cv2.connectedComponents(mask.astype(np.uint8))

        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        for c in range(3):
            processed_image[:, :, c] = np.where(mask, color[c], processed_image[:, :, c])
        
        category_name = label_id_to_category_id.get(category, f'Unknown({category})')
        confidence_score = np.mean(predicted_confidences[mask])
        detected_categories.append((category, category_name, color.tolist(), confidence_score))

        if category_name not in category_counts:
            category_counts[category_name] = 0
        # Soustraire 1 pour ignorer le label de fond
        category_counts[category_name] += num_labels - 1
        color_mapping[category_name] = f'rgb({color[0]}, {color[1]}, {color[2]})'
        print(f"Category: {category_name}, Icon: {get_icon(category_name)}")  # Ligne de débogage
    
    result_image_filename = os.path.basename(image_path) + "_seg.jpg"
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_image_filename)
    cv2.imwrite(result_image_path, processed_image)

    original_image_filename = os.path.basename(image_path) + "_orig.jpg"
    original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_image_filename)
    cv2.imwrite(original_image_path, original_image)
    
    return original_image_filename, result_image_filename, detected_categories, category_counts


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('file')
        if not files or files[0].filename == '':
            return redirect(request.url)
        results = []
        for file in files:
            if file.filename.split('.')[-1] in ['mp4', 'avi', 'mov']:
                if len(files) > 1:
                    result = "Please upload only one video file at a time."
                    return render_template('result.html', result=result, results=[])
                with tempfile.NamedTemporaryFile(delete=False, dir=app.config['TEMP_UPLOAD_FOLDER']) as tmp:
                    file_path = tmp.name
                    file.save(file_path)
                result_image_filename, detected_objects = process_video(file_path)
                results.append((result_image_filename, detected_objects, len(detected_objects)))
            elif file.filename.split('.')[-1] in ['jpg', 'jpeg', 'png']:
                with tempfile.NamedTemporaryFile(delete=False, dir=app.config['TEMP_UPLOAD_FOLDER']) as tmp:
                    file_path = tmp.name
                    file.save(file_path)
                result_image_filename, detected_objects, num_objects = process_image(file_path)
                results.append((result_image_filename, detected_objects, num_objects))
            else:
                result = "Please upload a valid video or image file for detection."
                return render_template('result.html', result=result, results=[])
        return render_template('result.html', result="Detection Complete", results=results)
    return render_template('index.html')


@app.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    model_path = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('file')
        if not files or files[0].filename == '':
            return redirect(request.url)
        if 'model_file' in request.files:
            model_file = request.files['model_file']
            if model_file and model_file.filename != '':
                model_path = os.path.join(app.config['MODEL_FOLDER'], model_file.filename)
                model_file.save(model_path)
        results = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, dir=app.config['TEMP_UPLOAD_FOLDER']) as tmp:
                file_path = tmp.name
                file.save(file_path)
            if file.filename.split('.')[-1] in ['jpg', 'jpeg', 'png']:
                original_image_filename, result_image_filename, detected_categories, category_counts = segment_image(file_path, model_path)
                results.append((original_image_filename, result_image_filename, detected_categories, category_counts))
            else:
                result = "Please upload a valid image file for segmentation."
                return render_template('segmentation.html', result=result, results=[])
        return render_template('segmentation.html', result="Segmentation Complete", results=results)
    return render_template('segmentation.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

