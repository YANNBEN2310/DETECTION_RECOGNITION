from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
import tempfile
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')
UPLOAD_FOLDER = 'static/uploaded_files'
TEMP_UPLOAD_FOLDER = 'temp_uploads'  # Temporary upload folder

# Create the temporary upload folder if it doesn't exist
os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_UPLOAD_FOLDER'] = TEMP_UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 500  

# Set environment variable for temp directory
os.environ['TMPDIR'] = app.config['TEMP_UPLOAD_FOLDER']

# Load YOLO
net = cv2.dnn.readNet("models/yolov4-tiny.weights", "models/yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

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

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            detected_objects.append((label, confidence))
            color = (0, 255, 0)  # Green color for the rectangle
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
    return result_image_filename, detected_objects

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return "Error processing video.", []

    processed_frame, detected_objects = detect_objects(frame)
    result_image_filename = os.path.basename(video_path) + ".jpg"
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_image_filename)
    cv2.imwrite(result_image_path, processed_frame)
    print(f"Processed frame saved at: {result_image_path}")  # Debugging line
    cap.release()
    return result_image_filename, detected_objects

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
                results.append((result_image_filename, detected_objects))
            elif file.filename.split('.')[-1] in ['jpg', 'jpeg', 'png']:
                with tempfile.NamedTemporaryFile(delete=False, dir=app.config['TEMP_UPLOAD_FOLDER']) as tmp:
                    file_path = tmp.name
                    file.save(file_path)
                result_image_filename, detected_objects = process_image(file_path)
                results.append((result_image_filename, detected_objects))
            else:
                result = "Please upload a valid video or image file for detection."
                return render_template('result.html', result=result, results=[])
        return render_template('result.html', result="Detection Complete", results=results)
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)
