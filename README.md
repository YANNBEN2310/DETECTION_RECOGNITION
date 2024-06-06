# DETECTION_RECOGNITION

This project is a Flask web application for object detection and image segmentation using TensorFlow and OpenCV.

## Run the Flask App

### 1. Create a Virtual Environment

Open your terminal or command prompt and navigate to the project directory. Then create a virtual environment using the following command:

```bash
python3 -m venv myenv
```

### 2. Activate the Virtual Environment

**On Windows**

```bash
/MY_PATH$.\myenv\Scripts\Activate
```

**On Linux**

```bash
/MY_PATH$source myenv/bin/activate
```

### 3. Download the Necessary Libraries
Install the following libraries via the file requirements.txt

```bash
/MY_PATH$pip install -r requirements.txt
```

If this doesn't work, install the requirements individually:

```bash
/MY_PATH$pip install flask
/MY_PATH$pip install flask-socketio
/MY_PATH$pip install opencv-python
/MY_PATH$pip install numpy
/MY_PATH$pip install eventlet
/MY_PATH$pip install tensorflow
```

### 4. Run the App

Ensure you have the model files in the models directory:

- models/model.h5
- models/yolov4-tiny.weights
- models/yolov4-tiny.cfg
- models/coco.names

Then, run the application:

```bash
$python app.py
```

The application will start running on http://127.0.0.1:5001.

