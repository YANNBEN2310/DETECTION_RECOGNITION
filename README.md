# DETECTION_RECOGNITION

This project is a Flask web application for object detection and image segmentation using TensorFlow and OpenCV.

## Access the Application
We deployed the application online so that you can skip all the steps if you just want to test it. The application is also accessible online at: [https://romy-detection-recognition.onrender.com/](https://romy-detection-recognition.onrender.com/)

**Note**: It may have error 502 sometimes since it is a big app and we are students... we have no money, so try to test it a few times if it doesn't work the first time.

**Note**: You can see the code and output of the training of our model in the file `ROMY_DR_PROJECT_NOTEBOOK.html`.

## How to Use
1. **Select an Action**: Choose between 'Detection' and 'Segmentation'.
2. **Upload an Image**: Use the upload button to select an image from your computer.
3. **Click 'Scan'**: Click the 'Scan' button associated with the chosen action.
4. **View Results**:
    - For **Detection**, the app returns the image with bounding boxes around detected objects.
    - For **Segmentation**, the app returns a segmented image with labels for different regions.

## Classes
The application recognizes the following eight classes:
- Vehicle
- Human
- Sky
- Nature
- Object
- Construction
- Flat
- Void

## Run the Flask App Locally

**IMPORTANT**: It is important to run it on Python 3.12.3. It is recommended to use Docker to avoid any issues.

### 1. Create a Virtual Environment

Open your terminal or command prompt and navigate to the project directory. Then create a virtual environment using the following command:

```bash
$ python -m venv myenv
```

### 2. Activate the Virtual Environment

**On Windows**
```bash
$ .\myenv\Scripts\Activate
```
**On linux**
```bash
$ source myenv/bin/activate
```
### 3. Download the Necessary Libraries

Run the following command to install the libraries via the file `requirements.txt`:

```bash
$ pip install -r requirements.txt
```

## Run the App
### 1. Locally
Then, run the application:
```bash
$ python app.py
```
with this the application will be accessible on the first local URL on the terminal: http://127.0.0.1:5001

### Docker 
If you have Docker installed, you can just clone the repository and run the following commands:
```bash
$ docker build -t romy-detection-recognition .
```
This command will build the Docker image for the application. It may take some time, so please be patient. Once it completes, run the following command:

```bash
$ docker run -p 5001:5001 romy-detection-recognition:latest
```

With this, the application will be accessible on the local URL: http://127.0.0.1:5001













