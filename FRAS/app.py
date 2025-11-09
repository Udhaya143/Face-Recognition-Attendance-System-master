from flask import Flask, render_template
import check_camera
import Capture_Image
import Train_Image
import Recognize
import automail
import os

app = Flask(__name__)

# Ensure required folders exist
for folder in ["Attendance", "TrainingImage", "TrainingImageLabel", "StudentDetails"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_camera')
def check_camera_route():
    check_camera.camer()
    return "<h3>Camera check completed successfully!</h3><a href='/'>Back to Home</a>"

@app.route('/capture_faces')
def capture_faces():
    Capture_Image.takeImages()
    return "<h3>Face capturing completed!</h3><a href='/'>Back to Home</a>"

@app.route('/train_images')
def train_images():
    Train_Image.TrainImages()
    return "<h3>Image training completed and model saved!</h3><a href='/'>Back to Home</a>"

@app.route('/recognize_attendance')
def recognize_attendance():
    Recognize.recognize_attendance()
    return "<h3>Attendance recognition completed!</h3><a href='/'>Back to Home</a>"

@app.route('/auto_mail')
def auto_mail():
    automail.send_mail()
    return "<h3>Email sent successfully!</h3><a href='/'>Back to Home</a>"

if __name__ == '__main__':
    app.run(debug=True)
