import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
 
sio = socketio.Server()
 
app = Flask(__name__) #'__main__'
speed_limit = 10


def preprocess(image):
    '''Takes the path of the image and returns the preprocessed result'''

    # Crop image to remove unnecessary features (front part of the car, skies, etc)
    img = image[60:-25, :, :]

    # Decrease size for easier processing and for interfacing with the model architecture from NVIDIA's paper
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    
    # Convert from RGB to YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    # Normalize values (values will be between 0-1)
    img = img / 255

  return img
 
 
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
 
 
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)
 
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
 
 
if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)