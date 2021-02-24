import numpy as np
from flask import Flask, request, render_template
import pickle

from fastai.vision import *
from fastai import *

import os

cwd = os.getcwd()
path = Path()

application = Flask(__name__)

model = load_learner(path, '/exportt.pkl')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    file = request.files['file']

    if file:
        filename = file.filename
        file.save(os.path.join("resources/tmp", filename))

    to_predict = "resources/tmp/"+filename
    img = open_image(to_predict)

    #Getting the prediction from the model
    prediction = model.predict(img)[0]

    #Render the result in the html template
    return render_template('index.html', prediction_text='Your Prediction :  {} '.format(prediction))
© 2021 GitHub, Inc.
