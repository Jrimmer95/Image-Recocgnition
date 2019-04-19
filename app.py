#Jordan, Morsal, Parthiv
import os

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from image_recognition_final import *


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "media")

app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_DIR


@app.route('/')
def index():
    return render_template('index_body.html')


@app.route('/results', methods=['GET'])
@app.route('/results/<file>', methods=['GET'])
def results(file=None):

    if file:
        data = {}
        data_other = {}

        # this is the path of the image on the disk that has just been uploaded
        img = os.path.join(UPLOAD_DIR, file)

        # sends the image to the prediction network and returns the first element in the list of predictions
        prediction_list = prediction(img)[0]
        predictionparam = prediction_list[0]
        prediction_rest = prediction_list[1:]
        #print(prediction_list)
        #print(prediction_rest)
        print("Most Likely")
        print(predictionparam)
        for i in prediction_rest:
            resnetid, identity, likelyhood = i
            data_other[identity] = {'prediction': identity,'prediction_percentage': likelyhood}
        resnetid, identity, likelyhood = predictionparam
        print("Similar Categories")
        print(data_other)
        data['prediction_1'] = {'file_name': url_for('uploaded_file', filename=file),
                                'prediction': identity,'prediction_percentage': likelyhood}

    return render_template('predict_results_body.html', data=data, data2=data_other)


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    """
    Upload path.
    """
    if request.method == 'POST':
        _file = request.files['file']
        _file.save(os.path.join(UPLOAD_DIR, _file.filename))
        return "Success!"

    return redirect(url_for('/'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Creates a url for media files.
    """
    return send_from_directory(app.config['UPLOAD_DIR'], filename)


if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0')
