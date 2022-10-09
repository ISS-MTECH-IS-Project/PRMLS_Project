from urllib import response
from flask import Flask, jsonify, request
from flask import render_template
import os
from flask_cors import CORS, cross_origin
from models import *
from datetime import datetime
from PIL import Image
app = Flask(__name__)
cors = CORS(app)

UPLOAD_FOLDER = "./static/images"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
__classifier = Classifier()


def convert_image(file):
    image = Image.open(file)
    image_rgb = image.convert('RGB')
    image_rgb.save(file)


@app.route('/')
@cross_origin()
def hello_world():
    return render_template('index.html')


def uploadFile(file1):
    path = os.path.join(app.config["UPLOAD_FOLDER"], file1.filename)
    file1.save(path)
    convert_image(path)
    return file1.filename


@app.route('/api/classify', methods=['POST'])
@cross_origin()
def classifyImage():
    image = request.files["image"]
    filename = uploadFile(image)
    resp = __classifier.classify(filename)
    return jsonify(resp)
