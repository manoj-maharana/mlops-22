from flask import Flask
from flask import request
from joblib import load
import numpy as np

app = Flask(__name__)
model_path = "svm_gamma=0.005_C=1.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


# get x and y somehow    
#     - query parameter
#     - get call / methods
#     - post call / methods ** 

@app.route("/sum", methods=['POST'])
def sum():
    x = request.json['x']
    y = request.json['y']
    z = x + y 
    return {'sum':z}



@app.route("/predict", methods=['POST'])
def predict_digit():
    image = request.json['image']
    print("done loading")
    predicted = model.predict([image])
    return {"y_predicted":int(predicted[0])}

@app.route("/predictmultipleimages", methods=['POST'])
def predict_multiple_image_digit():
    image_1 = request.json['image']
    image_2 = request.json['image']
    print("done loading")
    predicted_1 = model.predict([image_1])
    predicted_2 = model.predict([image_2])
    return {"both matched:": str(predicted_1[0] == predicted_2[0])}

@app.route("/predictmultipleimages", methods=['POST'])
def predict_multiple_image_digit():
    image_1 = request.json['image']
    image_2 = request.json['image']
    print("done loading")
    predicted_1 = model.predict([image_1])
    predicted_2 = model.predict([image_2])
    return {"both matched:": str(predicted_1[0] == predicted_2[0])}

if __name__ == "__main__":
    
    app.run(host='0.0.0.0', port=5000)