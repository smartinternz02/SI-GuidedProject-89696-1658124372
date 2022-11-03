# importing the necessary dependencies
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import load_model
import pickle
import os

app = Flask(__name__)
model = load_model("seedling_exception.h5")

@app.route('/')
def home():
    #return render_template('index.html')
    try:
        return render_template("index.html")
    except Exception as e:
        return str(e)

@app.route('/Prediction')
def prediction():
    return render_template('predict.html')

@app.route('/predict', methods=["POST", "GET"])  # route to show the predictions in a web
def upload():
    if request.method == 'POST':
        f = request.files['img']
        print("current path")
        basepath = os.path.dirname("_file_")
        print("current path", basepath)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # a= x.shape        
        preds = np.argmax(model.predict(x), axis=1)        
        # print("prediction", preds)
        index = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered-Cranesbill', 'Sugar beet']
        text = "The predicted seedling is : " + str(index[preds[0]])
        # return str(text)
        return render_template("predict.html", z=text)

if __name__ == "__main__":
    app.run(debug=False)
