# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import flasgger
from flasgger import Swagger

# Load the Random Forest CLassifier model
model = pickle.load(open('random_model.pkl', 'rb'))

app = Flask(__name__)
Swagger(app)

@app.route('/')
def Home():
    return "Welcome all"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():

    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=model.predict([[variance,skewness,curtosis,entropy]])

    return "The predicted values is " + str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note by reading csv file
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
            description: The output values

    """
    df_test=pd.read_csv(request.files.get("file"))
    prediction_file=model.predict(df_test)

    return "The predicted values is " + str(list(prediction_file))

if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0',port=8000)
