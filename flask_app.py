# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load the Random Forest CLassifier model
model = pickle.load(open('random_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def Home():
    return "Welcome all"

@app.route('/predict', methods=['GET'])
def predict_note_authentication():
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=model.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted values is " + str(prediction)

@app.route('/predict_file', methods=['POST'])
def predict_note_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction_file = model.predict(df_test)
    return "The predicted values is " + str(prediction_file)


if __name__ == "__main__":
    app.run(debug=True)
