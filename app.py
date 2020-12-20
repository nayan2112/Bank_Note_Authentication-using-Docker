import numpy as np
import pandas as pd
import pickle
import streamlit as st

from PIL import Image

model = pickle.load(open('random_model.pkl', 'rb'))

def welcome():
    return "Welcome to Bank Note Authentication Web App."

def predict_note_authentication(variance,skewness,curtosis,entropy):

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

    prediction=model.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction

def main():
    # st.title("Bank Note Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:8px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Note Authenticator ML App </h2>
    </div>
    <br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    if st.button("About"):
        st.markdown("A Machine Learning Web Application to predict whether the bank note is genuine or forged. Data were extracted from images that were taken from genuine and forged banknote-like specimens and can be used for binary classification problems. Wavelet Transform tool were used to extract features from images.")
    # if st.button("Predict the Bank Note"):
    st.header("Predict the Bank Note")
    variance = st.text_input("Variance")
    skewness = st.text_input("skewness")
    curtosis = st.text_input("curtosis")
    entropy = st.text_input("entropy")
    result = ""
    if st.button("Predict"):
        result=predict_note_authentication(variance,skewness,curtosis,entropy)
        if result == [1]:
            result = "[Genuine Note]"
        else:
            result = "[Forged Note]"
    st.success('The Bank Note is {}'.format(result))
    # if st.button("About"):
    #     st.markdown("A Machine Learning Web Application to predict whether the bank note is genuine or forged. Data were extracted from images that were taken from genuine and forged banknote-like specimens and can be used for binary classification problems. Wavelet Transform tool were used to extract features from images.")
    #     #st.text("A Machine Learning Web Application to predict whether the bank note is genuine or forged. Data were extracted from images that were taken from genuine and forged banknote-like specimens and can be used for binary classification problems. Wavelet Transform tool were used to extract features from images."
    st.markdown("@2020 Nayan Jain")
    st.markdown("Bulit with Streamlit")

if __name__ == "__main__":
    main()
