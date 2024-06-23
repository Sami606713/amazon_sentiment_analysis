import streamlit as st
import numpy as np
import pandas as pd
from utils import load_model,processor

st.title("Amazon Sentiment Analysis")

text=st.text_input("Enter your text here: ")

if st.button("Check Sentiment"):
    # Convert the text into dataframe
    input_df=pd.DataFrame({"verified_reviews":[text]})

    # Load the model
    model=load_model()

    # predict the model
    result=model.predict(input_df)[0]
    if result==0:
        st.error("Review is negative")

    elif(result==1):
        st.success("Review is positive")
    # st.write(result)