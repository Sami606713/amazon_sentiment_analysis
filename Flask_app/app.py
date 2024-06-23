from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from utils import load_model,processor


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html',result=None)

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/predict",methods=['POST'])
def predict():
    text = request.form['user_text']
    # Add your sentiment analysis logic here
    input_df=pd.DataFrame({"verified_reviews":[text]})

    # Load the model
    model=load_model()

    # predict the model
    result=model.predict(input_df)[0]
    if result==0:
        result="Negative"
    elif(result==1):
        result="Positive"
    print("User text:  ",text)
    print("text Result:  ",result)
    # Return a response to the client (you can render a template or return JSON)
    return render_template('index.html', result=result)


if __name__=="__main__":
    app.run(debug=True)