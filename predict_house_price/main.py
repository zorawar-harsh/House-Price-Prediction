from flask import Flask, render_template, request
import pandas as pd 
import numpy as np
import pickle
app=Flask(__name__)
with open('RidgeModel.pkl', 'rb') as file:
    pipe = pickle.load(file)
data=pd.read_csv('Cleaned_data.csv')


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('2.html', locations=locations)



@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('BHK')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location,bath, bhk,sqft)

    if not bhk or not bath or not sqft:
        return "Please provide valid inputs for BHK, Bathrooms, and Square Feet."

    try:

        bhk = float(bhk)
        bath = float(bath)
        sqft = float(sqft)
    except ValueError:
        return "Invalid input. Please ensure BHK, Bathrooms, and Square Feet are numbers."
    input =pd.DataFrame([[location,sqft,bath,bhk]],columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction=pipe.predict(input)[0]




    return str(prediction)


if __name__=="__main__":
    app.run(debug=True, port=5001)
