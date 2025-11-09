import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


st.title("Model Deployment: Diabetes Prediction App")
st.write("This app predicts whether a person is likely to have diabetes using logistic regression.")


# Input fields

def user_input_features():
    Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    Glucose = st.number_input("Glucose", 0, 200, 120)
    BloodPressure = st.number_input("Blood Pressure", 0, 140, 70)
    SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
    Insulin = st.number_input("Insulin", 0, 900, 80)
    BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    Age = st.number_input("Age", 10, 100, 30)
    data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age}

    features = pd.DataFrame(data,index=[0])
    return features

df=user_input_features()
st.subheader('User Input Parameters')
st.write(df)

diabetes=pd.read_csv("diabetes.csv")
diabetes.drop(["Outcome"],inplace=True,axis=1)
diabetes= diabetes.dropna()

X = diabetes.iloc[:,[0,1,2,3,4,5,6,7]]
Y = diabetes.iloc[:,0]
diab=LogisticRegression()
diab.fit(X,Y)



# Predict

if st.button("Predict"):

	prediction = diab.predict(df)[0]
	probability = diab.predict_proba(df)[0][1]

	st.subheader('Predicted Result')

	if prediction == 1:
		st.error(f"The person is **likely diabetic** (Probability: {probability:.2f})")
	else:
		st.success(f"The person is **not likely diabetic** (Probability: {probability:.2f})")


st.markdown("---")
st.caption("Created by Chanakya Dhiman | Built with Streamlit & Logistic Regression")
	