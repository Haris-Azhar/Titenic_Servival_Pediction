import joblib
import streamlit as st
import numpy as np

model = joblib.load('titenic_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Well Come to Titenic survivel Prediction")
st.header('Enter person details')

Pclass = st.number_input('Pclass',1,10)
Sex = st.number_input('Sex',0,1)
Age = st.number_input('Age',1,80)
if st.button('Prediction'):

    new_data = np.array([[Pclass, Sex, Age]])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)

    if prediction[0] == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("💀 Passenger Did Not Survive")




