import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('Student_Performance.pkl')
encoded = joblib.load('LabelEncoded.pkl')
scaler = joblib.load('Standard_Scaler.pkl')

st.title('Prediction App')
st.write('Student Performance')

hour = st.number_input('How many hours did you study')
previous_score = st.number_input('Enter your previous score')
sleep = st.number_input('How many hours did you sleep')
practice = st.number_input('How many question papers did you practice')
extra_activity = st.selectbox('Have you participated in any extracurricular activities',["Yes","No"])

encoded_value = encoded.fit_transform([extra_activity])[0]

user_data= np.array([[hour,previous_score,sleep,practice,encoded_value]])
scaled = scaler.transform(user_data)


predicts = model.predict(scaled)
pd = pd.Series(predicts[0])
pr = round(pd)
if st.button('predict'):
    st.write(f'predicted price of the given data is {pr[0]}')