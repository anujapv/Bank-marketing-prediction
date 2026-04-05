import streamlit as st
from PIL import Image    
import joblib
import numpy as np


model=joblib.load('bank.pkl')
le1=joblib.load('le1.pkl')
le2=joblib.load('le2.pkl')
scaler=joblib.load('scaler.pkl')

st.title('Bank Data Analysis')
st.header('Data Analysis')
st.subheader('Data Description')
st.write('Enter description here')



options=['unknown','failure','success','other']
poutcome=st.selectbox('Previous campaign outcome:',options)


housing_opt=['yes','no']
housing_loan=st.selectbox('Do you have any housing loan:',housing_opt)

balance=st.number_input('Account Balance in euros:',value=0)

campaign_raw=st.number_input('No.of calls during campaign:',value=0)
previous_raw=st.number_input('No: of previous calls:',value=0)


if st.button("Predict"):
    try:
       
        poutcome_enc = le2.transform([poutcome])[0]
        housing_enc = le1.transform([housing_loan])[0]
        

        
        input_data = np.array([[
            poutcome_enc,
            housing_enc,
            balance,
            campaign_raw,
            previous_raw
        ]])

        X_scaled = joblib.load('scaler.pkl')

        input_scaled = scaler.transform(input_data)


        prediction = model.predict(input_scaled)[0]


        if prediction == 1:
            st.success("Prediction: Customer WILL subscribe")
        else:
            st.error("Prediction: Customer will NOT subscribe")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
