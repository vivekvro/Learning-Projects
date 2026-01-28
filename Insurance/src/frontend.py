import streamlit as st
import requests


api_url = " http://127.0.0.1:8000/charges"

st.title("Insurance Charges Predictor")
st.markdown("Enter your details below:")

age = st.number_input("Age",min_value=1,max_value=72,value=30)
sex= st.selectbox("Gender",options=['male','female'])
height = st.number_input("Height (m)",min_value=0.5,max_value=2.5,value=1.72)
weight = st.number_input("Weight (kg)",min_value=10.1,value=66.3)
children = st.number_input("Children count",min_value=0)
smoker= st.selectbox("do you smoke",options=['yes','no'])
region = st.selectbox("select your region",options=['southeast','southwest','northeast','northwest'])
if st.button("Predict Insurance Charges"):

    input_data = {
        "age": age,
        "sex": sex,
        "height":height,
        'weight':weight,
        "children": children,
        "smoker": smoker,
        "region": region
    }

    try:
        response = requests.post(api_url, json=input_data)
        result = response.json()

        if response.status_code == 200 and "predicted" in result:
            st.success(f"Predicted charges: **{result['predicted']}**")
        else:
            st.error(f"API error: {response.status_code}")
            st.write(result)

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure it's running.")
