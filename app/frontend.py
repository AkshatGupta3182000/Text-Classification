import streamlit as st
import requests

st.title("Text Classification App")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type = ["csv","xlsx"])

if uploaded_file is not None:
    if st.button("Get Predictions"):
        files = {"file":(uploaded_file.name,uploaded_file,uploaded_file.type)}
        url = "https://text-classification-7whk.onrender.com/predict_file"
        with st.spinner("Sending file for the Predictions..."):
            response = requests.post(url,files=files)

        if response.status_code == 200:
            st.success("Predictions received!")
            st.download_button(
                label = "Download Predictions",
                data = response.content,
                file_name= f"{uploaded_file.name.split('.')[0]}_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:
            st.error(f"Error: {response.json().get('detail', 'Unexpected error')}")