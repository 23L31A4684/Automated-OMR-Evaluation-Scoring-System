import streamlit as st
import requests
import os
from PIL import Image
import pandas as pd

st.title("Automated OMR Evaluation System")

# Upload Image
student_id = st.text_input("Student ID", "101")
version = st.selectbox("Version", ["A", "B"])
uploaded_file = st.file_uploader("Upload OMR Image (jpeg)", type=["jpeg", "jpg"])

if uploaded_file:
    # Save uploaded file
    image_path = f"samples/uploaded_{student_id}.jpeg"
    os.makedirs("samples", exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Preview
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Sheet", use_container_width=True)
    
    # Upload Key from Excel (manual for now)
    if st.button("Upload Key from Excel"):
        st.info("Ensure key.xlsx is in the project root with sheets 'Set - A' and 'Set - B'.")

# Evaluate
if st.button("Evaluate"):
    if os.path.exists(image_path):
        response = requests.get("http://127.0.0.1:8000/evaluate", params={
            "student_id": student_id,
            "version": version,
            "image_path": image_path
        })
        if response.status_code != 200:
            st.error(f"Backend Error {response.status_code}: {response.text}")
        else:
            try:
                result = response.json()
                st.success("Evaluation Successful!")
                st.write("Results:", result)
            except requests.exceptions.JSONDecodeError:
                st.error("Invalid response from backend. Check backend logs.")
    else:
        st.error("Image not uploaded or path invalid.")

# Dashboard
st.subheader("Results Dashboard")
response = requests.get("http://127.0.0.1:8000/get_all_results/")
if response.status_code == 200:
    try:
        results = response.json()
        if results and "error" not in results[0]:  # Check if results are valid
            df = pd.DataFrame(results)
            st.dataframe(df)
            st.download_button("Export CSV", df.to_csv(index=False), "results.csv", "text/csv")
        else:
            st.error(f"Backend Error: {results.get('error', 'No results')}")
    except requests.exceptions.JSONDecodeError:
        st.error("Invalid response from backend for results.")
else:
    st.error(f"Failed to fetch results: {response.status_code} - {response.text}")