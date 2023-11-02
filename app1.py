import streamlit as st
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_model():
    with open('brain_tumor','rb') as f:
        data = pickle.load(f)
    return data

data = load_model()
dec = {0:'No_Tumor', 1:'pituitary_tumor', 2:'meningioma_tumor', 3:'glioma_tumor'} 


st.title("Brain Tumor Classification")
st.write("Upload an MRI scan for tumor classification")

uploaded_file = st.file_uploader("Choose an MRI scan...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    #img = cv2.imread(uploaded_file)
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 0)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = data.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off') 
    
    st.write("Prediction Results:" ,dec[p[0]])
    

