import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

from imagenet import prediction

data_path='sample_img'

st.header('HEADER')


st.sidebar.info("INFO")
st.sidebar.title("Capabilities")


showpred =1

#files listed in pulldown
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

if st.sidebar.button('Train Model'):
    '''separate python file containing training module '''
    #Training.train()

if st.sidebar.button('Predict'):
    showpred = 1
    #prediction = Testing.predict((model),data_path+ '/' + imageselect)
    

imageselect = st.sidebar.selectbox("Pick an image.", onlyfiles)
if st.sidebar.button('Predict Animal'):
    showpred = 1
    label, prob = prediction(imageselect)



st.title('Object Classification')

st.write("Pick an image from the menu/sidebar to view.")

st.write("When you're ready, submit a prediction on the left.")

st.write("")
image = Image.open(data_path +'/' + imageselect)
st.image(image, caption="Let's predict the animal!", use_column_width=True)

if showpred == 1:
    if prediction == 0:
        st.write("This is a **horse!**")
    if prediction == 1:
        st.write("This is an **elephant!**")
    if prediction == 2:
        st.write("This is a **cat!**")








#for single file upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Loading Model & Classifying...")
    label, prob = prediction(uploaded_file)
    st.write(f'{label} : {prob}')