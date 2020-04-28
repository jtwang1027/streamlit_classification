import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

from imagenet import prediction

data_path='sample_img'


### HEADING
st.title('Object Classification')

st.write("Pick an image from the menu/sidebar to view.")

st.write("When you're ready, submit a prediction from the sidebar.")


### SIDEBAR
st.sidebar.info("Welcome")
st.sidebar.title("Capabilities")



#files listed in pulldown
onlyfiles = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]


if st.sidebar.button('Train Model'):
    '''separate python file containing training module '''
    #Training.train()


imageselect = st.sidebar.selectbox("Pick an image.", onlyfiles)

if imageselect:
    #show image
    image = Image.open(imageselect)
    #caption containing prediction
    st.image(image, caption=f'File Selected: {imageselect}', use_column_width=True)    


if st.sidebar.button('Run Prediction'):
    
    label, prob = prediction(imageselect)
    
    #show image
    image = Image.open(imageselect)
    #caption containing prediction
    st.write(f'Class: {label} at Probability: {prob}')

    








#for single file upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Loading Model & Classifying...")
    label, prob = prediction(uploaded_file)
    st.write(f'{label} : {prob}')