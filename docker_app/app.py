import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from imagenet import prediction


st.header('test h')
st.title('Uber pickups in NYC')

#df=pd.read_csv('https://github.com/jtwang1027/pyspark_aws/raw/master/logP_dataset.csv')
#st.write(df.head())

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Loading Model & Classifying...")
    label, prob = prediction(uploaded_file)
    st.write(f'{label} : {prob}')