import streamlit as st
import pandas as pd
import numpy as np

st.header('test h')
st.title('Uber pickups in NYC')

df=pd.read_csv('https://github.com/jtwang1027/pyspark_aws/raw/master/logP_dataset.csv')

st.write(df.head())