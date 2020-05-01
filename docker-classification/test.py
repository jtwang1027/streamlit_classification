from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

#eval & plotting
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt


#local file
from imagenet import prediction

df= pd.read_csv('temp.csv')

#feat_sel = st.sidebar.selectbox("Plot feature ROC", df['real_label'].unique().tolist() )

def ROC(df, feature):
    #input predictions df + desired feature (chosen from dropdown) to get ROC and AUC

    #score= roc_auc_score(predictions)
    #testy: 0 or 1
    #lr_probs: model predicted probabilities
    nrow, _ =df.shape 

    testy= np.zeros(nrow,)
    hits=np.where(df['real_label']==feature)[0]
    testy[hits]= 1 
    
    lr_fpr, lr_tpr, _ = roc_curve(testy, df[feature])

    plt.plot([0,1], [0,1], linestyle='--')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Imagenet')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.title(f'ROC for {feature}')
    plt.legend()
    # show the plot
    plt.pyplot()

    return None 

ROC(df, feature=feat_sel)



