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

data_path='sample_img'

pred_label =[]
all_prob=[]

#GENERALIZE THIS
label_true =['lawn_mower','ladjbug', 'guacomole']

tot_cat= list(set(label_true))
'''

for img in onlyfiles:
    #collect predictions
    pred, prob =prediction(img)
    pred_label.append(pred)
    all_prob.append(prob)
'''
#fpr, tpr, thresholds = roc_curve(y, probs)


testy= [1,1,1,1]
lr_probs= [0.8, 0.6,0.3,0.2]

# calculate roc curves
#ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
lr_fpr, lr_tpr, _ = roc_curve(lr_probs, lr_probs)

# plot the roc curve for the model
#plt(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='--', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()