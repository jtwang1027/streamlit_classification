import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

import torch
import json

#eval & plotting
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt


#local file
from imagenet import prediction

data_path='sample_img'


### HEADING
st.title('Object Classification')

st.write("Pick an image from the menu/sidebar to view.")

st.write("When you're ready, submit a prediction from the sidebar.")


### SIDEBAR
st.sidebar.info("Welcome")



#files listed in pulldown
onlyfiles = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
files_no_path = [f for f in listdir(data_path) if isfile(join(data_path, f))]


st.sidebar.title("Single Image Exploration")
imageselect = st.sidebar.selectbox("Pick an image.", onlyfiles)

if imageselect:
    #show image
    image = Image.open(imageselect)
    #caption containing prediction
    st.image(image, caption=f'File Selected: {imageselect}', use_column_width=True)    


if st.sidebar.button('View single prediction'):
    
    label, most_prob, _ = prediction(imageselect)
    
    #show image
    image = Image.open(imageselect)
    #caption containing prediction
    st.write(f'Class: {label} at Probability: {most_prob}')

st.write("") #add newline separator
st.write("")

st.sidebar.title("Whole dataset Analysis")

eval_button=st.sidebar.button('Complete dataset eval /Re-select ROC feature')

stored_cat=pd.read_csv('categories.csv')
keep=stored_cat['filename'].isin(files_no_path) #keep only ones in data_path dir
stored_cat=stored_cat.loc[keep,'real_label'].unique().tolist()
stored_cat= ['']+ stored_cat #creates empty default option
feat_sel = st.sidebar.selectbox("Select feature ROC", stored_cat)

if eval_button:

    @st.cache(suppress_st_warning=True)
    def complete_eval(onlyfiles):
        #input list of files, returns predictions: contains best category for each img
        #all_prob: entire prob distribution for each img
        st.write('Generating predictions...')
        
        weights_warning , my_bar= None, None
        my_bar= st.progress(0) #start
        weights_warning = st.warning('start warning')
        

        pred_label =[]
        removef=[] #remove files not compatible with algo

        for idx, img in enumerate(onlyfiles): #TODO add menu bar


            print(img)
            #collect predictions
            
            try:
                pred, _ ,probab =prediction(img)    
                pred_label.append(pred)                      
            except:
                #remove files not compatible with algorithm
                print('check for black/white')
                removef.append(img) 
            if idx==0: 
                # if it's the first prediction create variable to collect data
                all_prob=torch.unsqueeze(probab,dim=0) #convert to 2D tensor
            else: 
            #probab is a  tensor with prob for each category
                temp=torch.unsqueeze(probab,dim=0) #convert to 2D tensor
                all_prob=torch.cat([all_prob,temp], dim=0) #concatenate and accumulate 

            #update progress bar
            weights_warning.warning(f'Predictions generated for {idx+1} images')
            my_bar.progress((idx+1)/len(onlyfiles))




        print(all_prob.shape)
        onlyfiles = [fil for fil in onlyfiles if fil not in removef ] 
            
        predictions= pd.DataFrame({ 'filename':onlyfiles,
                        'pred_label':pred_label}) 
        predictions['filename']=predictions['filename'].apply(lambda x: x.split('/')[1])


        #read in category label names & format
        file_read = open("imagenet_class_index.json").read()
        categ = json.loads(file_read)
        categ=pd.DataFrame(categ).T
        categ=categ.set_index(0).to_dict()[1]#dict- code: name

        #read in validation data
        val=pd.read_csv('val_annotations.txt', sep='\t', names=['filename','class_id'], usecols=[0,1])
        val['real_label']=val.apply(lambda row: categ[row['class_id']], axis=1)

        

        #merge validation + predictions
        predictions=predictions.merge(val, on='filename',how='left')
        
        #add prob distribution across categories
        all_prob= pd.DataFrame(all_prob.numpy()) #each column is a prob
        all_prob.columns= categ.values()
        output=predictions.merge(all_prob, left_index=True, right_index=True,how='left')

        
        output.to_csv('temp.csv', index=False)
        
        return output        
    
    def cm_plot(predictions):
        #input predictions df to get confusion matrix
        #confusion matrix just needs max prob for each image

        cm=confusion_matrix(predictions['pred_label'], predictions['real_label'])
        print(cm)

        df_cm= pd.DataFrame(cm,index=range(cm.shape[0]),columns= range(cm.shape[0]))
        print(df_cm)
        
        #ADD COLORING
        #add auc score
        sns.heatmap(df_cm)
        plt.title('Confusion Matrix')
        st.pyplot()
        return None
    
    
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
        plt.plot(lr_fpr, lr_tpr, marker='.', label='MobileNet')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.title(f'ROC for {feature}')
        plt.legend()
        # show the plot
        st.pyplot()

        return None 


    #if complete datased eval is selected
    df= complete_eval(onlyfiles=onlyfiles)   
    cm_plot(df) #generate confusion matrix
    if feat_sel:
        ROC(df, feature=feat_sel)
    
