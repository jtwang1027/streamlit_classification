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
from imagenet import predict

data_path='sample_img'


### HEADING
st.title('Object Classification')

st.write("Pick an image from the menu/sidebar to view.")

st.write("When you're ready, submit a prediction from the sidebar.")


### SIDEBAR
st.sidebar.info("Welcome")

model_sel = st.sidebar.selectbox("Select pretrained model",[
                                            'vgg11',
                                            'alexnet',
                                            'resnet18',
                                            'mobilenet_v2'])

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
    
    print(model_sel)
    label, most_prob, _ = predict(imageselect, model_sel)
    
    #show image
    image = Image.open(imageselect)
    #caption containing prediction
    st.write(f'Class: {label}')
    st.write(f'Probability: {most_prob}')

st.write("") #add newline separator
st.write("")

st.sidebar.title("Whole dataset Analysis")

eval_button=st.sidebar.button('Complete dataset eval /Re-select ROC feature')

#stored_cat=pd.read_csv('categories.csv')
#keep=stored_cat['filename'].isin(files_no_path) #keep only ones in data_path dir
#stored_cat=stored_cat.loc[keep,'real_label'].unique().tolist()

feat_sel = st.sidebar.selectbox("Select feature ROC", 
        ['',
        'bucket','sports_car', 'desk',
        'sewing_machine','monarch','chimpanzee', 'stopwatch', 'lampshade'
        ]
)

if eval_button:

    @st.cache(suppress_st_warning=True)
    def complete_eval(onlyfiles):
        #input list of files, returns predictions: contains best category for each img
        #all_prob: entire prob distribution for each img
        
        st.write("loading model...")
        st.write('Generating predictions...')
        
        weights_warning , my_bar= None, None
        my_bar= st.progress(0) #start
        weights_warning = st.warning('...')
        
        print(model_sel)

        pred_label =[]
        removef=[] #remove files not compatible with algo

        for idx, img in enumerate(onlyfiles): #TODO add menu bar


            print(img)
            #collect predictions
            
            try:
                pred, _ ,probab =predict(img, model_sel= model_sel)    
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
        #val=pd.read_csv('val_annotations.txt', sep='\t', names=['filename','class_id'], usecols=[0,1])
        #val['real_label']=val.apply(lambda row: categ[row['class_id']], axis=1)
        val=pd.read_csv('validation.csv')
        

        

        #merge validation + predictions
        predictions=predictions.merge(val, on='filename',how='left')
        
        #add prob distribution across categories
        all_prob= pd.DataFrame(all_prob.numpy()) #each column is a prob
        all_prob.columns= categ.values()
        output=predictions.merge(all_prob, left_index=True, right_index=True,how='left')

        
        output.to_csv('predictions_output.csv', index=False)

        st.write('Number of images evaluated from each category')
        img_sum=val.groupby('real_label').count()        
        img_sum.plot(kind='bar')
        plt.xticks(rotation=10) #beware of cutoff
        plt.ylabel('Number of images')
        plt.title('Distribution of images')
        st.pyplot()

        
        return output        
    
    def cm_plot(predictions):
        #input predictions df to get confusion matrix
        #confusion matrix just needs max prob for each image
        
        #convert predictions outside of real_labels to 'other'
        predictions.loc[~predictions['pred_label'].isin(predictions['real_label']),'pred_label']='other'

        #find list of all cats predicted & real
        axis_cats= list(predictions.real_label.unique())
        axis_cats.append('other')
        

        
        cm=confusion_matrix(predictions['pred_label'], predictions['real_label'], labels=axis_cats)
        print(cm)

        df_cm= pd.DataFrame(cm,index=range(cm.shape[0]),columns= range(cm.shape[0]))
        print(df_cm)
        
        #ADD COLORING
        
        sns.heatmap(df_cm, vmin=0, xticklabels=axis_cats, yticklabels=axis_cats, annot=True)
        plt.title('Confusion Matrix')
        st.pyplot()
        return None
    
    def ROC(df, feature):
        #input predictions df + desired feature (chosen from dropdown) to get ROC and AUC

        #score= roc_auc_score(predictions)
        #testy: 0 or 1
        #second arg: model predicted probabilities
        nrow, _ =df.shape 

        testy= np.zeros(nrow,)
        hits=np.where(df['real_label']==feature)[0]
        
        testy[hits]= 1 
        
        lr_fpr, lr_tpr, _ = roc_curve(testy, df[feature])
        print(df.loc[hits,[feature, 'pred_label']])
        auc_score = roc_auc_score(testy, df[feature])


        plt.plot([0,1], [0,1], linestyle='--')
        plt.plot(lr_fpr, lr_tpr, marker='.', label=model_sel)
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.title(f'ROC for {feature}: \n AUC: {auc_score}')
        plt.legend()
        # show the plot
        st.pyplot()

        return None 


    #if complete datased eval is selected
    df= complete_eval(onlyfiles=onlyfiles)   
    cm_plot(df) #generate confusion matrix
    if feat_sel:
        ROC(df, feature=feat_sel)
    
