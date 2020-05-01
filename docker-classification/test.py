

import pandas as pd
import numpy as np

from os import listdir, rename
from os.path import isfile, join


'''
file_read = open("imagenet_class_index.json").read()
categ = json.loads(file_read)
categ=pd.DataFrame(categ).T
categ=categ.set_index(0).to_dict()[1]#dict- code: name

#read in validation data
val=pd.read_csv('val_annotations.txt', sep='\t', names=['filename','class_id'], usecols=[0,1])
val['real_label']=val.apply(lambda row: categ[row['class_id']], axis=1)
'''

cat= pd.read_csv('categories.csv')

#every category has 50 images in the dataset
cat.groupby('real_label').count()

#pick 10 categories (500 images)

chosen= ['bucket',
        'sports_car',
        'desk',
        'sewing_machine',
        'jellyfish',
        'pay-phone', 
        'goldfish',
        'plate',
        'bathtub',
        'teddy'
        ]

files=cat.loc[cat['real_label'].isin(chosen),'filename']

old_path= [join('images' ,f) for f in files]
new_path= [join('sample_img' ,f) for f in files]

#rename(old_path, new_path)

for old, new in zip(old_path, new_path):
    rename(old, new) #relies on relative path
