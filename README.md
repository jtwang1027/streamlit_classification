# Streamlit App for visualizing image classification performance on Kubernetes

## Motivation
Imagenet is an image database for visual object recognition. It contains more than 14 million hand-annotated images. Various types of models have been developed to with great accuracy including: ResNeXt 152, Densenet 201, Darknet, AlexNet, VGG-16. Here, we have used a computationally light convolutional neural network--MobileNet. After training a model, it's important to be able to visualize the results. Summary metrics are helpfui, but being able to directly examine specific images or categories may help better understand the model--especially in the case on image data. 

## Streamlit App
The app was buit using [streamlit](https://docs.streamlit.io/), which is an open source library for building custom web apps. The ability to display data, charts, interactive widgets/sidebars, and other media makes it great for showcasing machine learning/data science.

The app was dockerized and can be pulled and run with the code below. 
```
docker run -p 8501:8501 jtwang1027/streamlit_imagenet 
```

## Kubernetes Deployment





## Load testing using 