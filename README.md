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

Pull our docker image from DockerHub: 

`docker pull jtwang1027/streamlit_imagenet:multimodel`

Set the project id and time zone: 

`export PROJECT_ID=project-id` `export ZONE=us-central1`

Change the name of the image for later Container Registry: 

`docker tag 24a04cb15635 gcr.io/$PROJECT_ID/sl`

Upload the container image to a registry so that GKE can download and run it: 

Configure the Docker command-line tool to authenticate to Container Registry: 

`gcloud auth configure-docker`

Upload the image to your Container Registry: 

`docker push gcr.io/${PROJECT_ID}/sldocker push gcr.io/${PROJECT_ID}/sl`
    
Set the project ID and Compute Engine zone options for the gcloud tool: 

`gcloud config set project $PROJECT_ID` `gcloud config set compute/zone $ZONE`

Create a two-node cluster named imagenet-cluster: 

`gcloud container clusters create imagenet-cluster --num-nodes=2`

Deploy our application: 

`kubectl create deployment imagenet-web --image=gcr.io/${PROJECT_ID}/sl:latest`

Expose our application to traffic from the internet: 

`kubectl expose deployment imagenet-web --type=LoadBalancer --port 80 --target-port 8501`

Scale up our application:

`kubectl scale deployment imagenet-web --replicas=3`

Delete the Service: This deallocates the Cloud Load Balancer created for your Service:

`kubectl delete service imagenet-web`

Delete the cluster: This deletes the resources that make up the cluster, such as the compute instances, disks and network resources: 

`gcloud container clusters delete imagenet-cluster`



## Load testing using 
