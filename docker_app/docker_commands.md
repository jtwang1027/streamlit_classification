hey,

I just copied the self-driving demo and put it in this container. I also have this skeleton app.py


change so weights are pre-downloaded with docker app


```
docker run -p 5031:5031 jtwang1027/streamlit_app

sudo docker pull jtwang1027/streamlit_app

docker push jtwang1027/streamlit_app

sudo docker build -t jtwang1027/streamlit_app .
```
