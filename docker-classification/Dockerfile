FROM pytorch/pytorch

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY . /app


# Install packages from requirements.txt
# hadolint ignore=DL3013
RUN pip install --upgrade pip &&\
    pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit","run"]

CMD ["app.py"]


