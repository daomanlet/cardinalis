FROM python:3.11-alpine
COPY ./app /app
WORKDIR /app
RUN sudo apt-get update
RUN sudo apt install apache2-dev
RUN cd /app
RUN pip install pipenv
RUN pipenv requirements > requirements.txt
RUN pip install -r requirements.txt
RUN pip install gdown
RUN mkdir ./deploy
RUN gdown https://drive.google.com/file/d/1OuqJ7KCl5HMSnrVFmnt7l6gWpfPsYrUr -O ./deploy/mobilenet_v2.h5
