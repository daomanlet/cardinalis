FROM python:3.11-alpine
COPY ./app /app
WORKDIR /app
RUN cd /app
RUN pip install pipenv
RUN pipenv requirements > requirements.txt
RUN pip install -r requirements.txt
RUN pip install gdown
RUN mkdir ./deploy
RUN gdown https://drive.google.com/file/d/1OuqJ7KCl5HMSnrVFmnt7l6gWpfPsYrUr/view?usp=sharing -O ./deploy/mobilenet_v2.h5
