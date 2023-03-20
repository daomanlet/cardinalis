FROM tensorflow/tensorflow:2.11.0rc2
COPY ./app /app
WORKDIR /app/main
RUN cd /app/main
RUN python --version
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install pipenv
RUN pipenv requirements > requirements.txt
RUN pip install -r requirements.txt 
#RUN pip install gdown
RUN mkdir ./deploy
#RUN gdown https://drive.google.com/file/d/1OuqJ7KCl5HMSnrVFmnt7l6gWpfPsYrUr -O ./deploy/mobilenet_v2.h5
COPY ./mobilenet_v2.h5 ./deploy/mobilenet_v2.h5
CMD ["/app/main/start_daemon.sh"]
