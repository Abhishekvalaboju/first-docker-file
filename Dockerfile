FROM ubuntu:18.04
FROM python:3.6.0
WORKDIR /C:/pythonProject/Backend/dockertest
ADD . /C:/pythonProject/Backend/dockertest
RUN /bin/sh -c pip install requirements.txt
CMD ["python, FaceRecognition.py"]