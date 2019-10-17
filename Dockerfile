FROM tensorflow/tensorflow:latest-py3

RUN pip3 install flask pillow

WORKDIR /work/
COPY static/* ./static/
COPY trainer.py .
COPY server.py .

RUN python3 trainer.py
CMD python3 server.py
