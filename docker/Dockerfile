#FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./svm_gamma=0.005_C=1.joblib /exp/svm_gamma=0.005_C=1.joblib
RUN pip3 install --no-cache-dir -r /exp/requirements.txt

COPY /models /exp/
COPY /results /exp/
WORKDIR /exp
#CMD ["python3", "./plot_graphs.py"]
COPY . .
#COPY app.py /api

#ENV FLASK_APP=app.py

# Unit tests
# RUN pip install pytest && pytest
# ARG x
# ARG y

EXPOSE 5000

ENV classi_name="svm"
ENV randomstate="24"
# CMD ["python3", "./plot_graphs.py","--x=${randomstate}","--y=${classi_name}"]
CMD ["python3", "./plot_graphs.py","--x=24","--y=svm"]

