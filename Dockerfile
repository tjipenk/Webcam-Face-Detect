FROM hdgigante/python-opencv:4.7.0-ubuntu

WORKDIR /apps
COPY . /apps

RUN pip install -r requirements.txt
CMD ["python", "main.py"]