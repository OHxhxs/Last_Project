FROM python:3.10.6-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx



COPY . /code/last

WORKDIR /code/last

CMD ["python", "server/app.py"]
CMD ["last", "last.app.main:app", "--host", "0.0.0.0", "--port", "80"]