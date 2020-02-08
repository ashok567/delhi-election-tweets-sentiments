FROM python:3-onbuild

COPY apps/ /apps/
COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /apps
CMD ["python3", "sentiment-analysis.py"]
