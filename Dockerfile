FROM python:3-onbuild

COPY apps/two-eat.py /apps/
COPY apps/config.py /apps/
COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /apps
CMD ["python3", "two-eat.py"]
