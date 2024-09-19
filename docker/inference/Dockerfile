FROM tensorflow/tensorflow:latest-gpu
COPY requirements.txt .
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt
WORKDIR /home/app
COPY inference.py .
COPY load_image.jpg .
COPY alert_system.py .
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "5500"]