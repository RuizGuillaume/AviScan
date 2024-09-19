FROM tensorflow/tensorflow:latest-gpu
COPY requirements.txt .
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt
WORKDIR /home/app
COPY mlruns/ ./mlruns/
COPY prod_model_id.txt .
COPY training.py .
COPY alert_system.py .
CMD ["uvicorn", "training:app", "--host", "0.0.0.0", "--port", "5500"]