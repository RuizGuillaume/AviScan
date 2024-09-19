FROM ubuntu:20.04
COPY requirements.txt .
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt
WORKDIR /home/app
COPY preprocessing.py .
COPY alert_system.py .
COPY SizeManager.py .
COPY UnderSampling.py .
COPY CleanDB.py .
COPY DatasetCorrection.py .
CMD ["uvicorn", "preprocessing:app", "--host", "0.0.0.0", "--port", "5500"]