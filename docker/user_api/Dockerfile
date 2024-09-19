FROM ubuntu:20.04
COPY requirements.txt .
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt
WORKDIR /home/app
COPY user_api.py .
EXPOSE 5000
CMD ["uvicorn", "user_api:app", "--host", "0.0.0.0", "--port", "5000"]