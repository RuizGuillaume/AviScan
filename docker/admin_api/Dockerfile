FROM ubuntu:20.04
COPY requirements.txt .
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt
WORKDIR /home/app
COPY admin_api.py .
COPY authorized_users.json .
COPY alert_system.py .
EXPOSE 5100
CMD ["uvicorn", "admin_api:app", "--host", "0.0.0.0", "--port", "5100"]