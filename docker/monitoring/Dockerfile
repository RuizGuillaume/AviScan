FROM tensorflow/tensorflow:latest-gpu
WORKDIR /home/app
COPY requirements.txt .
RUN apt-get update
RUN apt-get install python3-pip -y
RUN apt-get install -y git
RUN apt-get install -y supervisor
RUN pip3 install -r requirements.txt
COPY monitoring.py .
COPY system_monitor.py .
COPY drift_monitor.py .
COPY alert_system.py .
COPY supervisord.conf .
RUN mkdir -p /home/app/volume_data/logs
CMD ["/usr/bin/supervisord"]