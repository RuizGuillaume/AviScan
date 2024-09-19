FROM ubuntu:20.04
RUN apt-get update && apt-get install python3-pip dos2unix -y && pip3 install mlflow
WORKDIR /home/app/
COPY wait_for_mlruns.sh .
RUN dos2unix wait_for_mlruns.sh
RUN chmod +x wait_for_mlruns.sh
EXPOSE 5200
CMD ["./wait_for_mlruns.sh"]