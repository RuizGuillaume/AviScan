FROM ubuntu:20.04
COPY requirements.txt .
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt
WORKDIR /home/app/
COPY streamlit.py .
COPY oiseau_cover.jpg .
COPY python_logo.png .
COPY docker_logo.png .
COPY mlflow_logo.png .
COPY tensorflow_logo.png .
COPY github_logo.png .
COPY architecture.svg .
COPY application.svg .
COPY pipeline_ci_cd.svg .
COPY logo.png .
EXPOSE 5300
CMD ["streamlit", "run", "streamlit.py"]