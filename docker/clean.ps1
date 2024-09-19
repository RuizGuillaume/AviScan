docker container stop user_api
docker container stop admin_api
docker container stop inference
docker container stop preprocessing
docker container stop training
docker container stop monitoring
docker container stop mlflowui
docker container stop streamlit

docker container rm user_api
docker container rm admin_api
docker container rm inference
docker container rm preprocessing
docker container rm training
docker container rm monitoring
docker container rm mlflowui
docker container rm streamlit

#docker volume rm docker_main_volume

docker image prune -a -f

Write-Output "Clean up was successful "
