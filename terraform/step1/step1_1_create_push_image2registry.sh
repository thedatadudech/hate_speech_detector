
#Login to container registry

docker pull abdullahmarkus/mlopszoomcamp_project:mage_initial
docker pull abdullahmarkus/mlopszoomcamp_project:mlflow_initial

az acr login --name hatespeechctn

#tagging and pushing mage image
docker tag abdullahmarkus/mlopszoomcamp_project:mage_initial hatespeechctn.azurecr.io/mage_initial:latest
docker push hatespeechctn.azurecr.io/mage_initial:latest


#tagging and pushing mlflow image
docker tag abdullahmarkus/mlopszoomcamp_project:mlflow_initial hatespeechctn.azurecr.io/mlflow_initial:latest
docker push hatespeechctn.azurecr.io/mlflow_initial:latest

