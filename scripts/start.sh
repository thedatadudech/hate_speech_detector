
#Starter script file to load environmental variables, please fill out the TODO: with your credentials
 

docker rmi $(docker images -f "dangling=true" -q)

  
export PROJECT_NAME=hate_speech_detector 
export MAGE_CODE_PATH=/home/src 
export POSTGRES_HOST=<TODO:>
export POSTGRES_DB_MAGE=mage 
export POSTGRES_DB_MLFLOW=mlflow 
export POSTGRES_DBNAME=postgres 
export POSTGRES_PASSWORD=<TODO:>
export POSTGRES_USER=hatespeechadmin 
export POSTGRES_PORT=5432 
export MLFLOW_TRACKING_URI="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB_MLFLOW}"
export MAGE_DATABASE_CONNECTION_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB_MAGE}"
export DEFAULT_ARTIFACT_ROOT=/data/artifacts 
export SMTP_EMAIL=$SMTP_EMAIL=$SMTP_EMAIL  
export SMTP_PASSWORD=$SMTP_PASSWORD
#export VECTORIZER_PATH=/data/cv/hate_speech_detector/cv_best_model.pkl
#export BESTMODEL_PATH=/data/mlmodel/hate_speech_detector 
export BESTMODEL_PATH="https://hatespeechstorage.blob.core.windows.net/hatespeech-data/mlmodel/hate_speech_detector/best_model_fittedX.pkl?sp=r&st=2024-08-09T19:59:28Z&se=2024-08-23T03:59:28Z&spr=https&sv=2022-11-02&sr=b&sig=s091yRCR%2BgO25Z%2FMG2dLl2XbAsYX74Qslk6KHpHTlmA%3D"


echo "CHECK all Env variables"
echo "MAGE_CODE_PATH:" $MAGE_CODE_PATH
echo "POSTGRES_HOST:" $POSTGRES_HOST
echo "POSTGRES_DB_MAGE:" $POSTGRES_DB_MAGE
echo "POSTGRES_DB_MLFLOW:" $POSTGRES_DB_MLFLOW
echo "POSTGRES_DBNAME:" $POSTGRES_DBNAME
echo "POSTGRES_PASSWORD:" $POSTGRES_PASSWORD
echo "POSTGRES_USER:" $POSTGRES_USER
echo "POSTGRES_PORT:" $POSTGRES_PORT
echo "MLFLOW_TRACKING_URI:" $MLFLOW_TRACKING_URI
echo "MAGE_DATABASE_CONNECTION_URL:" $MAGE_DATABASE_CONNECTION_URL
echo "DEFAULT_ARTIFACT_ROOT:" $DEFAULT_ARTIFACT_ROOT
echo "SMTP_EMAIL:" $SMTP_EMAIL
echo "SMTP_PASSWORD:" $SMTP_PASSWORD
echo "VECTORIZER_PATH:" $VECTORIZER_PATH
echo "BESTMODEL_PATH:" $BESTMODEL_PATH


if [ "$1" == "--all" ]; then
  if [ "$2" == "--build" ]; then    
      docker compose build --no-cache
  fi
    docker compose up
else
  if [ "$2" == "--build" ]; then    
      docker compose build "$1" --no-cache
  fi
  docker compose up "$1"
fi

