install:
	pip install -r requirements.txt
	python setup.py install

prepare:
	python src/cli_version/data_preparation.py	

model:
	python src/cli_version/model.py

evaluate:
	python src/cli_version/evaluation.py


.PHONY: predict

TEXT ?=We have to put Trump in the bullseye
predict:
	@echo "Predicting with text: $(TEXT)"
	python src/predict.py --text "$(TEXT)"


all:
	make prepare
	make model
	make evaluate
	make predict


#Dockerize images
mage:
	scripts/start.sh mage

mlflow:
	scripts/start.sh mage

mage-build:
	scripts/start.sh mage --build 

mlflow-build:
	scripts/start.sh mlflow --build

flask-build:
	scripts/start.sh flask --build

gradio-build:
	scripts/start.sh gradio --build

compose-all-build:
	scripts/start.sh --build



#Cleaning section
complete-cleaning:
	scripts/cleaning_resources.sh



#Tools section
test:
	pytest --disable-warnings 

format:
	black .

lint:
	flake8 .

pre-commit-install:
	pre-commit install

