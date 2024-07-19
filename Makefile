install:
	pip install -r requirements.txt
	python setup.py install

prepare:
	python src/data_preparation.py	

model:
	python src/model.py

evaluate:
	python src/evaluation.py


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


test:
	pytest --disable-warnings 

format:
	black .

lint:
	flake8 .

pre-commit-install:
	pre-commit install

