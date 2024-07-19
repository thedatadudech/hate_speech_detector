#!/bin/bash


# Create data directories
mkdir -p data/raw
mkdir -p data/processed

# Create notebooks directory
mkdir -p notebooks

# Create models directory
mkdir -p models

# Create src directory and files
mkdir -p src
touch src/__init__.py
touch src/data_preparation.py
touch src/feature_engineering.py
touch src/model.py
touch src/evaluation.py
touch src/monitoring.py
touch src/config.py
touch src/train.py
touch src/predict.py
touch src/utils.py

# Create tests directory and files
mkdir -p tests
touch tests/__init__.py
touch tests/test_data_preparation.py
touch tests/test_feature_engineering.py
touch tests/test_model.py
touch tests/test_evaluation.py
touch tests/test_integration.py

# Create other project files
touch Dockerfile
touch mage_pipeline.py
touch requirements.txt
touch Makefile
touch pre-commit-config.yaml
touch setup.py
touch README.md

# Create GitHub Actions workflow directory and file
mkdir -p .github/workflows
touch .github/workflows/ci-cd.yml

echo "Project structure created successfully!"
