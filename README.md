# HATE_SPEECH_DETECTOR

## Overview

The Hate Speech Detector is designed to identify and classify hate speech across social media and other digital platforms. Utilizing cutting-edge machine learning techniques, this tool aims to moderate content and foster a healthier online environment.

## Problem Statement

The rise of digital platforms has led to an increase in hate speech, negatively impacting individuals and communities by promoting violence and discrimination. The Hate Speech Detector seeks to accurately identify such content, enabling actions for its removal or marking, thus promoting inclusive and respectful online communication.

## Technology Stack

- **MLflow**: Manages experiments, model versioning, and deployment, streamlining the process of optimizing machine learning models.
- **Gradio**: Provides an easy-to-use library for creating web apps for machine learning models, allowing users to interact with the Hate Speech Detector in real-time.
- **Evidently**: Monitors model performance and data quality, offering insights into how the model performs over time and identifying potential issues early.
- **Flask**: Serves as the backend, using this micro web framework for Python to integrate with other components and provide a RESTful API.
- **Mage**: Orchestrates workflows, automating the machine learning lifecycle from data preparation to training and deployment, facilitating team collaboration and development process efficiency.

## Architecture

The Hate Speech Detector's architecture is modular, ensuring flexibility and scalability. Key components include the data processing module, machine learning model, web interface, and monitoring system, interconnected through Flask and Mage for a seamless workflow from data input to output.

## Use Cases

- Moderating content on social media platforms
- Monitoring comments on news websites and blogs
- Assisting organizations in adhering to online communication policies

## Example Code

### Data Preparation

```python
import pandas as pd
from data_preparation import preprocess_data

# Sample data
data = {"tweet": ["This is a tweet", "Another tweet"], "class": [0, 1]}
df = pd.DataFrame(data)

# Preprocess the data
processed_df = preprocess_data(df)
```

### Model Training with MLflow

```python
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(processed_df['tweet'], processed_df['class'], test_size=0.2)
model = RandomForestClassifier()

with mlflow.start_run():
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "hate_speech_detector")
```

### Creating a Web Interface with Gradio

```python
import gradio as gr

def predict(tweet):
    prediction = model.predict([tweet])
    return prediction[0]

iface = gr.Interface(fn=predict, inputs="text", outputs="label")
iface.launch()
```

### Monitoring with Evidently

```python
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

dashboard = Dashboard(tabs=[DataDriftTab()])
dashboard.calculate(processed_df, reference_data, column_mapping=None)
dashboard.show()
```

### Setting up the Flask Backend

```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tweet = data['tweet']
    prediction = model.predict([tweet])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
```

This comprehensive guide, including example code, outlines the integration of various technologies into the Hate Speech Detector project, from data preparation to deployment and monitoring.
