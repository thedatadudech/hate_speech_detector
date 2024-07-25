import gradio as gr
import requests


# Define the function to send text to the MLflow model API and get the prediction
def predict_hate_speech(text):
    url = "http://flask_app:5002/predict"  # Adjust the URL if your API is hosted elsewhere
    # headers = {"Content-Type": "application/json"}
    data = {"text": text}
    print(data)
    response = requests.post(url, json=data)
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        return f"{prediction}"
    else:
        return f"Error: {response.text}"


# Create the Gradio interface
iface = gr.Interface(
    fn=predict_hate_speech,
    inputs=gr.Textbox(lines=5, label="Enter Text"),
    outputs="text",
    title="Hate Speech Detection",
    description="Enter text and click submit to classify it as hate speech or not.",
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
