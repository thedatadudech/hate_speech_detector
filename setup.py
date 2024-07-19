from setuptools import setup, find_packages


setup(
    name="hate_speech_detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "mlflow",
        "evidently",
        "nltk",
        "emoji",
        "numpy",
        "scikit-learn",
        "optuna",
        "gradio",
        "flask",
        "requests",
    ],
)
