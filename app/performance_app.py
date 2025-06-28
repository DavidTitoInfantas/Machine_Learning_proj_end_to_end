"""Modules for deploying the performance app."""

import re
from pathlib import Path

import gradio as gr
import pandas as pd

from src.utils import load_object

# Path folder
pasta = Path("./artifacts")
# Regex for extract date and hour
padrao = re.compile(r"model_(\d{8})_(\d{4})\.pkl")
# List for dupla (data, hora, caminho)
modelos = []

for arquivo in pasta.iterdir():
    if arquivo.is_file():
        match = padrao.match(arquivo.name)
        if match:
            data = int(match.group(1))
            hora = int(match.group(2))
            modelos.append((data, hora, arquivo))

# Ordena por data e hora (ambos em ordem decrescente)
modelo_mais_recente = (
    max(modelos, key=lambda x: (x[0], x[1]))[2] if modelos else None
)

#
model = load_object(modelo_mais_recente)
preprocessor = load_object("artifacts/preprocessor.pkl")
# print(model)

if modelo_mais_recente:
    print(
        f"Modelo mais recente: {modelo_mais_recente}"
    )  # modelo_mais_recente[2].name
else:
    print("Nenhum modelo encontrado no formato esperado.")


def predict_perform_student(
    gender,
    race_ethnicity,
    parental_level_of_education,
    lunch,
    test_preparation_course,
    reading_score,
    writing_score,
):
    """Predict performance of students based on their features.

    Args:
        gender (str): Gender of the student
        race_ethnicity (str): Race ethnicity of the student
        parental_level_of_education (str): Parental level of education
        lunch (str): Type of lunch
        test_preparation_course (str): Test preparation course status
        reading_score (int): Reading score of the student
        writing_score (int): Writing score of the student
    Returns:
        str: Predicted performance label
    """
    data_dic = {
        "gender": [gender],
        "race_ethnicity": [race_ethnicity],
        "parental_level_of_education": [
            parental_level_of_education
        ],
        "lunch": [lunch],
        "test_preparation_course": [test_preparation_course],
        "reading_score": [reading_score],
        "writing_score": [writing_score],
    }

    data_df = pd.DataFrame(data_dic)

    data_scaled = preprocessor.transform(data_df)
    pred = model.predict(data_scaled)[0]
    label = f"Predicted Performance: {round(pred,2)}"
    return label


inputs = [
    gr.Radio(['female', 'male'], label="gender"),
    gr.Radio(
        ['group A', 'group B', 'group C', 'group D', 'group E'],
        label="race_ethnicity",
    ),
    gr.Radio(
        [
            "bachelor's degree",
            "some college",
            "master's degree",
            "associate's degree",
            "high school",
            "some high school",
        ],
        label="parental_level_of_education",
    ),
    gr.Radio(['standard', 'free/reduced'], label="lunch"),
    gr.Radio(['none', 'completed'], label="test_preparation_course"),
    gr.Slider(0, 100, step=1, label="reading_score"),
    gr.Slider(0, 100, step=1, label="writing_score"),
]

outputs = [gr.Label(num_top_classes=5)]

examples = [
    ["female", "group B", "bachelor's degree", "standard", "none", 72, 72],
    ["female", "group C", "some college", "standard", "completed", 69, 90],
    ["female", "group B", "master's degree", "standard", "none", 90, 95],
    ["male", "group A", "associate's degree", "free/reduced", "none", 47, 57],
]

title = "Student Performance Prediction"
description = (
    "Enter the details to predict student performance based on their features."
)
article = "This app is a part of the Beginner's Guide to CI/CD for Machine Learning. It teaches how to automate training, evaluation, and deployment of models to Hugging Face using GitHub Actions."

gr.Interface(
    fn=predict_perform_student,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()
