"""Modules for deploying the performance app."""

# import gradio as gr
# import skops.io as sio

# pipe = sio.load("./Model/drug_pipeline.skops", trusted=True)

import re
from pathlib import Path

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
print(model)

if modelo_mais_recente:
    print(
        f"Modelo mais recente: {modelo_mais_recente}"
    )  # modelo_mais_recente[2].name
else:
    print("Nenhum modelo encontrado no formato esperado.")


# def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
#     """Predict drugs based on patient features.
#
#     Args:
#         age (int): Age of patient
#         sex (str): Sex of patient
#         blood_pressure (str): Blood pressure level
#         cholesterol (str): Cholesterol level
#         na_to_k_ratio (float): Ratio of sodium to potassium in blood
#
#     Returns:
#         str: Predicted drug label
#     """
#     features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
#     predicted_drug = pipe.predict([features])[0]
#
#     label = f"Predicted Drug: {predicted_drug}"
#     return label
#
#
# inputs = [
#     gr.Slider(15, 74, step=1, label="Age"),
#     gr.Radio(["M", "F"], label="Sex"),
#     gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
#     gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
#     gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
# ]
# outputs = [gr.Label(num_top_classes=5)]
#
# examples = [
#     [30, "M", "HIGH", "NORMAL", 15.4],
#     [35, "F", "LOW", "NORMAL", 8],
#     [50, "M", "HIGH", "HIGH", 34],
# ]
#
#
# title = "Drug Classification"
# description = "Enter the details to correctly identify Drug type?"
# article = "This app is a part of the Beginner's Guide to CI/CD for Machine Learning. It teaches how to automate training, evaluation, and deployment of models to Hugging Face using GitHub Actions."
#
#
# gr.Interface(
#     fn=predict_drug,
#     inputs=inputs,
#     outputs=outputs,
#     examples=examples,
#     title=title,
#     description=description,
#     article=article,
#     theme=gr.themes.Soft(),
# ).launch()
#
