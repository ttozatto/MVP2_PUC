import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODEL_NAMES = [
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "CategoricalNB",
    "SVC",
]

FEATURE_COLUMNS = [
    "Age",
    "Hours_Per_Week",
    "Work_Life_Balance_Score",
    "Social_Isolation_Score",
    "Back Pain",
    "Eye Strain",
    "Neck Pain",
    "Shoulder Pain",
    "Wrist Pain",
    "Gender_Female",
    "Gender_Male",
    "Gender_Non-binary",
    "Gender_Prefer not to say",
    "Region_Africa",
    "Region_Asia",
    "Region_Europe",
    "Region_North America",
    "Region_Oceania",
    "Region_South America",
    "Industry_Customer Service",
    "Industry_Education",
    "Industry_Finance",
    "Industry_Healthcare",
    "Industry_Manufacturing",
    "Industry_Marketing",
    "Industry_Professional Services",
    "Industry_Retail",
    "Industry_Technology",
    "Job_Role_Account Manager",
    "Job_Role_Business Analyst",
    "Job_Role_Consultant",
    "Job_Role_Content Writer",
    "Job_Role_Customer Service Manager",
    "Job_Role_Data Analyst",
    "Job_Role_Data Scientist",
    "Job_Role_DevOps Engineer",
    "Job_Role_Digital Marketing Specialist",
    "Job_Role_Executive Assistant",
    "Job_Role_Financial Analyst",
    "Job_Role_HR Manager",
    "Job_Role_IT Support",
    "Job_Role_Marketing Specialist",
    "Job_Role_Operations Manager",
    "Job_Role_Product Manager",
    "Job_Role_Project Manager",
    "Job_Role_Quality Assurance",
    "Job_Role_Research Scientist",
    "Job_Role_Sales Representative",
    "Job_Role_Social Media Manager",
    "Job_Role_Software Engineer",
    "Job_Role_Technical Writer",
    "Job_Role_UX Designer",
    "Work_Arrangement_Hybrid",
    "Work_Arrangement_Onsite",
    "Work_Arrangement_Remote",
    "Salary_Range_$100K-120K",
    "Salary_Range_$120K+",
    "Salary_Range_$40K-60K",
    "Salary_Range_$60K-80K",
    "Salary_Range_$80K-100K",
]

GENDER_OPTIONS = ["Female", "Male", "Non-binary", "Prefer not to say"]
REGION_OPTIONS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
INDUSTRY_OPTIONS = [
    "Customer Service", "Education", "Finance", "Healthcare",
    "Manufacturing", "Marketing", "Professional Services", "Retail", "Technology",
]
JOB_ROLE_OPTIONS = [
    "Account Manager", "Business Analyst", "Consultant", "Content Writer",
    "Customer Service Manager", "Data Analyst", "Data Scientist", "DevOps Engineer",
    "Digital Marketing Specialist", "Executive Assistant", "Financial Analyst",
    "HR Manager", "IT Support", "Marketing Specialist", "Operations Manager",
    "Product Manager", "Project Manager", "Quality Assurance", "Research Scientist",
    "Sales Representative", "Social Media Manager", "Software Engineer",
    "Technical Writer", "UX Designer",
]
WORK_ARRANGEMENT_OPTIONS = ["Hybrid", "Onsite", "Remote"]
SALARY_RANGE_OPTIONS = ["$40K-60K", "$60K-80K", "$80K-100K", "$100K-120K", "$120K+"]
HEALTH_ISSUE_OPTIONS = ["Dor nas Costas", "Fadiga Ocular", "Dor no Pescoço", "Dor no Ombro", "Dor no Pulso"]
HEALTH_ISSUE_MAP = {
    "Dor nas Costas": "Back Pain",
    "Fadiga Ocular": "Eye Strain",
    "Dor no Pescoço": "Neck Pain",
    "Dor no Ombro": "Shoulder Pain",
    "Dor no Pulso": "Wrist Pain",
}

BURNOUT_LABELS = {1: "Baixo", 2: "Médio", 3: "Alto"}


def load_scaler():
    """Load the shared scaler from disk."""
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def load_models():
    """Load all ML models from disk."""
    loaded = {}
    for name in MODEL_NAMES:
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(model_path, "rb") as f:
            loaded[name] = pickle.load(f)
    return loaded


scaler = load_scaler()
models = load_models()

test_result = pytest.main(["-v", os.path.join(os.path.dirname(__file__), "test_models.py")])
if test_result != pytest.ExitCode.OK:
    sys.exit("Servidor abortado: testes dos modelos falharam.")


def build_feature_vector(data):
    """Build a DataFrame with the correct feature columns from the input data."""
    row = {col: 0 for col in FEATURE_COLUMNS}

    row["Age"] = int(data["idade"])
    row["Hours_Per_Week"] = int(data["horas_trabalho_semana"])
    row["Work_Life_Balance_Score"] = int(data["equilibrio_trabalho_vida"])
    row["Social_Isolation_Score"] = int(data["isolamento_social"])

    gender = data["genero"]
    col = f"Gender_{gender}"
    if col in row:
        row[col] = 1

    region = data["continente"]
    col = f"Region_{region}"
    if col in row:
        row[col] = 1

    industry = data["industria"]
    col = f"Industry_{industry}"
    if col in row:
        row[col] = 1

    job_role = data["cargo"]
    col = f"Job_Role_{job_role}"
    if col in row:
        row[col] = 1

    arrangement = data["regime_trabalho"]
    arrangement_map = {"Presencial": "Onsite", "Híbrido": "Hybrid", "Remoto": "Remote"}
    arrangement_en = arrangement_map.get(arrangement, arrangement)
    col = f"Work_Arrangement_{arrangement_en}"
    if col in row:
        row[col] = 1

    salary = data["faixa_salarial"]
    col = f"Salary_Range_{salary}"
    if col in row:
        row[col] = 1

    health_issues = data.get("problemas_saude", [])
    for issue in health_issues:
        issue_en = HEALTH_ISSUE_MAP.get(issue, issue)
        if issue_en in row:
            row[issue_en] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_burnout(features_df):
    """Run prediction through all 4 models and return the averaged burnout level."""
    scaled = scaler.transform(features_df)
    predictions = []
    for name, model in models.items():
        pred = model.predict(scaled)[0]
        predictions.append(float(pred))

    avg = np.mean(predictions)
    level = int(np.round(avg))
    level = max(1, min(3, level))
    return level, avg, predictions


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Corpo da requisição vazio ou JSON inválido"}), 400

    required = [
        "idade", "continente", "industria", "genero",
        "horas_trabalho_semana", "equilibrio_trabalho_vida",
        "isolamento_social", "cargo", "regime_trabalho", "faixa_salarial",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Campos obrigatórios ausentes: {missing}"}), 400

    features_df = build_feature_vector(data)
    level, avg, individual = predict_burnout(features_df)

    return jsonify({
        "nivel_burnout": level,
        "label": BURNOUT_LABELS[level],
        "media_modelos": round(avg, 4),
        "previsoes_individuais": {
            name: pred for name, pred in zip(MODEL_NAMES, individual)
        },
    })


@app.route("/options", methods=["GET"])
def options():
    """Return the valid options for each categorical field."""
    return jsonify({
        "genero": GENDER_OPTIONS,
        "continente": REGION_OPTIONS,
        "industria": INDUSTRY_OPTIONS,
        "cargo": JOB_ROLE_OPTIONS,
        "regime_trabalho": ["Presencial", "Híbrido", "Remoto"],
        "faixa_salarial": SALARY_RANGE_OPTIONS,
        "problemas_saude": HEALTH_ISSUE_OPTIONS,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
