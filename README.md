# Previsão de Nível de Burnout — MVP Sprint 2

**Pós-Graduação em Engenharia de Software — PUC-Rio**
Disciplina: Qualidade de Software, Segurança e Sistemas Inteligentes

## Sobre o Projeto

Este MVP propõe um sistema de previsão do nível de burnout de profissionais com base em dados demográficos, condições de trabalho e saúde física. O usuário preenche um formulário web e recebe uma classificação entre três níveis: **Baixo (1)**, **Médio (2)** ou **Alto (3)**, calculada a partir da média de quatro modelos de machine learning.

### Dataset

[Remote Work Health Impact Survey 2025](https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025) (Kaggle) — 3.157 registros com informações sobre idade, gênero, região, cargo, regime de trabalho, horas semanais, equilíbrio trabalho–vida, isolamento social, problemas de saúde e nível de burnout.

## Estrutura do Repositório

```
├── MVP2.ipynb            # Notebook com análise exploratória, pré-processamento e treinamento
├── backend/
│   ├── app.py            # API Flask (endpoints /predict e /options)
│   ├── test_models.py    # Testes automatizados (pytest) de qualidade dos modelos
│   └── requirements.txt  # Dependências Python
├── frontend/
│   ├── index.html        # Formulário de entrada
│   ├── style.css         # Estilos
│   └── script.js         # Lógica de requisição à API e exibição do resultado
├── models/               # Modelos treinados e scaler (.pkl)
└── test_sample/          # Dados de teste para validação dos modelos (.pkl)
```

## Modelos Utilizados

| Modelo | Abordagem |
|---|---|
| **KNeighborsClassifier** | Classificação por vizinhos mais próximos |
| **DecisionTreeClassifier** | Árvore de decisão |
| **CategoricalNB** | Naive Bayes para features categóricas |
| **SVC** | Máquina de vetores de suporte |

Todos os modelos são otimizados via **GridSearchCV** com **5-fold Cross-Validation** e métrica **F1 micro**. A saída final é a média das previsões dos 4 modelos, arredondada para o nível mais próximo (1, 2 ou 3).

## Testes Automatizados

Na inicialização do servidor Flask, testes pytest são executados automaticamente para garantir que cada modelo atinja um **F1 score micro > 0.35** nos dados de teste. Se algum modelo falhar, o servidor não inicia.

```bash
pytest backend/test_models.py -v
```

## Como Executar

### Pré-requisitos

- Python 3.10+

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt
python app.py
```

O servidor inicia em `http://localhost:5000` após a execução dos testes.

### Frontend

Abra o arquivo `frontend/index.html` diretamente no navegador. O formulário consome a API em `http://localhost:5000`.

## Endpoints da API

### `GET /options`

Retorna as opções válidas para os campos categóricos do formulário.

### `POST /predict`

Recebe os dados do profissional e retorna a previsão de burnout.

**Corpo da requisição:**

```json
{
  "idade": 30,
  "genero": "Male",
  "continente": "South America",
  "industria": "Technology",
  "cargo": "Software Engineer",
  "regime_trabalho": "Remoto",
  "faixa_salarial": "$60K-80K",
  "horas_trabalho_semana": 45,
  "equilibrio_trabalho_vida": 3,
  "isolamento_social": 4,
  "problemas_saude": ["Dor nas Costas", "Fadiga Ocular"]
}
```

**Resposta:**

```json
{
  "nivel_burnout": 2,
  "label": "Médio",
  "media_modelos": 2.25,
  "previsoes_individuais": {
    "KNeighborsClassifier": 2.0,
    "DecisionTreeClassifier": 3.0,
    "CategoricalNB": 2.0,
    "SVC": 2.0
  }
}
```

## Notebook (MVP2.ipynb)

O notebook documenta todo o pipeline de dados:

1. **Coleta** — Carregamento via Croissant (JSON-LD) do Kaggle
2. **Análise Exploratória** — Distribuição do target e relação com features categóricas e numéricas
3. **Pré-processamento** — MultiLabelBinarizer para problemas de saúde (múltipla escolha), One-Hot Encoding para demais categóricas, MinMaxScaler para normalização
4. **Treinamento** — GridSearchCV com 5-fold CV para os 4 classificadores
5. **Exportação** — Modelos, scaler e dados de teste salvos em `.pkl`