# VMK thesis

Применение методов машинного обучения для оценки финансового состояния компаний

## Installation

```bash
# create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate

# install dependencies
# pip install -r requirements.txt

# or install project in editable mode
pip install -e .
```

## Usage

```bash
docker compose -f docker/docker-compose.yml up

# option 1 
#http://127.0.0.1:8888/lab

# option 2
#mlflow server
#--backend-store-uri sqlite:////mlruns/mlflow.db
#--default-artifact-root mlflow-artifacts:/
#--host 0.0.0.0
#--port 5000
#python3 -m src/
train --model base;
train --model linear;
train --model main;
train --model selected;
echo "All done! You can find results "
echo "http://localhost:5005/#/experiments"
```