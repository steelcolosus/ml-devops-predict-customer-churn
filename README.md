# Predict Customer Churn

- This repository contains the Predict Customer Churn project, a part of the ML DevOps Engineer Nanodegree by Udacity.

## Project Description
The goal of this project is to develop a machine learning model to predict customer churn using the provided dataset, following coding best practices. The project uses Python, venv for virtual environments, and pytest for testing.

## Files and data description
- **churn_library.py**: Python library containing functions for data preprocessing, EDA, feature engineering, model training, and model evaluation.
- **churn_notebook.ipynb:** Jupyter Notebook containing the step-by-step implementation of the project.
- **churn_script_logging_and_tests.py:** Script to run logging and tests for the project.
- **data/bank_data.csv:** Dataset containing customer information and churn status.
- **Guide.ipynb:** Jupyter Notebook providing a guide for project implementation.
- **images/eda:** Folder containing EDA-related images.
- **images/results:** Folder containing model results and performance-related images.
- **logs/churn_library.log:** Log file for the project.
- **models:** Folder containing the saved machine learning models.
- **pytest.ini:** Configuration file for pytest, used to configure pytest logs. Edit this file if you want to modify the logging settings.
- **requirements.txt:** List of Python packages required to run the project.

### Repository structure

```
.
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── Guide.ipynb
├── images
│   ├── eda
│   │   ├── Churn.png
│   │   ├── Customer_Age.png
│   │   ├── Heatmap.png
│   │   ├── Marital_Status.png
│   │   └── Total_Trans_Ct.png
│   └── results
│       ├── classification_report_lr.png
│       ├── classification_report_rf.png
│       ├── feature_importance.png
│       └── roc_curve.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── __pycache__
│   ├── churn_library.cpython-310.pyc
│   ├── churn_library.cpython-38.pyc
│   └── churn_script_logging_and_tests.cpython-38-pytest-6.2.4.pyc
├── pytest.ini
├── README.md
└── requirements.txt

```

## Pytest.ini Configuration
The pytest.ini file is used to configure pytest logs. The following settings are available in the file:
```
[pytest]
log_cli = true
log_cli_level = INFO
log_cli_format = %(name)s - %(levelname)s - %(message)s
log_format = %(name)s - %(levelname)s - %(message)s

log_file = logs/churn_library.log
log_file_format = %(asctime)s - %(name)s - %(levelname)s - %(message)s
log_file_level = INFO

```
Edit this file if you want to modify the logging settings for pytest.

## Getting started

1. Clone the repository

```bash
git clone https://github.com/yourusername/predict-customer-churn.git
```

2. Create and activate a virtual environment using venv

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install packages

```bash
pip install -r requirements.txt
```

4. Run tests using pytest

```bash
pytest churn_script_logging_and_tests.py
```