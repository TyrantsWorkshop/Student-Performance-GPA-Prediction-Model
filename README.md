# Student Performance GPA Prediction Model

## Student GPA and Dropout Risk Predictor

This project is a Flask web app that predicts:

- student dropout risk (`Dropout: Yes` or `Dropout: No`)
- numeric GPA on a 0.0 to 4.0 scale
- burnout status based on predicted GPA

The app uses supervised machine learning with two models:

- a classification model for dropout prediction
- a regression model for GPA prediction

The input features intentionally exclude `GPA`, `CGPA`, and `Semester_GPA` to avoid target leakage.

## Run Locally

```powershell
pip install -r requirements.txt
python student_gpa_dropout_app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Deploy on Render

Use these settings:

```text
Build Command:
pip install -r requirements.txt

Start Command:
gunicorn student_gpa_dropout_app:app
```

Make sure `student_dropout_dataset_v3.csv` is included in the repository.
