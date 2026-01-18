# ================================
# CREDIT DEFAULT MODEL TRAINING
# ================================

# ----------- Imports ------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import joblib

# ----------- Load Dataset ------------
df = pd.read_csv("default of credit card clients(Data).csv")

# ----------- Rename Target Column ------------
df.rename(columns={'default payment next month': 'default'}, inplace=True)

# ----------- Drop ID Column ------------
df.drop(columns=['ID'], inplace=True)

# ----------- Fix Categorical Columns ------------
df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)   # Others
df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)            # Others

# ----------- Separate Features & Target ------------
X = df.drop('default', axis=1)
y = df['default']

# ----------- Train-Test Split ------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ----------- Define Column Groups ------------
amount_cols = [col for col in X.columns if 'AMT' in col or col == 'LIMIT_BAL']
other_cols = [col for col in X.columns if col not in amount_cols]

# ----------- Preprocessing Pipeline ------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), amount_cols),
        ("pass", "passthrough", other_cols)
    ]
)

# ----------- Full ML Pipeline ------------
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ))
])

# ----------- Train Model ------------
pipeline.fit(X_train, y_train)

# ----------- Save Pipeline ------------
joblib.dump(pipeline, "credit_default_pipeline.pkl")

print("✅ Pipeline saved successfully as credit_default_pipeline.pkl")
