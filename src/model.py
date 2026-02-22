from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from src.config import DATA_PROCESSED, MODELS_DIR, OUTPUT_DATASET_MODEL, OUTPUT_MODEL, OUTPUT_SCORE, TargetConfig
from src.data_prep import load_and_prepare_from_disk


def build_target_proxy(df: pd.DataFrame, cfg: TargetConfig) -> pd.DataFrame:
    """
    Target proxy (alto risco operacional):
    - glosa_flag == 1  OR
    - dias_internado > p90 OR
    - valor_total_conta > p90
    """
    out = df.copy()

    # robustez para glosa_flag (0/1 ou Sim/Não)
    glosa = out["glosa_flag"].astype("string").str.strip().str.upper()
    glosa01 = glosa.isin(["1", "SIM", "TRUE"])

    p90_dias = out["dias_internado"].quantile(cfg.p90_q)
    p90_custo = out["valor_total_conta"].quantile(cfg.p90_q)

    out["target_risco"] = (glosa01 | (out["dias_internado"] > p90_dias) | (out["valor_total_conta"] > p90_custo)).astype(int)
    return out


def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    # Features: sem leakage (não usar dias_internado/valor_total_conta no target composto)
    features_num = ["idade", "tempo_autorizacao_horas", "itens_qtd", "itens_distintos", "pct_itens_glosados"]

    features_cat = [
        "perfil_hospital",
        "tipo_plano",
        "segmentacao_plano",
        "acomodacao",
        "carater_internacao",
        "tipo_internacao",
        "especialidade_responsavel",
        "complexidade",
        "empresa_auditoria",
        "status_regulacao",
        "auditoria_responsavel",
        "uti_flag",
        "suporte_ventilatorio_flag",
        "hemodialise_flag",
    ]

    # manter apenas colunas existentes
    features_num = [c for c in features_num if c in df.columns]
    features_cat = [c for c in features_cat if c in df.columns]
    features = features_num + features_cat

    X = df[features].copy()
    y = df["target_risco"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, features_num),
            ("cat", cat_pipe, features_cat),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(max_iter=3000, class_weight="balanced")

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", clf),
    ])

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "auc": float(roc_auc_score(y_test, y_prob)),
        "classification_report": classification_report(y_test, y_pred, output_dict=False),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_cols_num": features_num,
        "feature_cols_cat": features_cat,
    }

    return model, metrics


def build_score(df: pd.DataFrame, model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    prob = model.predict_proba(X)[:, 1]

    out = df[["senha_internacao"]].copy()
    out["prob_risco"] = prob
    out["score_prioridade"] = (out["prob_risco"] * 100).round(1)

    out["classe_risco"] = pd.cut(
        out["score_prioridade"],
        bins=[0, 40, 70, 100],
        labels=["Baixo", "Médio", "Alto"],
        include_lowest=True
    )
    return out


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_from_disk()

    # target
    df = build_target_proxy(df, TargetConfig())

    # salva dataset para auditoria/reprodutibilidade
    df.to_csv(OUTPUT_DATASET_MODEL, index=False)

    # treino
    model, m = train_model(df)

    print("AUC:", round(m["auc"], 4))
    print("\nClassification report:\n", m["classification_report"])
    print("\nConfusion matrix:\n", m["confusion_matrix"])

    # salvar modelo
    joblib.dump(model, OUTPUT_MODEL)

    # score para PBI
    feature_cols = m["feature_cols_num"] + m["feature_cols_cat"]
    score = build_score(df, model, feature_cols)
    score.to_csv(OUTPUT_SCORE, index=False)

    print("\nArquivos gerados:")
    print("-", OUTPUT_DATASET_MODEL)
    print("-", OUTPUT_SCORE)
    print("-", OUTPUT_MODEL)


if __name__ == "__main__":
    main()