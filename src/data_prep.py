from __future__ import annotations

import pandas as pd

from src.metrics import add_length_of_stay, aggregate_items, to_datetime_safe, to_numeric_safe
from src.config import BASE_01_PATH, BASE_02_PATH


def load_csv_or_xlsx(path):
    if str(path).lower().endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def prepare_dataset(df_base_01: pd.DataFrame, df_base_02: pd.DataFrame) -> pd.DataFrame:
    df1 = df_base_01.copy()
    df2 = df_base_02.copy()

    # tipos base 01
    for c in ["idade", "tempo_autorizacao_horas", "valor_total_conta", "valor_pago"]:
        if c in df1.columns:
            df1[c] = to_numeric_safe(df1[c])

    for c in [
        "data_nascimento",
        "data_solicitacao_autorizacao",
        "data_autorizacao_senha",
        "data_admissao",
        "data_alta",
    ]:
        if c in df1.columns:
            df1[c] = to_datetime_safe(df1[c])

    df1 = add_length_of_stay(df1)

    # agregações itens
    agg_itens = aggregate_items(df2)

    # merge final
    df = df1.merge(agg_itens, on="senha_internacao", how="left")

    # internações sem itens: zera agregados essenciais
    for c in ["itens_qtd", "itens_distintos", "valor_itens_total", "valor_glosado_total", "qtd_itens_glosados"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    for c in ["pct_itens_glosados", "pct_valor_glosado"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df


def load_and_prepare_from_disk() -> pd.DataFrame:
    df1 = load_csv_or_xlsx(BASE_01_PATH)
    df2 = load_csv_or_xlsx(BASE_02_PATH)
    return prepare_dataset(df1, df2)