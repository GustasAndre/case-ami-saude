from __future__ import annotations

import numpy as np
import pandas as pd


def to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def to_flag01(s: pd.Series) -> pd.Series:
    """
    Converte representações comuns para 0/1 (Int64 com NA).
    Aceita: Sim/Não, S/N, True/False, 1/0 (strings).
    """
    s = s.astype("string").str.strip().str.upper()
    mapping = {
        "SIM": 1, "S": 1, "TRUE": 1, "1": 1,
        "NÃO": 0, "NAO": 0, "N": 0, "FALSE": 0, "0": 0,
    }
    return s.map(mapping).astype("Int64")


def aggregate_items(df_items: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a Base 02 (itens) para 1 linha por internação.

    Espera colunas:
    - senha_internacao, item_id, codigo_item, valor_total_item, valor_glosado, glosa_item_flag
    """
    df = df_items.copy()

    if "glosa_item_flag" in df.columns:
        df["glosa_item_flag_01"] = to_flag01(df["glosa_item_flag"])
    else:
        df["glosa_item_flag_01"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    for c in ["valor_total_item", "valor_glosado"]:
        if c in df.columns:
            df[c] = to_numeric_safe(df[c])

    agg = df.groupby("senha_internacao", as_index=False).agg(
        itens_qtd=("item_id", "count"),
        itens_distintos=("codigo_item", pd.Series.nunique),
        valor_itens_total=("valor_total_item", "sum"),
        valor_glosado_total=("valor_glosado", "sum"),
        qtd_itens_glosados=("glosa_item_flag_01", "sum"),
    )

    agg["pct_itens_glosados"] = agg["qtd_itens_glosados"] / agg["itens_qtd"]
    agg["pct_valor_glosado"] = agg["valor_glosado_total"] / agg["valor_itens_total"]

    agg.replace([np.inf, -np.inf], np.nan, inplace=True)
    agg["pct_itens_glosados"] = agg["pct_itens_glosados"].fillna(0)
    agg["pct_valor_glosado"] = agg["pct_valor_glosado"].fillna(0)

    return agg


def add_length_of_stay(df_internacoes: pd.DataFrame) -> pd.DataFrame:
    """
    Cria dias_internado com base em data_admissao e data_alta.
    """
    df = df_internacoes.copy()
    df["data_admissao"] = to_datetime_safe(df["data_admissao"])
    df["data_alta"] = to_datetime_safe(df["data_alta"])

    df["dias_internado"] = (df["data_alta"] - df["data_admissao"]).dt.days
    df.loc[df["dias_internado"] < 0, "dias_internado"] = np.nan  # saneamento

    return df