from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

# Se você já tem essas utils no projeto, ótimo (você usa no notebook)
from src.utils import pipeline_universal_limpeza, validacao_cid, trat_senha_internacao

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BASE1_PATH = RAW_DIR / "Base 1 – Internações Hospitalares.csv"
BASE2_PATH = RAW_DIR / "Base 2 – Itens da internação.csv"


def p95(s: pd.Series) -> float:
    s = s.dropna()
    return float(np.percentile(s, 95)) if len(s) else np.nan


def build_dim_tempo(df_internacao: pd.DataFrame) -> pd.DataFrame:
    # Pegue só as datas que você vai usar no painel
    cols = [
        "data_solicitacao_autorizacao",
        "data_autorizacao_senha",
        "data_admissao",
        "data_alta",
        "data_item",  # se existir no df de itens; aqui é só exemplo
    ]
    datas = []
    for c in cols:
        if c in df_internacao.columns:
            datas.append(df_internacao[c])

    if not datas:
        return pd.DataFrame(columns=["data"])

    all_dates = pd.concat(datas, ignore_index=True).dropna().drop_duplicates()
    all_dates = pd.to_datetime(all_dates, errors="coerce").dropna().drop_duplicates()

    dim = pd.DataFrame({"data": all_dates.sort_values().unique()})
    dim["ano"] = dim["data"].dt.year
    dim["mes"] = dim["data"].dt.month
    dim["ano_mes"] = dim["data"].dt.to_period("M").astype(str)
    dim["dia"] = dim["data"].dt.day
    dim["dia_semana"] = dim["data"].dt.day_name()
    return dim


def main() -> None:
    # -------------------------
    # 1) Load
    # -------------------------
    df_base_01 = pd.read_csv(BASE1_PATH)
    df_base_02 = pd.read_csv(BASE2_PATH)

    # -------------------------
    # 2) Limpeza / tipagem (seu pipeline)
    # -------------------------
    df_base_01 = pipeline_universal_limpeza(df_base_01)
    df_base_02 = pipeline_universal_limpeza(df_base_02)

    # -------------------------
    # 3) Features Base 01 (como no notebook)
    # -------------------------
    # Status
    df_base_01["status_internacao"] = "ENCERRADA"
    df_base_01.loc[df_base_01["data_alta"].isna(), "status_internacao"] = "ATIVA"

    # Dias internado (ATIVA usa hoje)
    df_base_01["dias_internado"] = (
        df_base_01["data_alta"].fillna(pd.Timestamp.today()) - df_base_01["data_admissao"]
    ).dt.days

    # Idade (aprox)
    adm = pd.to_datetime(df_base_01["data_admissao"], errors="coerce")
    nasc = pd.to_datetime(df_base_01["data_nascimento"], errors="coerce")
    idade_aprox = (adm - nasc).dt.days / 365.25
    df_base_01["idade"] = np.floor(pd.to_numeric(idade_aprox, errors="coerce")).astype("Int64")

    # Tempo autorização (horas)
    delta = df_base_01["data_autorizacao_senha"] - df_base_01["data_solicitacao_autorizacao"]
    df_base_01["tempo_autorizacao_horas"] = delta.dt.total_seconds() / 3600

    # Validações CID e senha
    df_base_01 = validacao_cid(df_base_01)
    df_base_01 = trat_senha_internacao(df_base_01)

    # -------------------------
    # 4) Features Base 02 + agregação por internação
    # -------------------------
    df_base_02 = trat_senha_internacao(df_base_02)

    # Numéricos
    for c in ["quantidade_solicitada", "quantidade_autorizada", "valor_unitario", "valor_total_item", "valor_glosado"]:
        if c in df_base_02.columns:
            df_base_02[c] = pd.to_numeric(df_base_02[c], errors="coerce")

    # Flags úteis
    df_base_02["item_glosado_flag"] = df_base_02["valor_glosado"].fillna(0) > 0
    df_base_02["delta_qtd"] = df_base_02["quantidade_solicitada"] - df_base_02["quantidade_autorizada"]
    df_base_02["delta_qtd_flag"] = df_base_02["delta_qtd"].fillna(0) > 0

    agg_itens = (
        df_base_02
        .groupby("senha_internacao", as_index=False)
        .agg(
            itens_qtd_total=("item_id", "count"),
            itens_tipos_distintos=("tipo_item", "nunique"),
            itens_subtipos_distintos=("subtipo_item", "nunique"),
            itens_codigos_distintos=("codigo_item", "nunique"),
            itens_valor_total=("valor_total_item", "sum"),
            itens_valor_medio=("valor_total_item", "mean"),
            itens_valor_max=("valor_total_item", "max"),
            itens_valor_p95=("valor_total_item", p95),
            glosa_item_qtd=("item_glosado_flag", "sum"),
            valor_glosado_total=("valor_glosado", "sum"),
            motivos_glosa_distintos=("motivo_glosa", "nunique"),
            delta_qtd_total=("delta_qtd", "sum"),
            delta_qtd_itens_qtd=("delta_qtd_flag", "sum"),
        )
    )

    # Percentuais
    agg_itens["glosa_item_pct"] = agg_itens["glosa_item_qtd"] / agg_itens["itens_qtd_total"].replace(0, np.nan)
    agg_itens["valor_glosado_pct"] = agg_itens["valor_glosado_total"] / agg_itens["itens_valor_total"].replace(0, np.nan)
    agg_itens["delta_qtd_pct_itens"] = agg_itens["delta_qtd_itens_qtd"] / agg_itens["itens_qtd_total"].replace(0, np.nan)

    # -------------------------
    # 5) Montagem das tabelas do PBI
    # -------------------------
    # Fatos
    fact_internacao = df_base_01.copy()
    fact_itens = df_base_02.copy()
    agg_itens_internacao = agg_itens.copy()

    # Dimensões (mínimas)
    dim_hospital = (
        df_base_01[["hospital_id", "hospital_nome", "perfil_hospital"]]
        .drop_duplicates()
        .sort_values("hospital_id")
        .reset_index(drop=True)
    )

    dim_beneficiario_cols = [
        "beneficiario_id", "numero_carteirinha", "nome_beneficiario",
        "data_nascimento", "sexo", "uf", "municipio"
    ]
    dim_beneficiario_cols = [c for c in dim_beneficiario_cols if c in df_base_01.columns]
    dim_beneficiario = (
        df_base_01[dim_beneficiario_cols]
        .drop_duplicates()
        .sort_values("beneficiario_id")
        .reset_index(drop=True)
    )

    dim_plano_cols = ["tipo_plano", "segmentacao_plano", "acomodacao"]
    dim_plano_cols = [c for c in dim_plano_cols if c in df_base_01.columns]
    dim_plano = (
        df_base_01[dim_plano_cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    dim_tempo = build_dim_tempo(df_base_01)

    # -------------------------
    # 6) Export (CSV para facilidade no PBI)
    # -------------------------
    fact_internacao.to_csv(PROCESSED_DIR / "fact_internacao.csv", index=False)
    fact_itens.to_csv(PROCESSED_DIR / "fact_itens.csv", index=False)
    agg_itens_internacao.to_csv(PROCESSED_DIR / "agg_itens_internacao.csv", index=False)

    dim_hospital.to_csv(PROCESSED_DIR / "dim_hospital.csv", index=False)
    dim_beneficiario.to_csv(PROCESSED_DIR / "dim_beneficiario.csv", index=False)
    dim_plano.to_csv(PROCESSED_DIR / "dim_plano.csv", index=False)
    dim_tempo.to_csv(PROCESSED_DIR / "dim_tempo.csv", index=False)

    print("OK: arquivos gerados em data/processed")


if __name__ == "__main__":
    main()