from __future__ import annotations

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Iterable, Literal, Optional



def pipeline_universal_limpeza(df):
    colunas_string = [
        'senha_internacao', 'beneficiario_id', 'numero_carteirinha', 
        'hospital_id', 'item_id', 'codigo_item'
    ]

    colunas_datas = [
        'data_nascimento', 'data_solicitacao_autorizacao', 'data_autorizacao_senha', 
        'data_admissao', 'data_alta', 'data_item'
    ]

    colunas_categoricas = [
        'sexo', 'uf', 'perfil_hospital', 'tipo_plano', 'segmentacao_plano', 
        'acomodacao','motivo_alta','carater_internacao', 'tipo_internacao','especialidade_responsavel',
        'complexidade', 'status_regulacao', 'uti_flag', 'suporte_ventilatorio_flag', 
        'hemodialise_flag','auditoria_responsavel','glosa_flag', 'tipo_item', 'subtipo_item', 
        'unidade_medida', 'setor_execucao', 'flag_pacote', 'glosa_item_flag'
    ]
    
    for col in df.columns:
        if col in colunas_datas:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        elif col in colunas_categoricas:
            df[col] = df[col].astype('category')
            
        elif col in colunas_string:
            df[col] = df[col].astype(str).str.strip()
            
        elif col in ['idade', 'quantidade_solicitada', 'quantidade_autorizada']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('int32')
            
        elif 'valor' in col or 'tempo' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    return df

def cid_formato_valido(cid):
    if pd.isna(cid):
        return False
    return bool(re.match(r"^[A-Z][0-9]{2}(\.[0-9])?$", cid))


def cid_compativel_especialidade(cid, especialidade, mapa):
    if pd.isna(cid):
        return True  # CID secundário ausente não é erro
    if pd.isna(especialidade):
        return True
    if especialidade not in mapa:
        return True
    return cid[0] in mapa[especialidade]





Method = Literal["iqr", "zscore"]


@dataclass(frozen=True)
class OutlierConfig:
    method: Method = "iqr"
    iqr_k: float = 1.5           # 1.5 (padrão), 3.0 (mais conservador)
    zscore_k: float = 3.0        # 3.0 (padrão)
    include_cols: Optional[list[str]] = None
    exclude_cols: Optional[list[str]] = None
    min_non_null: int = 10       # evita calcular outlier em colunas quase vazias


def find_outliers(
    df: pd.DataFrame,
    cfg: OutlierConfig = OutlierConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detecta outliers em colunas numéricas.

    Retorna:
      - report: resumo por coluna (limites, qtd outliers, %)
      - mask: dataframe booleano com mesma index do df e colunas numéricas avaliadas
              (True = é outlier naquela coluna)
    """
    if df.empty:
        report = pd.DataFrame(columns=[
            "col", "method", "n_non_null", "lower", "upper", "n_outliers", "pct_outliers"
        ])
        mask = pd.DataFrame(index=df.index)
        return report, mask

    # 1) Escolha de colunas numéricas
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if cfg.include_cols is not None:
        num_cols = [c for c in num_cols if c in cfg.include_cols]

    if cfg.exclude_cols is not None:
        num_cols = [c for c in num_cols if c not in cfg.exclude_cols]

    if not num_cols:
        report = pd.DataFrame(columns=[
            "col", "method", "n_non_null", "lower", "upper", "n_outliers", "pct_outliers"
        ])
        mask = pd.DataFrame(index=df.index)
        return report, mask

    # 2) Máscara por coluna
    mask = pd.DataFrame(False, index=df.index, columns=num_cols)
    rows = []

    for col in num_cols:
        s = df[col]
        s_non_null = s.dropna()
        n_non_null = int(s_non_null.shape[0])

        # pula colunas com poucos dados
        if n_non_null < cfg.min_non_null:
            rows.append({
                "col": col,
                "method": cfg.method,
                "n_non_null": n_non_null,
                "lower": np.nan,
                "upper": np.nan,
                "n_outliers": 0,
                "pct_outliers": 0.0,
            })
            continue

        if cfg.method == "iqr":
            q1 = s_non_null.quantile(0.25)
            q3 = s_non_null.quantile(0.75)
            iqr = q3 - q1

            # se iqr == 0, não tem dispersão -> sem outlier via iqr
            if iqr == 0 or pd.isna(iqr):
                lower, upper = np.nan, np.nan
                out = pd.Series(False, index=s.index)
            else:
                lower = q1 - cfg.iqr_k * iqr
                upper = q3 + cfg.iqr_k * iqr
                out = (s < lower) | (s > upper)

        elif cfg.method == "zscore":
            mu = s_non_null.mean()
            sigma = s_non_null.std(ddof=0)

            if sigma == 0 or pd.isna(sigma):
                lower, upper = np.nan, np.nan
                out = pd.Series(False, index=s.index)
            else:
                z = (s - mu) / sigma
                out = z.abs() > cfg.zscore_k
                # limites "equivalentes" só para report
                lower = mu - cfg.zscore_k * sigma
                upper = mu + cfg.zscore_k * sigma
        else:
            raise ValueError(f"method inválido: {cfg.method}")

        mask[col] = out.fillna(False)

        n_outliers = int(mask[col].sum())
        pct_outliers = float(n_outliers / max(len(s_non_null), 1))

        rows.append({
            "col": col,
            "method": cfg.method,
            "n_non_null": n_non_null,
            "lower": float(lower) if pd.notna(lower) else np.nan,
            "upper": float(upper) if pd.notna(upper) else np.nan,
            "n_outliers": n_outliers,
            "pct_outliers": pct_outliers,
        })

    report = pd.DataFrame(rows).sort_values(["n_outliers", "pct_outliers"], ascending=False)
    return report, mask


def filter_outlier_rows(
    df: pd.DataFrame,
    mask: pd.DataFrame,
    how: Literal["any", "all"] = "any",
) -> pd.DataFrame:
    """
    Filtra linhas do df que são outliers em pelo menos uma coluna (any)
    ou em todas as colunas avaliadas (all).
    """
    if mask.empty:
        return df.iloc[0:0].copy()

    row_mask = mask.any(axis=1) if how == "any" else mask.all(axis=1)
    return df.loc[row_mask].copy()

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_outlier_boxplots(
    df: pd.DataFrame,
    mask: pd.DataFrame,
    max_cols: int | None = None,
    ncols: int = 3,
    figsize_per_plot: tuple[float, float] = (5.5, 4.0),
    show_outlier_points: bool = True,
    return_outlier_values: bool = True,
) -> pd.DataFrame | None:
    """
    Plota boxplots das colunas que possuem outliers (baseado no `mask`)
    e (opcional) retorna uma tabela com os valores outliers.

    - df: dataframe original
    - mask: dataframe booleano (True = outlier naquela coluna)
    - max_cols: limita quantas colunas plota (útil se tiver muitas)
    - ncols: número de colunas no grid de gráficos
    - figsize_per_plot: tamanho por subplot (largura, altura)
    - show_outlier_points: sobrepõe pontos dos outliers
    - return_outlier_values: retorna dataframe longo com os outliers
    """

    if mask is None or mask.empty:
        print("mask está vazia — nada para plotar.")
        return None

    # 1) Colunas com pelo menos 1 outlier
    cols_with_outliers = mask.columns[mask.any(axis=0)].tolist()

    if not cols_with_outliers:
        print("Nenhuma coluna com outliers pelo mask.")
        return None

    if max_cols is not None:
        cols_with_outliers = cols_with_outliers[:max_cols]

    # 2) Grid de plots
    nplots = len(cols_with_outliers)
    nrows = math.ceil(nplots / ncols)
    fig_w = figsize_per_plot[0] * ncols
    fig_h = figsize_per_plot[1] * nrows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)  # facilita indexar mesmo com 1 linha

    # 3) Plotar cada coluna
    for i, col in enumerate(cols_with_outliers):
        ax = axes[i]
        s = df[col].dropna()

        # Boxplot sem “fliers” padrão (vamos mostrar os outliers manualmente)
        ax.boxplot(s.values, vert=True, showfliers=False)
        ax.set_title(col)
        ax.set_ylabel("valor")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        if show_outlier_points:
            out_idx = mask.index[mask[col].fillna(False)]
            out_vals = df.loc[out_idx, col].dropna()

            # jitter leve no eixo x (pra espalhar pontos)
            if len(out_vals) > 0:
                x = 1 + np.random.uniform(-0.06, 0.06, size=len(out_vals))
                ax.scatter(x, out_vals.values, s=18, alpha=0.85)

                # anota alguns valores extremos (top/bottom 3) pra “apresentação”
                # (evita poluir quando tem muitos)
                if len(out_vals) >= 1:
                    # pega até 3 menores e 3 maiores
                    extremes = pd.concat([out_vals.nsmallest(3), out_vals.nlargest(3)]).drop_duplicates()
                    for idx, val in extremes.items():
                        ax.annotate(
                            f"{val:.2f}" if pd.api.types.is_number(val) else str(val),
                            xy=(1, val),
                            xytext=(6, 0),
                            textcoords="offset points",
                            fontsize=8
                        )

    # 4) Limpar axes sobrando
    for j in range(nplots, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Boxplots das colunas com outliers (outliers sobrepostos)", fontsize=14)
    fig.tight_layout()

    # 5) Tabela de valores outliers (long format)
    if return_outlier_values:
        outlier_rows = []
        for col in cols_with_outliers:
            idxs = mask.index[mask[col].fillna(False)]
            if len(idxs) == 0:
                continue
            tmp = df.loc[idxs, [col]].copy()
            tmp = tmp.rename(columns={col: "valor"})
            tmp["coluna"] = col
            tmp["index"] = tmp.index
            outlier_rows.append(tmp[["coluna", "index", "valor"]])

        outlier_values = pd.concat(outlier_rows, ignore_index=True) if outlier_rows else pd.DataFrame(
            columns=["coluna", "index", "valor"]
        )
        return outlier_values

    return None


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def boxplot_iqr_todas_colunas(
    df: pd.DataFrame,
    mask: pd.DataFrame,
    ncols: int = 3,
    figsize_per_plot: tuple = (5, 6),
    annotate_extremes: bool = True,
):
    """
    Plota boxplots clássicos (Tukey) para TODAS as colunas
    que possuem outliers segundo o mask (IQR).

    - df: dataframe original
    - mask: dataframe booleano (True = outlier naquela coluna)
    """

    # 1️⃣ Seleciona colunas com pelo menos um outlier
    cols = mask.columns[mask.any(axis=0)].tolist()

    if not cols:
        print("Nenhuma coluna com outliers para plotar.")
        return

    # 2️⃣ Grid automático
    nplots = len(cols)
    nrows = math.ceil(nplots / ncols)

    fig_w = figsize_per_plot[0] * ncols
    fig_h = figsize_per_plot[1] * nrows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    # 3️⃣ Loop por coluna
    for i, col in enumerate(cols):
        ax = axes[i]
        s = df[col].dropna()

        # Quartis
        q1 = s.quantile(0.25)
        q2 = s.quantile(0.50)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Boxplot padrão (sem fliers)
        ax.boxplot(s.values, vert=True, showfliers=False)

        # Outliers reais (IQR)
        out_idx = mask.index[mask[col]]
        out_vals = df.loc[out_idx, col].dropna()

        if len(out_vals) > 0:
            x = np.ones(len(out_vals)) + np.random.uniform(-0.05, 0.05, len(out_vals))
            ax.scatter(x, out_vals, alpha=0.8)

        # Linhas de referência
        ax.axhline(q1, linestyle="--", linewidth=1)
        ax.axhline(q2, linestyle="-", linewidth=1)
        ax.axhline(q3, linestyle="--", linewidth=1)

        ax.axhline(lower, linestyle=":", linewidth=1)
        ax.axhline(upper, linestyle=":", linewidth=1)

        # Títulos
        ax.set_title(col)
        ax.set_ylabel("valor")
        ax.set_xticks([])

        # Anota extremos (opcional)
        if annotate_extremes and len(out_vals) > 0:
            extremes = pd.concat([
                out_vals.nsmallest(2),
                out_vals.nlargest(2)
            ]).drop_duplicates()

            for val in extremes:
                ax.annotate(
                    f"{val:.2f}",
                    xy=(1, val),
                    xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=8
                )

    # 4️⃣ Remove eixos vazios
    for j in range(nplots, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Boxplots clássicos (IQR) — colunas com outliers",
        fontsize=14
    )
    fig.tight_layout()
    plt.show()

def validacao_cid(
        df: pd.DataFrame,
        cid_principal_col: str = 'cid_principal',
        cid_secundario_col: str = "cid_secundario",
        especialidade_col: str = "especialidade_responsavel",
        internacao_tipo_col: str = "tipo_internacao",
        internacao_carater_col: str = "carater_internacao"
):
    df_cid = pd.read_csv(r'C:\Projetos\case-ami-saude\data\raw\CID-10-CATEGORIAS.CSV',sep=';', encoding='latin1')
    mapa_especialidade_cid = {
    "Clínica Médica": (
        "A","B","E","I","J","K","N","R"
    ),
    "Pediatria": (
        "A","B","E","J","P","Q","R","Z"
    ),
    "Oncologia": (
        "C","D"
    ),
    "Nefrologia": (
        "N"
    ),
    "Ginecologia/Obstetrícia": (
        "O","N"
    ),
    "Pneumologia": (
        "J"
    ),
    "Cardiologia": (
        "I"
    ),
    "Gastroenterologia": (
        "K"
    ),
    "Ortopedia": (
        "M","S","T"
    ),
    "Neurologia": (
        "G","I"
    ),
    "Infectologia": (
        "A","B"
    ),
}
    # Verifica se o CID está no formato correto (uma letra maiúscula seguida de dois dígitos, opcionalmente seguido por um ponto e mais um dígito)
    cid_pattern = re.compile(r"^[A-Z][0-9]{2}$")

    df["cid_principal_formato_ok"] = df[cid_principal_col].str.match(cid_pattern)
    df["cid_secundario_formato_ok"] = df[cid_secundario_col].str.match(cid_pattern)

    #Verifica se o cid está presente na tabela de CID-10
    cids_validos = set(df_cid['CAT'])

    df["cid_principal_valido"] = df[cid_principal_col].isin(cids_validos)
    df["cid_secundario_valido"] = (df[cid_secundario_col].isna() | df[cid_secundario_col].isin(cids_validos))

    #Verifica se o CID pertence a especialidade médica responsável pela internação, usando o mapa de especialidade para letras iniciais do CID
    df["cid_principal_compativel_especialidade"] = df.apply(
    lambda x: cid_compativel_especialidade(
        x[cid_principal_col],
        x[especialidade_col],
        mapa_especialidade_cid
    ),
    axis=1
)

    df["cid_secundario_compativel_especialidade"] = df.apply(
        lambda x: cid_compativel_especialidade(
            x[cid_secundario_col],
            x[especialidade_col],
            mapa_especialidade_cid
        ),
        axis=1
    )

    #flags inconsistencias de CID
    cidp = df[cid_principal_col].astype("string").str.strip().str.upper()
    cids = df[cid_secundario_col].astype("string").str.strip().str.upper()

    tipo = df[internacao_tipo_col].astype("string").str.strip()
    carater = df[internacao_carater_col].astype("string").str.strip()
    esp = df[especialidade_col].astype("string").str.strip()
    # Camada 1- 
    """
    Tipo de internação Obstétrica → CID principal deve ser capítulo O

    Tipo de internação Psiquiátrica → CID principal deve ser capítulo F 
    """

    df["alerta_swap_obstetrica"] = (
        (tipo == "Obstétrica")
        & ~cidp.str.startswith("O", na=False)
        & cids.str.startswith("O", na=False)
    )
    df["alerta_swap_psiquiatrica"] = (
        (tipo == "Psiquiátrica")
        & ~cidp.str.startswith("F", na=False)
        & cids.str.startswith("F", na=False)
    )

    # Camada 2 - CID secundário mais compatível com a especialidade do que o CID principal

    df["cidp_comp_esp"] = df.apply(
        lambda x: cid_compativel_especialidade(x[cid_principal_col], x[especialidade_col], mapa_especialidade_cid),
        axis=1
    )

    df["cids_comp_esp"] = df.apply(
        lambda x: cid_compativel_especialidade(x[cid_secundario_col], x[especialidade_col], mapa_especialidade_cid),
        axis=1
    )

    df["alerta_swap_especialidade"] = (~df["cidp_comp_esp"]) & (df["cids_comp_esp"])

    # Camada 3 - CIDs genéricos (capítulos R e Z) não deveriam ser principais se houver um CID específico presente
    """
    CIDs dos capítulos R e Z são frequentemente usados como diagnóstico provisório.
    Quando eles aparecem como principal e existe um CID secundário mais específico, isso é um padrão clássico de hierarquia invertida.
    """
    df["alerta_swap_generico"] = (
        cidp.str.startswith(("R","Z"), na=False)
        & cids.notna()
        & ~cids.str.startswith(("R","Z"), na=False)
    )

    # Camada 4 - Para internações eletivas, se o CID principal for de Ortopedia (capítulos S ou T), mas houver um CID secundário compatível com a especialidade, pode ser um swap de CID
    """
    Eu não usei caráter como regra dura, mas como priorização.
    Por exemplo, trauma como CID principal em internação eletiva é raro. Quando isso acontecia e o secundário era compatível com a especialidade, eu marcava como suspeito.
    """

    df["alerta_swap_carater"] = (
        (carater == "Eletiva")
        & cidp.str.startswith(("S","T"), na=False)
        & df["cids_comp_esp"]  # secundário faz mais sentido pro setor
    )

    swap_cols = [
    "alerta_swap_obstetrica",
    "alerta_swap_psiquiatrica",
    "alerta_swap_especialidade",
    "alerta_swap_generico",
    "alerta_swap_carater",
]

    df["alerta_swap"] = df[swap_cols].any(axis=1)
    df["score_swap"] = df[swap_cols].sum(axis=1)

    return df


def trat_senha_internacao(
        df: pd.DataFrame,
        senha_col: str = "senha_internacao"
        ):
    """
    Valida senha no formato:
    SI + ANO (4 dígitos) + 7 números
    Exemplo válido: SI20261234567
    """
    df[senha_col] = df[senha_col].astype(str).str.strip().str.upper()
    padrao = r"^SI\d{4}\d{7}$"
    df[f"{senha_col}_valida"] = df[senha_col].apply(lambda x: bool(re.match(padrao, x)))
    return df