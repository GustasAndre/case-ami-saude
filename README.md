# Case A.M.I Saúde — Painel Gerencial Analítico + Score de Risco (GIH)

Este repositório contém a solução completa do case técnico para Cientista de Dados Júnior, incluindo:
- **Painel Gerencial (Power BI)**: `dashboard/painel.pbix`
- **Análise Exploratória (PDF)**: `reports/analise_exploratoria.pdf`
- **Síntese Executiva (PDF)**: `reports/sintese_executiva.pdf`
- **Pipeline de dados + Modelagem** (Python): `src/` e `notebooks/`

## 1. Contexto e Objetivo
A partir de duas bases (internações e itens), o objetivo é apoiar a tomada de decisão na Gestão de Internações Hospitalares (GIH) por meio de:
- indicadores assistenciais, regulatórios e financeiros
- detecção de inconsistências/anomalias
- **score de priorização** para destacar internações com maior risco operacional

## 2. Estrutura do projeto
- `data/raw/`: arquivos de entrada (não versionados)
- `data/processed/`: saídas processadas e tabelas finais (inclui score para Power BI)
- `src/`: código reutilizável (preparo, métricas, modelagem)
- `notebooks/`: notebooks para narrativa (EDA e modelagem)
- `dashboard/`: PBIX
- `reports/`: PDFs finais (EDA + síntese)

## 3. Reprodutibilidade (ambiente)
### Opção A — Conda (recomendada)
```bash
conda env create -f environment.yml
conda activate case-ami-saude
```
### Opção B — pip/venv
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```
pip install -r requirements.txt

## 4. Como Rodar o pipeline:
### 4.1 Colocar dados em data/raw/
    * Base 01: Base 1 – Internações Hospitalares.csv (ou .xlsx)
    * Base 02: Base 2 – Itens da Internação.csv (ou .xlsx)
### 4.2 Gerar Dataset final e score:
```bash
python -m src.model
```
### Saídas Geradas:
    * data/processed/dataset_modelagem.csv (1 linha por internação)
    * data/processed/score_risco_internacoes.csv (para importar no Power BI)
    * models/model_risco_internacao.joblib (modelo treinado)

## 5. Integração com o Power Bi
No Power BI, importe:
    * data/processed/score_risco_internacoes.csv
Relacione com a tabela de internações por:
    * senha_internacao
use:
    * score_prioridade (0 a 100)
    * classe_risco (Baixo/Médio/Alto)

## 6. Modelagem
Foi treinado um modelo de Regressão Logística (classificação binária) para estimar risco operacional alto.
O target foi construído como proxy:
    * glosa presente ou
    * permanência acima do p90 ou
    * custo total acima do p90
A saída do modelo é convertida em score de 0–100 e exportada para uso no painel.

## 7 Observações:
    * Dados não são versionados no repositório
    * O modelo é suporte à decisão e não substitiu avaliação clínica