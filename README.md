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