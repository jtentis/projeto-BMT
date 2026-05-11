# Projeto BMT

Auditoria automatica de trabalhos academicos: verificacao de alinhamento entre promessas e entregas com uso do DoCO.

## Baseline reproduzivel

Este repositorio contem um baseline simples, offline e reproduzivel para:

- ler artigos academicos em PDF;
- extrair e normalizar texto;
- identificar trechos de introducao, resultados e conclusao;
- comparar promessas e entregas com `TF-IDF + similaridade do cosseno`;
- exportar artefatos de dataset, protocolo experimental, metricas e resultados.

O objetivo desta versao e consolidar o dataset final da entrega, reduzir ambiguidades do desenho experimental e manter uma linha de comparacao estavel antes de evoluir para modelos mais complexos.

## Estrutura principal

```text
.
|-- data/
|   |-- raw/
|   `-- pdfs/
|-- notebooks/
|-- reports/
|   |-- baseline/
|   `-- 02_baseline.tex
|-- src/
|-- requirements.txt
`-- README.md
```

Pastas e arquivos mais relevantes:

- `data/raw/`: corpus final consolidado com 10 PDFs de entrada;
- `data/pdfs/`: copia auxiliar dos PDFs candidatos usados na consolidacao;
- `src/run_baseline.py`: comando principal do baseline;
- `reports/baseline/`: saidas geradas pela execucao;
- `reports/baseline/dataset/summary.json`: descricao e estatisticas do dataset final;
- `reports/baseline/protocol/experimental_protocol.json`: protocolo experimental em formato estruturado;
- `reports/baseline/protocol/experimental_protocol.md`: protocolo experimental legivel;
- `reports/baseline/metrics/formal_metrics.md`: definicao das metricas formais.

## Ambiente

- Python 3.13
- Dependencias em `requirements.txt`

## Instalacao

```bash
python -m venv .venv
```

No Windows:

```bash
.venv\Scripts\activate
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

Instale as dependencias:

```bash
pip install -r requirements.txt
```

## Execucao

Comando oficial do baseline:

```bash
python -m src.run_baseline --input_dir data/raw --output_dir reports/baseline
```

## Dataset consolidado

O corpus final desta entrega contem 10 trabalhos academicos em PDF, coletados do Pantheon/UFRJ e versionados em `data/raw/`.

Documentos:

- `AMDavid.pdf`
- `ENPinho.pdf`
- `FMRolim.pdf`
- `GGSouza.pdf`
- `HBCReis.pdf`
- `HBMHenriques.pdf`
- `LCMarques.pdf`
- `LMFGaleno.pdf`
- `MDFonseca.pdf`
- `PVMNascimento.pdf`

Estatisticas atuais do corpus:

- documentos: 10;
- documentos processados sem erro: 10;
- paginas totais: 706;
- caracteres normalizados: 972.163;
- documentos com introducao detectada: 10;
- documentos com resultados/entrega detectados: 10;
- documentos com conclusao detectada: 10.

Os criterios de inclusao, exclusao, origem/coleta e limpeza estao registrados em `reports/baseline/dataset/summary.json`.

## Pipeline implementado

O baseline executa as seguintes etapas:

1. Ingestao dos PDFs em `data/raw/`.
2. Extracao de texto com `pypdf`.
3. Pre-processamento basico:
   - reparo heuristico de codificacao quando detectado;
   - remocao de quebras artificiais de hifenizacao;
   - normalizacao de espacos;
   - normalizacao sem acentos para busca por cabecalhos.
4. Segmentacao heuristica de secoes:
   - busca por titulos numerados como `1. Introducao`, `Resultados` e `6. Conclusao`;
   - registro explicito de secoes ausentes.
5. Inferencia de alinhamento:
   - comparacao entre `introduction` e as secoes de entrega disponiveis (`results` e `conclusion`);
   - vetorizacao com `TF-IDF`;
   - calculo de similaridade do cosseno;
   - classificacao do score em `high`, `medium`, `low` ou `insufficient_sections`.
6. Exportacao de artefatos:
   - resultados por documento;
   - estatisticas do dataset;
   - protocolo experimental;
   - metricas agregadas de cobertura.

## Protocolo experimental

Esta entrega usa avaliacao descritiva e reproduzivel sobre o corpus fixo de 10 documentos. Nao ha divisao em treino, validacao e teste porque o baseline atual nao treina modelo supervisionado nem ajusta parametros a partir de rotulos.

O controle de vazamento esta documentado em `reports/baseline/protocol/experimental_protocol.md`. Em resumo, o TF-IDF e ajustado somente nos dois trechos comparados dentro de cada documento, sem transferencia de vocabulario, pesos ou estatisticas entre documentos.

## Metricas

Metricas de alinhamento por documento:

- similaridade do cosseno com TF-IDF;
- melhor score entre `introduction` x `results` e `introduction` x `conclusion`;
- rotulo de alinhamento derivado do score.

Metricas agregadas de cobertura:

- sucesso de extracao;
- presenca de introducao, resultados e conclusao;
- documentos avaliaveis;
- distribuicao dos rotulos de alinhamento.

Metricas de BRI como `Precision@k`, `Recall@k`, `MAP` e `NDCG`, assim como metricas de classificacao como `accuracy`, `F1`, `macro-F1` e `RMSE`, ficam definidas como trabalho futuro porque exigem anotacao manual de relevancia ou rotulos esperados.

## Saidas geradas

Ao executar o comando principal, o baseline gera:

- `reports/baseline/extraction_summary.json`: resumo completo da execucao;
- `reports/baseline/dataset/summary.json`: descricao e estatisticas do dataset;
- `reports/baseline/protocol/experimental_protocol.json`: protocolo experimental estruturado;
- `reports/baseline/extracted_text/*.txt`: texto bruto extraido de cada PDF;
- `reports/baseline/extracted_text/*_normalized.txt`: versao normalizada do texto;
- `reports/baseline/segmentation/*_sections.json`: secoes identificadas por heuristica;
- `reports/baseline/alignment/results.json`: detalhes das comparacoes e scores;
- `reports/baseline/metrics.csv`: tabela consolidada com resultados por documento;
- `reports/baseline/metrics/coverage_metrics.json`: metricas agregadas de cobertura;
- `reports/baseline/metrics/coverage_metrics.csv`: versao tabular das metricas agregadas.

## Resultados atuais

Resultado consolidado da execucao atual:

```text
document_id,status,promise_section,delivery_sections,alignment_score,alignment_label,notes
AMDavid,ok,introduction,results;conclusion,0.4040,high,
ENPinho,ok,introduction,results;conclusion,0.3280,medium,
FMRolim,ok,introduction,results;conclusion,0.5123,high,
GGSouza,ok,introduction,results;conclusion,0.4525,high,
HBCReis,ok,introduction,results;conclusion,0.4680,high,
HBMHenriques,ok,introduction,results;conclusion,0.3190,medium,
LCMarques,ok,introduction,results;conclusion,0.0000,low,
LMFGaleno,ok,introduction,results;conclusion,0.6510,high,
MDFonseca,ok,introduction,results;conclusion,0.4065,high,
PVMNascimento,ok,introduction,results;conclusion,0.4224,high,
```

Distribuicao dos rotulos:

- `high`: 7 documentos;
- `medium`: 2 documentos;
- `low`: 1 documento;
- `insufficient_sections`: 0 documentos.

## Como reproduzir

1. Criar e ativar o ambiente virtual.
2. Instalar as dependencias com `pip install -r requirements.txt`.
3. Executar:

```bash
python -m src.run_baseline --input_dir data/raw --output_dir reports/baseline
```

4. Conferir:

```bash
python -c "import json; d=json.load(open('reports/baseline/extraction_summary.json', encoding='utf-8')); print(d['document_count'], d['processed_count'], d['error_count'])"
```

O resultado esperado e `10 10 0`.

## Limitacoes conhecidas

- A segmentacao depende de cabecalhos simples e numerados, podendo falhar em PDFs com estrutura diferente.
- A deteccao de secoes de entrega ainda e heuristica e pode selecionar secoes amplas demais.
- O baseline ainda nao implementa um classificador DoCO supervisionado.
- O corpus consolidado ainda e pequeno para generalizacao estatistica.
- A similaridade por `TF-IDF` captura principalmente sobreposicao lexical e nao nuances semanticas profundas.
- Nao ha anotacao manual nesta entrega, portanto metricas supervisionadas e metricas de ranking ficam como trabalho futuro.
