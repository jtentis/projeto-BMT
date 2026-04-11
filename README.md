# Projeto BMT

Auditoria automatica de trabalhos academicos: verificacao de alinhamento entre promessas e entregas com uso do DoCO.

## Baseline reproduzivel

Este repositorio contem o primeiro baseline funcional do projeto. A meta desta versao e oferecer um pipeline simples, offline e reproduzivel para:

- ler artigos academicos em PDF;
- extrair seu texto;
- identificar trechos de introducao, resultados e conclusao;
- comparar promessas e entregas com similaridade semantica;
- gerar resultados iniciais em arquivos reutilizaveis.

Nesta primeira versao, o baseline usa heuristicas simples de segmentacao por titulos numerados e `TF-IDF + similaridade do cosseno` para gerar um indicador inicial de alinhamento entre promessas e entregas.

## Estrutura principal

```text
.
|-- data/
|-- notebooks/
|-- reports/
|-- src/
|-- requirements.txt
`-- README.md
```

Pastas e arquivos mais relevantes:

- `data/raw/`: corpus pequeno versionado com os PDFs de entrada;
- `src/run_baseline.py`: comando principal do baseline;
- `reports/baseline/`: saidas geradas pela execucao;
- `reports/02_baseline.tex`: relatorio da entrega em LaTeX.

## Ambiente

- Python 3.13
- dependencias em `requirements.txt`

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

## Pipeline implementado

O baseline executa as seguintes etapas:

1. Ingestao dos PDFs em `data/raw/`.
2. Extracao de texto com `pypdf`.
3. Pre-processamento basico do texto:
   - remocao de quebras artificiais de hifenizacao;
   - normalizacao de espacos;
   - normalizacao para busca por cabecalhos.
4. Segmentacao heuristica de secoes:
   - busca por titulos numerados como `1. Introducao` e `6. Conclusao`;
   - registro explicito de secoes ausentes.
5. Inferencia de alinhamento:
   - comparacao entre `Introducao` e a melhor secao de entrega disponivel (`Resultados` ou `Conclusao`);
   - vetorizacao com `TF-IDF`;
   - calculo de similaridade do cosseno;
   - classificacao do score em `high`, `medium`, `low` ou `insufficient_sections`.

## Entradas

- PDFs academicos colocados em `data/raw/`.
- O repositorio ja inclui um PDF de exemplo:
  - `data/raw/proposta_bmt.pdf`

## Saidas geradas

Ao executar o comando principal, o baseline gera:

- `reports/baseline/extraction_summary.json`: resumo completo da execucao e dos documentos processados;
- `reports/baseline/extracted_text/*.txt`: texto bruto extraido de cada PDF;
- `reports/baseline/extracted_text/*_normalized.txt`: versao normalizada do texto;
- `reports/baseline/segmentation/*_sections.json`: secoes identificadas por heuristica;
- `reports/baseline/alignment/results.json`: detalhes das comparacoes e scores;
- `reports/baseline/metrics.csv`: tabela consolidada com as metricas iniciais.

## Resultado inicial atual

Com o corpus atual versionado no repositorio, o baseline produziu:

```text
document_id,status,promise_section,delivery_sections,alignment_score,alignment_label,notes
proposta_bmt,ok,introduction,conclusion,0.0640,low,
```

Interpretacao inicial:

- o documento de exemplo teve `Introducao` e `Conclusao` detectadas;
- a secao `Resultados` nao foi encontrada por titulo numerado;
- a comparacao entre `Introducao` e `Conclusao` gerou score baixo (`0.0640`), o que e esperado dado que o PDF e uma proposta de projeto, nao um artigo final com resultados consolidados.

## Como reproduzir

1. Criar e ativar o ambiente virtual.
2. Instalar as dependencias com `pip install -r requirements.txt`.
3. Executar:

```bash
python -m src.run_baseline --input_dir data/raw --output_dir reports/baseline
```

4. Inspecionar os arquivos gerados em `reports/baseline/`.

## Limitacoes conhecidas

- A segmentacao depende de titulos numerados simples e pode falhar em PDFs com estrutura muito diferente.
- O baseline ainda nao implementa um classificador DoCO supervisionado.
- O corpus atual e pequeno e serve apenas para demonstracao e reproducao inicial.
- A similaridade por `TF-IDF` nao captura nuances semanticas profundas como modelos de embeddings especializados.

## Referencia

- `Proposta_de_projeto_BMT.pdf`
