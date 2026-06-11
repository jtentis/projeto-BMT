# Projeto BMT

Auditoria automática de trabalhos acadêmicos para verificar alinhamento entre promessas da introdução e entregas em resultados, discussão ou conclusão. O projeto combina rotulagem DoCO, baseline TF-IDF, comparação com SciBERT, métricas formais e persistência em SQLite.

## Visão Geral

O pipeline processa PDFs acadêmicos, extrai texto, identifica seções retóricas, compara pares de seções e exporta resultados reprodutíveis. A versão atual trabalha com 50 PDFs do Pantheon/UFRJ.

A entrega final inclui:

- corpus consolidado com 50 PDFs;
- extração e normalização de texto;
- segmentação por cabeçalhos;
- rotulagem retórica compatível com DoCO;
- baseline TF-IDF com similaridade do cosseno;
- comparação com embeddings SciBERT;
- ground truth provisório com 100 pares;
- métricas de ranking e classificação;
- exportação para SQLite;
- demo CLI.

## Estrutura

```text
.
|-- data/
|   |-- annotations/
|   |-- pdfs/
|   `-- raw/
|-- demo/
|-- reports/
|   |-- baseline/
|   |-- evaluation/
|   `-- final/
|-- src/
|-- make_results.sh
|-- requirements.txt
`-- README.md
```

Arquivos principais:

- `src/run_baseline.py`: extrai texto, segmenta seções, aplica rótulos DoCO e calcula alinhamento.
- `src/evaluate.py`: calcula métricas formais a partir do ground truth.
- `src/export_database.py`: exporta os artefatos finais para SQLite.
- `src/validate_annotations.py`: valida a cobertura das anotações.
- `data/annotations/ground_truth.csv`: anotações provisórias e auditáveis.
- `demo/run_demo.py`: demonstração CLI da entrega final.
- `reports/05_artigo_final_JOAOTENTIS.tex`: artigo final local.

## Ambiente

- Python 3.13
- Dependências em `requirements.txt`

Crie o ambiente:

```bash
python -m venv .venv
```

Ative no Windows:

```bash
.venv\Scripts\activate
```

Ative no Linux/macOS:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

O `transformers` baixa o SciBERT (`allenai/scibert_scivocab_uncased`) na primeira execução com `--method scibert` ou `--method all`.

## Reprodução Completa

Em um shell compatível com Bash:

```bash
bash make_results.sh
```

No PowerShell, execute:

```bash
python -m src.run_baseline --input_dir data/raw --output_dir reports/baseline --method all
python -m src.validate_annotations
python -m src.evaluate
python -m src.export_database
python demo/run_demo.py
python demo/run_demo.py --from-db
```

## Execução Do Pipeline

Somente TF-IDF:

```bash
python -m src.run_baseline --input_dir data/raw --output_dir reports/baseline --method tfidf
```

Somente SciBERT:

```bash
python -m src.run_baseline --input_dir data/raw --output_dir reports/baseline --method scibert
```

Comparação completa:

```bash
python -m src.run_baseline --input_dir data/raw --output_dir reports/baseline --method all
```

## Avaliação Formal

Valide o ground truth:

```bash
python -m src.validate_annotations
```

Calcule as métricas:

```bash
python -m src.evaluate
```

Saídas:

- `reports/evaluation/ranking_metrics.csv`
- `reports/evaluation/classification_metrics.csv`
- `reports/evaluation/per_document_eval.csv`
- `reports/evaluation/pair_classification_eval.csv`
- `reports/evaluation/method_comparison.json`

## Demo

```bash
python demo/run_demo.py
```

A demo mostra estatísticas do corpus, comparação TF-IDF vs SciBERT, métricas formais e o caso crítico `LCMarques`.

Para consultar os resultados persistidos no SQLite:

```bash
python -m src.export_database
python demo/run_demo.py --from-db
```

## Banco SQLite

O exportador cria o banco em:

```text
reports/final/results.db
```

Tabelas principais:

- `documents`
- `alignment_results`
- `alignment_pairs`
- `ground_truth`
- `ranking_metrics`
- `classification_metrics`
- `per_document_eval`

Consulta rápida das métricas de ranking:

```bash
python -c "import sqlite3; con=sqlite3.connect('reports/final/results.db'); [print(r) for r in con.execute('SELECT method, precision_at_1, map, ndcg_at_2 FROM ranking_metrics')]"
```

Consulta do caso crítico:

```bash
python -c "import sqlite3; con=sqlite3.connect('reports/final/results.db'); [print(r) for r in con.execute(\"SELECT method, document_id, alignment_score, alignment_label FROM alignment_results WHERE document_id='LCMarques'\")]"
```

## Dataset

O corpus final contém 50 trabalhos acadêmicos em PDF, coletados do Pantheon/UFRJ e versionados em `data/raw/`.

### Origem e Termos de Uso

Os PDFs usados neste projeto foram obtidos no Pantheon, Repositório Institucional da UFRJ, mantido pelo Sistema de Bibliotecas e Informação (SiBI/UFRJ). Segundo as orientações de uso do Pantheon, o repositório participa do movimento de acesso aberto e tem a missão de reunir, preservar e disseminar a produção acadêmica da UFRJ.

Este repositório usa os documentos apenas para fins acadêmicos, experimentais e reprodutíveis no contexto da disciplina. Os direitos autorais permanecem com os autores, exceto quando houve cessão formal a terceiros. As orientações do Pantheon também informam que gestores das comunidades definem permissões de acesso e embargos, e que o repositório respeita direitos de propriedade intelectual de terceiros.

Fonte consultada: https://pantheon.ufrj.br/terms/guidance.jsp

Estatísticas atuais:

- documentos: 50;
- documentos processados sem erro: 50;
- documentos avaliáveis no ground truth: 50;
- páginas totais: 3.595;
- caracteres normalizados: 5.867.001;
- anotações no ground truth: 100 pares;
- documentos com introdução detectada: 41;
- documentos com resultados detectados: 40;
- documentos com conclusão detectada: 36.

Distribuição dos rótulos TF-IDF:

- `high`: 24 documentos;
- `medium`: 2 documentos;
- `low`: 14 documentos;
- `insufficient_sections`: 10 documentos.

## Resultados principais

Métricas de ranking:

```text
method,precision_at_1,precision_at_2,map,ndcg_at_1,ndcg_at_2
tfidf,0.52,0.47,0.52,0.52,0.52
scibert,0.50,0.47,0.51,0.46,0.5004
```

Métricas de classificação:

```text
method,accuracy,precision,recall,f1,macro_f1
tfidf,0.9333,0.9015,0.9327,0.9144,0.9144
scibert,0.5067,0.5098,0.3571,0.2752,0.2752
```

O TF-IDF ficou acima do SciBERT nas métricas formais atuais. Essa diferença não prova superioridade semântica do TF-IDF. Ela mostra que, neste corpus e com este ground truth provisório, a sobreposição lexical separa melhor os pares anotados. O SciBERT atribui pontuações altas a muitos textos acadêmicos longos e perde capacidade discriminativa sem calibração.

## Artefatos do artigo

- Dataset: `reports/baseline/dataset/summary.json`
- DoCO: `reports/baseline/doco_sections/`
- TF-IDF: `reports/baseline/alignment/tfidf_results.json`
- SciBERT: `reports/baseline/alignment/scibert_results.json`
- Comparação: `reports/baseline/alignment/comparison.csv`
- Ground truth: `data/annotations/ground_truth.csv`
- Métricas: `reports/evaluation/`
- Banco SQLite: `reports/final/results.db`

## Limitações

- A segmentação depende de cabeçalhos extraídos do PDF.
- Alguns PDFs geram sumários ou trechos vazios como seções candidatas.
- O ground truth é provisório e ainda precisa de auditoria independente.
- O SciBERT roda sem fine-tuning e sem calibração.
- As pontuações indicam alinhamento textual aproximado, não qualidade acadêmica.
