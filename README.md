# Projeto BMT

Auditoria automatica de trabalhos academicos: verificacao de alinhamento entre promessas e entregas com uso do DoCO.

## Baseline reproduzivel

Este repositorio contem o primeiro baseline funcional do projeto. A meta desta versao e oferecer um pipeline simples, offline e reproduzivel para:

- ler artigos academicos em PDF;
- extrair seu texto;
- identificar trechos de introducao, resultados e conclusao;
- comparar promessas e entregas com similaridade semantica;
- gerar resultados iniciais em arquivos reutilizaveis.

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

As proximas etapas do desenvolvimento preenchem o corpus, a extracao de texto, a segmentacao heuristica e a geracao das metricas finais.

## Referencia

- `Proposta_de_projeto_BMT.pdf`
