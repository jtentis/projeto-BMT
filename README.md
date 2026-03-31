# João Pedro Tentis e Pedro Mion
# Projeto para disciplina Busca e Mineração de Texto

Auditoria automatica de trabalhos academicos: verificacao de alinhamento entre promessas e entregas com uso do DoCO.

## Objetivo do projeto

Este projeto tem como objetivo desenvolver um sistema capaz de analisar documentos academicos em PDF e verificar se as promessas feitas na introducao estao alinhadas com os resultados e conclusoes apresentados no texto.

A proposta se baseia na **Document Components Ontology (DoCO)** para identificar a funcao retorica de trechos do documento, combinando:

- extracao e segmentacao de texto;
- classificacao de trechos com base no DoCO;
- analise de similaridade semantica entre promessas e entregas.

Ao final, a ideia e gerar um indicador de consistencia interna do trabalho academico, que possa apoiar revisores e leitores.

## Fluxo previsto

O sistema esta organizado em tres etapas principais:

1. Extracao e segmentacao do texto de artigos em PDF.
2. Classificacao dos trechos segundo categorias do DoCO, como introducao, resultados e conclusao.
3. Verificacao de alinhamento entre o que o trabalho promete e o que efetivamente entrega.

## Tecnologias e abordagens previstas

Com base na proposta, o projeto deve explorar recursos de mineracao de texto e processamento de linguagem natural, incluindo:

- classificacao de texto cientifico;
- embeddings para representacao vetorial;
- similaridade do cosseno para comparacao semantica;
- possivel uso de modelos voltados para texto cientifico, como SciBERT.

## Estrutura inicial do repositorio

O repositorio foi organizado inicialmente com a seguinte estrutura:

```text
.
|-- data/
|-- notebooks/
|-- reports/
|-- src/
`-- README.md
```

## Instrucoes preliminares de execucao

Este repositorio ainda esta em fase inicial. As instrucoes abaixo servem como ponto de partida para o desenvolvimento:

1. Clone o repositorio:

```bash
git clone https://github.com/jtentis/projeto-BMT
cd projeto-BMT
```

2. Crie e ative um ambiente virtual:

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

3. Instale as dependencias do projeto assim que elas forem definidas:

```bash
pip install -r requirements.txt
```

4. Organize o desenvolvimento seguindo a estrutura inicial:

- `data/` para corpus, PDFs e dados intermediarios;
- `src/` para o codigo-fonte do sistema;
- `notebooks/` para experimentos e analises exploratorias;
- `reports/` para relatorios, resultados e documentacao complementar.

## Status atual

O projeto esta em etapa de estruturacao inicial. As proximas fases previstas na proposta incluem:

- levantamento do corpus;
- anotacao manual e treinamento do classificador;
- implementacao da etapa de alinhamento;
- avaliacao dos resultados;
- redacao final do trabalho.

## Referencia da proposta

Este README foi elaborado a partir da proposta:

- `Proposta_de_projeto_BMT.pdf`
