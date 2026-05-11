# Metricas formais

Esta entrega mantem o baseline de alinhamento textual e adiciona metricas de cobertura do pipeline para avaliar se o desenho experimental esta estavel antes da evolucao dos modelos.

## Metricas calculadas

As metricas de alinhamento calculadas por documento sao:

- `tfidf_cosine_similarity`: similaridade do cosseno entre vetores TF-IDF;
- `best_introduction_to_delivery_score`: maior score entre `introduction` x `results` e `introduction` x `conclusion`;
- `alignment_label`: rotulo discreto derivado do score final.

Os limiares usados para `alignment_label` sao:

- `high`: score maior ou igual a 0.35;
- `medium`: score maior ou igual a 0.15 e menor que 0.35;
- `low`: score menor que 0.15;
- `insufficient_sections`: documento sem secoes suficientes para comparacao.

As metricas de cobertura do pipeline sao:

- `document_count`: quantidade de PDFs no corpus consolidado;
- `processed_count`: quantidade de documentos processados sem erro;
- `error_count`: quantidade de documentos com erro;
- `extraction_success_rate`: proporcao de documentos processados sem erro;
- `has_introduction_count`: documentos com introducao detectada;
- `has_results_count`: documentos com secao de entrega/resultados detectada;
- `has_conclusion_count`: documentos com conclusao detectada;
- `evaluable_count`: documentos com score de alinhamento calculado;
- `evaluable_rate`: proporcao de documentos avaliaveis;
- `alignment_label_distribution`: distribuicao dos rotulos de alinhamento.

## Metricas BRI e MT

Metricas de BRI como `Precision@k`, `Recall@k`, `MAP` e `NDCG` exigem julgamentos de relevancia anotados. Metricas de MT/classificacao como `accuracy`, `F1`, `macro-F1` e `RMSE` exigem rotulos esperados por documento ou por trecho. Como esta entrega nao inclui anotacao manual, essas metricas ficam formalmente definidas como trabalho futuro e nao sao computadas agora.

## Arquivos gerados

- `reports/baseline/metrics.csv`: resultados de alinhamento por documento;
- `reports/baseline/metrics/coverage_metrics.json`: metricas agregadas de cobertura;
- `reports/baseline/metrics/coverage_metrics.csv`: versao tabular das metricas agregadas.
