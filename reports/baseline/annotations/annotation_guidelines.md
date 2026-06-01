# Diretrizes de anotacao do ground truth

Este arquivo documenta o criterio usado em `data/annotations/ground_truth.csv`. As anotacoes sao manuais, provisoriais e baseadas na inspecao dos trechos extraidos pelo pipeline atual. Elas servem para viabilizar avaliacao formal na entrega final, nao para afirmar uma verdade absoluta sobre a qualidade dos trabalhos academicos.

## Unidade de anotacao

Cada linha avalia um par formado por:

- `query_section`: a secao de promessa, atualmente `introduction`;
- `candidate_section`: uma secao candidata de entrega, atualmente `results` ou `conclusion`;
- `document_id`: identificador do PDF no corpus consolidado.

## Escala de relevancia

A coluna `relevance` usa escala graduada:

- `0`: o candidato nao responde a promessa extraida ou a segmentacao esta ruidosa demais para comparacao confiavel;
- `1`: o candidato e parcialmente relacionado, mas cobre metodo, contexto, trecho amplo ou entrega incompleta;
- `2`: o candidato responde de forma clara a promessa extraida e descreve entrega, resultado ou conclusao alinhada.

## Alinhamento esperado

A coluna `expected_alignment` deriva da relevancia:

- `aligned`: usado quando `relevance = 2`;
- `partial`: usado quando `relevance = 1`;
- `not_aligned`: usado quando `relevance = 0`.

## Criterios praticos

Uma anotacao deve considerar o conteudo efetivamente extraido e segmentado pelo pipeline, nao o PDF ideal em abstrato. Se a secao extraida contem apenas sumario, texto truncado ou cabecalhos sem conteudo substantivo, o par deve receber penalizacao mesmo que o documento original possua conteudo relevante em outro ponto.

Secoes de desenvolvimento, proposta ou metodo podem receber `1` quando estao relacionadas ao objetivo, mas nao representam entrega final. Conclusoes que retomam explicitamente objetivo, contribuicao ou resultado principal podem receber `2`.

## Casos criticos

`LCMarques` foi marcado como caso critico porque a introducao extraida e muito curta e parece conter linhas de sumario. O score automatico `0.0000` e coerente com essa falha de segmentacao, mas nao deve ser interpretado como avaliacao definitiva do documento original.

`HBMHenriques` e `MDFonseca` tambem possuem pares penalizados por problemas de segmentacao: um candidato aponta para referencial teorico ou organizacao do texto, nao para uma entrega propriamente dita.

## Uso nas metricas

As metricas de ranking devem usar `relevance` como relevancia graduada para Precision@k, MAP e NDCG. As metricas de classificacao devem usar `expected_alignment` como rotulo esperado.
