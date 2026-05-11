# Protocolo experimental

Esta entrega usa uma avaliacao descritiva e reproduzivel do baseline BMT sobre o corpus final consolidado. O objetivo e reduzir ambiguidades do desenho experimental antes de evoluir para modelos supervisionados ou metricas dependentes de anotacao manual.

## Divisao dos dados

O corpus completo de 10 PDFs em `data/raw/` e usado como conjunto fixo de avaliacao. Nao ha divisao em treino, validacao e teste nesta etapa porque o baseline atual nao treina um modelo supervisionado nem ajusta parametros a partir de rotulos.

## Controle de vazamento

Nao ha vazamento treino-teste no protocolo atual pelos seguintes motivos:

- nao existe conjunto de treino supervisionado;
- nao existem rotulos manuais usados para ajustar limiares ou pesos;
- o TF-IDF e ajustado apenas nos dois trechos comparados dentro de cada documento;
- vocabulario, pesos e estatisticas de um documento nao sao reutilizados em outro documento.

## Rodadas e sementes

O experimento tem uma rodada deterministica. Nao ha sementes aleatorias configuradas porque o pipeline atual usa extracao textual, regras heuristicas de segmentacao e similaridade TF-IDF sem amostragem aleatoria.

## Justificativa para nao usar cross-validation

Cross-validation nao e usada porque o corpus ainda e pequeno e a etapa atual nao envolve treinamento supervisionado. O protocolo prioriza estabilidade, rastreabilidade dos artefatos e verificacao do desenho experimental.

## Baseline comparado

O baseline formal desta entrega e `TF-IDF + similaridade do cosseno`, comparando a secao de promessa (`introduction`) com as secoes de entrega disponiveis (`results` e `conclusion`).
