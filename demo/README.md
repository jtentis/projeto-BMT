# Demo da entrega final

Esta demo resume os resultados finais do projeto BMT a partir dos artefatos versionados em `reports/baseline/` e `reports/evaluation/`.

## Execucao

Na raiz do repositorio:

```bash
python demo/run_demo.py
```

Para ler os resultados persistidos no SQLite:

```bash
python demo/run_demo.py --from-db
```

## Saida esperada

A demo imprime:

- estatisticas do corpus consolidado;
- distribuicao de rotulos do baseline TF-IDF;
- comparacao TF-IDF vs SciBERT por documento;
- metricas formais de ranking;
- metricas formais de classificacao;
- destaque do caso critico `LCMarques`.
- consultas SQL quando executada com `--from-db`.

Antes de executar a demo do zero, reproduza os artefatos com:

```bash
bash make_results.sh
```
