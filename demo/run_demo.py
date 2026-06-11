from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "reports" / "baseline"
EVALUATION_DIR = ROOT / "reports" / "evaluation"
DATABASE_PATH = ROOT / "reports" / "final" / "results.db"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo da entrega final do projeto BMT.")
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Ler resultados do banco SQLite em reports/final/results.db.",
    )
    return parser


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def print_table(rows: list[dict[str, str]], columns: list[str]) -> None:
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))


def print_artifact_demo() -> None:
    dataset = read_json(BASELINE_DIR / "dataset" / "summary.json")
    coverage = read_json(BASELINE_DIR / "metrics" / "coverage_metrics.json")
    comparison = read_csv(BASELINE_DIR / "alignment" / "comparison.csv")
    ranking = read_csv(EVALUATION_DIR / "ranking_metrics.csv")
    classification = read_csv(EVALUATION_DIR / "classification_metrics.csv")

    print("Projeto BMT - demo da entrega final")
    print()
    print("Corpus")
    print(f"- documentos: {dataset['document_count']}")
    print(f"- processados: {dataset['processed_count']}")
    print(f"- erros: {dataset['error_count']}")
    print(f"- paginas: {dataset['page_count_total']}")
    print(f"- avaliaveis: {coverage['evaluable_count']}")
    print()

    print("Distribuicao TF-IDF")
    for label, count in coverage["alignment_label_distribution"].items():
        print(f"- {label}: {count}")
    print()

    print("Comparacao por documento")
    print_table(
        comparison,
        ["document_id", "tfidf_score", "tfidf_label", "scibert_score", "scibert_label"],
    )
    print()

    print("Metricas de ranking")
    print_table(ranking, ["method", "precision_at_1", "precision_at_2", "map", "ndcg_at_2"])
    print()

    print("Metricas de classificacao")
    print_table(classification, ["method", "accuracy", "precision", "recall", "macro_f1"])
    print()

    critical = next(row for row in comparison if row["document_id"] == "LCMarques")
    print("Caso critico")
    print(
        "LCMarques combina score TF-IDF 0.0000 com score SciBERT "
        f"{critical['scibert_score']}, sugerindo falha de segmentacao ou baixa "
        "sobreposicao lexical no baseline."
    )


def fetch_dicts(connection: sqlite3.Connection, query: str) -> list[dict[str, str]]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(query).fetchall()
    return [{key: str(row[key]) for key in row.keys()} for row in rows]


def print_database_demo() -> None:
    if not DATABASE_PATH.exists():
        raise SystemExit(
            "Banco SQLite nao encontrado. Execute python -m src.export_database primeiro."
        )

    with sqlite3.connect(DATABASE_PATH) as connection:
        document_stats = connection.execute(
            """
            SELECT
                COUNT(*) AS document_count,
                SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS processed_count,
                SUM(page_count) AS page_count
            FROM documents
            """
        ).fetchone()
        ranking = fetch_dicts(
            connection,
            """
            SELECT method, precision_at_1, precision_at_2, map, ndcg_at_2
            FROM ranking_metrics
            ORDER BY method
            """,
        )
        classification = fetch_dicts(
            connection,
            """
            SELECT method, accuracy, precision, recall, macro_f1
            FROM classification_metrics
            ORDER BY method
            """,
        )
        comparison = fetch_dicts(
            connection,
            """
            SELECT
                t.document_id,
                printf('%.4f', t.alignment_score) AS tfidf_score,
                t.alignment_label AS tfidf_label,
                printf('%.4f', s.alignment_score) AS scibert_score,
                s.alignment_label AS scibert_label
            FROM alignment_results t
            JOIN alignment_results s
              ON s.document_id = t.document_id
             AND s.method = 'scibert'
            WHERE t.method = 'tfidf'
            ORDER BY t.document_id
            """,
        )
        critical = fetch_dicts(
            connection,
            """
            SELECT
                t.document_id,
                printf('%.4f', t.alignment_score) AS tfidf_score,
                printf('%.4f', s.alignment_score) AS scibert_score
            FROM alignment_results t
            JOIN alignment_results s
              ON s.document_id = t.document_id
             AND s.method = 'scibert'
            WHERE t.method = 'tfidf'
              AND t.document_id = 'LCMarques'
            """,
        )[0]

    print("Projeto BMT - demo SQLite")
    print()
    print("Consulta SQL: resumo do corpus")
    print(f"- documentos: {document_stats[0]}")
    print(f"- processados: {document_stats[1]}")
    print(f"- paginas: {document_stats[2]}")
    print()

    print("Consulta SQL: comparacao por documento")
    print_table(
        comparison,
        ["document_id", "tfidf_score", "tfidf_label", "scibert_score", "scibert_label"],
    )
    print()

    print("Consulta SQL: metricas de ranking")
    print_table(ranking, ["method", "precision_at_1", "precision_at_2", "map", "ndcg_at_2"])
    print()

    print("Consulta SQL: metricas de classificacao")
    print_table(classification, ["method", "accuracy", "precision", "recall", "macro_f1"])
    print()

    print("Consulta SQL: caso critico")
    print(
        f"{critical['document_id']}: TF-IDF={critical['tfidf_score']} "
        f"SciBERT={critical['scibert_score']}"
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.from_db:
        print_database_demo()
    else:
        print_artifact_demo()


if __name__ == "__main__":
    main()
