from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exporta os artefatos finais do projeto para um banco SQLite."
    )
    parser.add_argument(
        "--output_db",
        type=Path,
        default=Path("reports/final/results.db"),
        help="Caminho do banco SQLite a ser criado.",
    )
    parser.add_argument(
        "--baseline_dir",
        type=Path,
        default=Path("reports/baseline"),
        help="Diretorio com artefatos do baseline.",
    )
    parser.add_argument(
        "--evaluation_dir",
        type=Path,
        default=Path("reports/evaluation"),
        help="Diretorio com metricas formais.",
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        default=Path("data/annotations/ground_truth.csv"),
        help="CSV com ground truth manual/provisorio.",
    )
    return parser


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def reset_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP TABLE IF EXISTS documents;
        DROP TABLE IF EXISTS alignment_results;
        DROP TABLE IF EXISTS alignment_pairs;
        DROP TABLE IF EXISTS ground_truth;
        DROP TABLE IF EXISTS ranking_metrics;
        DROP TABLE IF EXISTS classification_metrics;
        DROP TABLE IF EXISTS per_document_eval;

        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY,
            pdf_path TEXT NOT NULL,
            status TEXT NOT NULL,
            page_count INTEGER,
            text_length INTEGER,
            normalized_text_length INTEGER
        );

        CREATE TABLE alignment_results (
            method TEXT NOT NULL,
            document_id TEXT NOT NULL,
            status TEXT NOT NULL,
            promise_section TEXT NOT NULL,
            promise_doco_section TEXT NOT NULL,
            delivery_sections TEXT NOT NULL,
            delivery_doco_sections TEXT NOT NULL,
            alignment_score REAL,
            alignment_label TEXT NOT NULL,
            PRIMARY KEY (method, document_id)
        );

        CREATE TABLE alignment_pairs (
            method TEXT NOT NULL,
            document_id TEXT NOT NULL,
            promise_section TEXT NOT NULL,
            promise_doco_section TEXT NOT NULL,
            delivery_section TEXT NOT NULL,
            delivery_doco_section TEXT NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (method, document_id, promise_section, delivery_section)
        );

        CREATE TABLE ground_truth (
            document_id TEXT NOT NULL,
            query_section TEXT NOT NULL,
            candidate_section TEXT NOT NULL,
            relevance INTEGER NOT NULL,
            expected_alignment TEXT NOT NULL,
            notes TEXT NOT NULL,
            PRIMARY KEY (document_id, query_section, candidate_section)
        );

        CREATE TABLE ranking_metrics (
            method TEXT PRIMARY KEY,
            precision_at_1 REAL NOT NULL,
            precision_at_2 REAL NOT NULL,
            map REAL NOT NULL,
            ndcg_at_1 REAL NOT NULL,
            ndcg_at_2 REAL NOT NULL
        );

        CREATE TABLE classification_metrics (
            method TEXT PRIMARY KEY,
            accuracy REAL NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            f1 REAL NOT NULL,
            macro_f1 REAL NOT NULL
        );

        CREATE TABLE per_document_eval (
            method TEXT NOT NULL,
            document_id TEXT NOT NULL,
            ranked_sections TEXT NOT NULL,
            precision_at_1 REAL NOT NULL,
            precision_at_2 REAL NOT NULL,
            ap REAL NOT NULL,
            ndcg_at_1 REAL NOT NULL,
            ndcg_at_2 REAL NOT NULL,
            PRIMARY KEY (method, document_id)
        );
        """
    )


def insert_documents(connection: sqlite3.Connection, baseline_dir: Path) -> None:
    dataset = read_json(baseline_dir / "dataset" / "summary.json")
    rows = [
        (
            document["document_id"],
            document["pdf_path"],
            document["status"],
            document["page_count"],
            document["text_length"],
            document["normalized_text_length"],
        )
        for document in dataset["documents"]
    ]
    connection.executemany(
        """
        INSERT INTO documents (
            document_id,
            pdf_path,
            status,
            page_count,
            text_length,
            normalized_text_length
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_alignment(connection: sqlite3.Connection, baseline_dir: Path, method: str) -> None:
    payload = read_json(baseline_dir / "alignment" / f"{method}_results.json")
    result_rows = []
    pair_rows = []
    for document in payload["documents"]:
        result_rows.append(
            (
                document["method"],
                document["document_id"],
                document["status"],
                document["promise_section"],
                document["promise_doco_section"],
                ";".join(document["delivery_sections"]),
                ";".join(document["delivery_doco_sections"]),
                document["alignment_score"],
                document["alignment_label"],
            )
        )
        for pair in document["compared_pairs"]:
            pair_rows.append(
                (
                    pair["method"],
                    document["document_id"],
                    pair["promise_section"],
                    pair["promise_doco_section"],
                    pair["delivery_section"],
                    pair["delivery_doco_section"],
                    pair["score"],
                )
            )

    connection.executemany(
        """
        INSERT INTO alignment_results (
            method,
            document_id,
            status,
            promise_section,
            promise_doco_section,
            delivery_sections,
            delivery_doco_sections,
            alignment_score,
            alignment_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        result_rows,
    )
    connection.executemany(
        """
        INSERT INTO alignment_pairs (
            method,
            document_id,
            promise_section,
            promise_doco_section,
            delivery_section,
            delivery_doco_section,
            score
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        pair_rows,
    )


def insert_ground_truth(connection: sqlite3.Connection, ground_truth_path: Path) -> None:
    rows = [
        (
            row["document_id"],
            row["query_section"],
            row["candidate_section"],
            int(row["relevance"]),
            row["expected_alignment"],
            row["notes"],
        )
        for row in read_csv(ground_truth_path)
    ]
    connection.executemany(
        """
        INSERT INTO ground_truth (
            document_id,
            query_section,
            candidate_section,
            relevance,
            expected_alignment,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_ranking_metrics(connection: sqlite3.Connection, evaluation_dir: Path) -> None:
    rows = [
        (
            row["method"],
            float(row["precision_at_1"]),
            float(row["precision_at_2"]),
            float(row["map"]),
            float(row["ndcg_at_1"]),
            float(row["ndcg_at_2"]),
        )
        for row in read_csv(evaluation_dir / "ranking_metrics.csv")
    ]
    connection.executemany(
        """
        INSERT INTO ranking_metrics (
            method,
            precision_at_1,
            precision_at_2,
            map,
            ndcg_at_1,
            ndcg_at_2
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_classification_metrics(connection: sqlite3.Connection, evaluation_dir: Path) -> None:
    rows = [
        (
            row["method"],
            float(row["accuracy"]),
            float(row["precision"]),
            float(row["recall"]),
            float(row["f1"]),
            float(row["macro_f1"]),
        )
        for row in read_csv(evaluation_dir / "classification_metrics.csv")
    ]
    connection.executemany(
        """
        INSERT INTO classification_metrics (
            method,
            accuracy,
            precision,
            recall,
            f1,
            macro_f1
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_per_document_eval(connection: sqlite3.Connection, evaluation_dir: Path) -> None:
    rows = [
        (
            row["method"],
            row["document_id"],
            row["ranked_sections"],
            float(row["precision_at_1"]),
            float(row["precision_at_2"]),
            float(row["ap"]),
            float(row["ndcg_at_1"]),
            float(row["ndcg_at_2"]),
        )
        for row in read_csv(evaluation_dir / "per_document_eval.csv")
    ]
    connection.executemany(
        """
        INSERT INTO per_document_eval (
            method,
            document_id,
            ranked_sections,
            precision_at_1,
            precision_at_2,
            ap,
            ndcg_at_1,
            ndcg_at_2
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def main() -> None:
    args = build_parser().parse_args()
    args.output_db.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(args.output_db) as connection:
        reset_database(connection)
        insert_documents(connection, args.baseline_dir)
        insert_alignment(connection, args.baseline_dir, "tfidf")
        insert_alignment(connection, args.baseline_dir, "scibert")
        insert_ground_truth(connection, args.ground_truth)
        insert_ranking_metrics(connection, args.evaluation_dir)
        insert_classification_metrics(connection, args.evaluation_dir)
        insert_per_document_eval(connection, args.evaluation_dir)
        connection.commit()

    print(f"Banco SQLite salvo em: {args.output_db}")


if __name__ == "__main__":
    main()
