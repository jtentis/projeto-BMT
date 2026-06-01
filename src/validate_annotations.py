from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_CANDIDATES = {"results", "conclusion"}
EXPECTED_QUERY = "introduction"
VALID_RELEVANCE = {"0", "1", "2"}
VALID_ALIGNMENT = {"aligned", "partial", "not_aligned"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valida a cobertura do ground truth da avaliacao final."
    )
    parser.add_argument(
        "--metrics_csv",
        type=Path,
        default=Path("reports/baseline/metrics.csv"),
        help="CSV de resultados por documento gerado pelo baseline.",
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        default=Path("data/annotations/ground_truth.csv"),
        help="CSV com anotacoes manuais provisoriais.",
    )
    return parser


def read_metric_documents(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return {row["document_id"] for row in csv.DictReader(csv_file)}


def read_annotations(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def validate_annotations(metric_documents: set[str], rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    annotated_documents = {row["document_id"] for row in rows}

    missing_documents = sorted(metric_documents - annotated_documents)
    extra_documents = sorted(annotated_documents - metric_documents)
    if missing_documents:
        errors.append("Documentos sem anotacao: " + ", ".join(missing_documents))
    if extra_documents:
        errors.append("Documentos anotados fora das metricas: " + ", ".join(extra_documents))

    for document_id in sorted(metric_documents):
        pairs = {
            (row["query_section"], row["candidate_section"])
            for row in rows
            if row["document_id"] == document_id
        }
        expected_pairs = {(EXPECTED_QUERY, candidate) for candidate in EXPECTED_CANDIDATES}
        missing_pairs = sorted(expected_pairs - pairs)
        if missing_pairs:
            formatted = ", ".join(f"{query}->{candidate}" for query, candidate in missing_pairs)
            errors.append(f"Pares ausentes em {document_id}: {formatted}")

    for index, row in enumerate(rows, start=2):
        if row["query_section"] != EXPECTED_QUERY:
            errors.append(f"Linha {index}: query_section invalida")
        if row["candidate_section"] not in EXPECTED_CANDIDATES:
            errors.append(f"Linha {index}: candidate_section invalida")
        if row["relevance"] not in VALID_RELEVANCE:
            errors.append(f"Linha {index}: relevance invalida")
        if row["expected_alignment"] not in VALID_ALIGNMENT:
            errors.append(f"Linha {index}: expected_alignment invalido")

    return errors


def main() -> None:
    args = build_parser().parse_args()
    metric_documents = read_metric_documents(args.metrics_csv)
    rows = read_annotations(args.ground_truth)
    errors = validate_annotations(metric_documents, rows)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print(f"Ground truth valido: {len(metric_documents)} documentos e {len(rows)} anotacoes.")


if __name__ == "__main__":
    main()
