from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


LABELS = ["aligned", "partial", "not_aligned"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula metricas formais a partir dos resultados e do ground truth."
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        default=Path("data/annotations/ground_truth.csv"),
        help="CSV com anotacoes manuais provisoriais.",
    )
    parser.add_argument(
        "--tfidf_results",
        type=Path,
        default=Path("reports/baseline/alignment/tfidf_results.json"),
        help="Resultados do metodo TF-IDF.",
    )
    parser.add_argument(
        "--scibert_results",
        type=Path,
        default=Path("reports/baseline/alignment/scibert_results.json"),
        help="Resultados do metodo SciBERT.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/evaluation"),
        help="Diretorio para metricas formais.",
    )
    return parser


def read_ground_truth(path: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    records: dict[tuple[str, str, str], dict[str, object]] = {}
    with path.open(encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            key = (row["document_id"], row["query_section"], row["candidate_section"])
            records[key] = {
                "relevance": int(row["relevance"]),
                "expected_alignment": row["expected_alignment"],
                "notes": row["notes"],
            }
    return records


def read_method_results(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as json_file:
        payload = json.load(json_file)
    return list(payload["documents"])


def predicted_alignment(score: float) -> str:
    if score >= 0.35:
        return "aligned"
    if score >= 0.15:
        return "partial"
    return "not_aligned"


def dcg(relevances: list[int], k: int) -> float:
    total = 0.0
    for index, relevance in enumerate(relevances[:k], start=1):
        total += (2**relevance - 1) / math.log2(index + 1)
    return total


def average_precision(relevances: list[int]) -> float:
    relevant_seen = 0
    precision_sum = 0.0
    total_relevant = sum(1 for relevance in relevances if relevance > 0)
    if total_relevant == 0:
        return 0.0
    for index, relevance in enumerate(relevances, start=1):
        if relevance > 0:
            relevant_seen += 1
            precision_sum += relevant_seen / index
    return precision_sum / total_relevant


def precision_at_k(relevances: list[int], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for relevance in relevances[:k] if relevance > 0) / k


def ndcg_at_k(relevances: list[int], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevances, k) / ideal_dcg


def evaluate_method(
    method: str,
    documents: list[dict[str, object]],
    ground_truth: dict[tuple[str, str, str], dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_document_rows: list[dict[str, object]] = []
    classification_rows: list[dict[str, object]] = []

    for document in documents:
        document_id = str(document["document_id"])
        query_section = str(document["promise_section"])
        ranked_pairs = sorted(
            document["compared_pairs"],
            key=lambda pair: float(pair["score"]),
            reverse=True,
        )
        ranked_relevances: list[int] = []
        ranked_sections: list[str] = []

        for pair in ranked_pairs:
            candidate_section = str(pair["delivery_section"])
            key = (document_id, query_section, candidate_section)
            if key not in ground_truth:
                raise ValueError(f"Ground truth ausente para {document_id} {query_section}->{candidate_section}")
            truth = ground_truth[key]
            score = float(pair["score"])
            ranked_relevances.append(int(truth["relevance"]))
            ranked_sections.append(candidate_section)
            classification_rows.append(
                {
                    "method": method,
                    "document_id": document_id,
                    "query_section": query_section,
                    "candidate_section": candidate_section,
                    "score": score,
                    "predicted_alignment": predicted_alignment(score),
                    "expected_alignment": truth["expected_alignment"],
                    "relevance": truth["relevance"],
                }
            )

        per_document_rows.append(
            {
                "method": method,
                "document_id": document_id,
                "ranked_sections": ";".join(ranked_sections),
                "precision_at_1": precision_at_k(ranked_relevances, 1),
                "precision_at_2": precision_at_k(ranked_relevances, 2),
                "ap": average_precision(ranked_relevances),
                "ndcg_at_1": ndcg_at_k(ranked_relevances, 1),
                "ndcg_at_2": ndcg_at_k(ranked_relevances, 2),
            }
        )

    return per_document_rows, classification_rows


def summarize_ranking(per_document_rows: list[dict[str, object]]) -> dict[str, float]:
    count = len(per_document_rows)
    if count == 0:
        return {
            "precision_at_1": 0.0,
            "precision_at_2": 0.0,
            "map": 0.0,
            "ndcg_at_1": 0.0,
            "ndcg_at_2": 0.0,
        }
    return {
        "precision_at_1": sum(float(row["precision_at_1"]) for row in per_document_rows) / count,
        "precision_at_2": sum(float(row["precision_at_2"]) for row in per_document_rows) / count,
        "map": sum(float(row["ap"]) for row in per_document_rows) / count,
        "ndcg_at_1": sum(float(row["ndcg_at_1"]) for row in per_document_rows) / count,
        "ndcg_at_2": sum(float(row["ndcg_at_2"]) for row in per_document_rows) / count,
    }


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarize_classification(rows: list[dict[str, object]]) -> dict[str, float]:
    total = len(rows)
    correct = sum(1 for row in rows if row["predicted_alignment"] == row["expected_alignment"])
    per_label: dict[str, dict[str, float]] = {}
    for label in LABELS:
        true_positive = sum(
            1
            for row in rows
            if row["predicted_alignment"] == label and row["expected_alignment"] == label
        )
        false_positive = sum(
            1
            for row in rows
            if row["predicted_alignment"] == label and row["expected_alignment"] != label
        )
        false_negative = sum(
            1
            for row in rows
            if row["predicted_alignment"] != label and row["expected_alignment"] == label
        )
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_precision = sum(per_label[label]["precision"] for label in LABELS) / len(LABELS)
    macro_recall = sum(per_label[label]["recall"] for label in LABELS) / len(LABELS)
    macro_f1 = sum(per_label[label]["f1"] for label in LABELS) / len(LABELS)

    return {
        "accuracy": safe_divide(correct, total),
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "macro_f1": macro_f1,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rounded(value: float) -> float:
    return round(value, 4)


def main() -> None:
    args = build_parser().parse_args()
    ground_truth = read_ground_truth(args.ground_truth)
    method_inputs = {
        "tfidf": read_method_results(args.tfidf_results),
        "scibert": read_method_results(args.scibert_results),
    }

    all_per_document_rows: list[dict[str, object]] = []
    all_classification_pairs: list[dict[str, object]] = []
    ranking_summary_rows: list[dict[str, object]] = []
    classification_summary_rows: list[dict[str, object]] = []
    method_comparison: dict[str, object] = {}

    for method, documents in method_inputs.items():
        per_document_rows, classification_pairs = evaluate_method(method, documents, ground_truth)
        ranking_summary = summarize_ranking(per_document_rows)
        classification_summary = summarize_classification(classification_pairs)

        all_per_document_rows.extend(per_document_rows)
        all_classification_pairs.extend(classification_pairs)
        ranking_summary_rows.append(
            {
                "method": method,
                "precision_at_1": rounded(ranking_summary["precision_at_1"]),
                "precision_at_2": rounded(ranking_summary["precision_at_2"]),
                "map": rounded(ranking_summary["map"]),
                "ndcg_at_1": rounded(ranking_summary["ndcg_at_1"]),
                "ndcg_at_2": rounded(ranking_summary["ndcg_at_2"]),
            }
        )
        classification_summary_rows.append(
            {
                "method": method,
                "accuracy": rounded(classification_summary["accuracy"]),
                "precision": rounded(classification_summary["precision"]),
                "recall": rounded(classification_summary["recall"]),
                "f1": rounded(classification_summary["f1"]),
                "macro_f1": rounded(classification_summary["macro_f1"]),
            }
        )
        method_comparison[method] = {
            "ranking": ranking_summary_rows[-1],
            "classification": classification_summary_rows[-1],
            "document_count": len(per_document_rows),
            "pair_count": len(classification_pairs),
        }

    write_csv(
        args.output_dir / "ranking_metrics.csv",
        ranking_summary_rows,
        ["method", "precision_at_1", "precision_at_2", "map", "ndcg_at_1", "ndcg_at_2"],
    )
    write_csv(
        args.output_dir / "classification_metrics.csv",
        classification_summary_rows,
        ["method", "accuracy", "precision", "recall", "f1", "macro_f1"],
    )
    write_csv(
        args.output_dir / "per_document_eval.csv",
        all_per_document_rows,
        ["method", "document_id", "ranked_sections", "precision_at_1", "precision_at_2", "ap", "ndcg_at_1", "ndcg_at_2"],
    )
    write_csv(
        args.output_dir / "pair_classification_eval.csv",
        all_classification_pairs,
        ["method", "document_id", "query_section", "candidate_section", "score", "predicted_alignment", "expected_alignment", "relevance"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "method_comparison.json").write_text(
        json.dumps(method_comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Metricas formais salvas em: {args.output_dir}")
    for row in ranking_summary_rows:
        print(
            f"{row['method']}: P@1={row['precision_at_1']} MAP={row['map']} NDCG@2={row['ndcg_at_2']}"
        )


if __name__ == "__main__":
    main()
