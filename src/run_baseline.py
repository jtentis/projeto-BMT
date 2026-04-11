from __future__ import annotations

import argparse
import csv
import json
import re
import traceback
import unicodedata
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SectionData(TypedDict):
    header: str
    start_char: int
    end_char: int
    content: str
    content_length: int


class SegmentationResult(TypedDict):
    available_sections: list[str]
    missing_sections: list[str]
    sections: dict[str, SectionData]


class AlignmentResult(TypedDict):
    document_id: str
    status: str
    promise_section: str
    delivery_sections: list[str]
    compared_pairs: list[dict[str, object]]
    alignment_score: float | None
    alignment_label: str
    notes: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Executa o baseline reproduzivel do projeto BMT "
            "para analise de alinhamento em artigos academicos."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw"),
        help="Diretorio com os PDFs de entrada.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/baseline"),
        help="Diretorio onde os artefatos do baseline serao gerados.",
    )
    return parser


def list_pdf_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.pdf") if path.is_file())


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    page_texts = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    return "\n".join(page_texts).strip()


def save_text_artifact(output_dir: Path, pdf_path: Path, text: str) -> Path:
    extracted_dir = output_dir / "extracted_text"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    output_path = extracted_dir / f"{pdf_path.stem}.txt"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def fix_mojibake(text: str) -> str:
    if "Ã" not in text and "â" not in text:
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text
    if repaired.count("Ã") < text.count("Ã"):
        return repaired
    return text


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([a-zà-ÿ])-+\n([a-zà-ÿ])", r"\1\2", text, flags=re.IGNORECASE)
    normalized_lines = []
    for line in text.splitlines():
        collapsed = re.sub(r"[ \t]+", " ", line).strip()
        if collapsed:
            normalized_lines.append(collapsed)
    return "\n".join(normalized_lines).strip()


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_for_matching(text: str) -> str:
    text = strip_accents(text.lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def segment_sections(normalized_text: str) -> SegmentationResult:
    search_text = normalize_for_matching(normalized_text)

    section_patterns = {
        "introduction": r"(?:^|\s)(\d+\.\s*)(introducao|introduction)\b",
        "results": r"(?:^|\s)(\d+\.\s*)(resultados?|results?|resultado e discussao)\b",
        "conclusion": r"(?:^|\s)(\d+\.\s*)(conclusao|conclusoes|conclusion)\b",
    }
    boundary_pattern = re.compile(r"(?:^|\s)(\d+\.\s+[a-z])", flags=re.IGNORECASE)

    positions: list[tuple[str, int, int]] = []
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, search_text, flags=re.IGNORECASE)
        if match:
            positions.append((section_name, match.start(), match.end()))

    positions.sort(key=lambda item: item[1])

    sections: dict[str, SectionData] = {}
    for section_name, start_char, header_end in positions:
        end_char = len(search_text)

        for _, candidate_start, _ in positions:
            if candidate_start > start_char:
                end_char = min(end_char, candidate_start)

        for boundary_match in boundary_pattern.finditer(search_text, pos=header_end):
            if boundary_match.start() > start_char:
                end_char = min(end_char, boundary_match.start())
                break

        header = search_text[start_char:header_end].strip()
        content = search_text[header_end:end_char].strip()
        sections[section_name] = {
            "header": header,
            "start_char": start_char,
            "end_char": end_char,
            "content": content,
            "content_length": len(content),
        }

    return {
        "available_sections": sorted(sections.keys()),
        "missing_sections": [
            section_name
            for section_name in ("introduction", "results", "conclusion")
            if section_name not in sections
        ],
        "sections": sections,
    }


def save_json_artifact(output_dir: Path, relative_dir: str, name: str, payload: object) -> Path:
    artifact_dir = output_dir / relative_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifact_dir / name
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def compute_similarity(text_a: str, text_b: str) -> float:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([text_a, text_b])
    similarity_matrix = cosine_similarity(matrix, matrix)
    score = similarity_matrix[0, 1]
    return float(score)


def label_alignment(score: float | None) -> str:
    if score is None:
        return "insufficient_sections"
    if score >= 0.35:
        return "high"
    if score >= 0.15:
        return "medium"
    return "low"


def run_alignment(documents: list[dict[str, object]], output_dir: Path) -> Path:
    results: list[AlignmentResult] = []

    for document in documents:
        document_id = str(document["document_id"])
        if document["status"] != "ok":
            results.append(
                {
                    "document_id": document_id,
                    "status": "error",
                    "promise_section": "introduction",
                    "delivery_sections": [],
                    "compared_pairs": [],
                    "alignment_score": None,
                    "alignment_label": "insufficient_sections",
                    "notes": ["Documento com falha na extração."],
                }
            )
            continue

        sections = document["sections"]
        if not isinstance(sections, dict):
            continue

        section_map = sections["sections"]
        introduction = section_map.get("introduction")
        compared_pairs: list[dict[str, object]] = []
        notes: list[str] = []

        if introduction is None:
            notes.append("Introducao nao encontrada para comparacao.")

        for delivery_name in ("results", "conclusion"):
            delivery_section = section_map.get(delivery_name)
            if introduction is None or delivery_section is None:
                continue

            score = compute_similarity(introduction["content"], delivery_section["content"])
            compared_pairs.append(
                {
                    "promise_section": "introduction",
                    "delivery_section": delivery_name,
                    "score": round(score, 4),
                }
            )

        if not compared_pairs:
            alignment_score: float | None = None
            notes.append("Nao houve secoes suficientes para calcular alinhamento.")
        else:
            alignment_score = max(pair["score"] for pair in compared_pairs if isinstance(pair["score"], float))

        result: AlignmentResult = {
            "document_id": document_id,
            "status": "ok",
            "promise_section": "introduction",
            "delivery_sections": [str(pair["delivery_section"]) for pair in compared_pairs],
            "compared_pairs": compared_pairs,
            "alignment_score": alignment_score,
            "alignment_label": label_alignment(alignment_score),
            "notes": notes,
        }
        results.append(result)

    json_path = save_json_artifact(output_dir, "alignment", "results.json", {"documents": results})

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "document_id",
                "status",
                "promise_section",
                "delivery_sections",
                "alignment_score",
                "alignment_label",
                "notes",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "document_id": result["document_id"],
                    "status": result["status"],
                    "promise_section": result["promise_section"],
                    "delivery_sections": ";".join(result["delivery_sections"]),
                    "alignment_score": "" if result["alignment_score"] is None else f"{result['alignment_score']:.4f}",
                    "alignment_label": result["alignment_label"],
                    "notes": " | ".join(result["notes"]),
                }
            )

    return json_path


def run_extraction(input_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    for pdf_path in list_pdf_files(input_dir):
        record: dict[str, object] = {
            "document_id": pdf_path.stem,
            "pdf_path": str(pdf_path),
            "status": "ok",
        }
        try:
            text = extract_text_from_pdf(pdf_path)
            repaired_text = fix_mojibake(text)
            normalized_text = normalize_whitespace(repaired_text)
            sections = segment_sections(normalized_text)

            record["text"] = repaired_text
            record["text_length"] = len(repaired_text)
            record["normalized_text"] = normalized_text
            record["normalized_text_length"] = len(normalized_text)
            record["sections"] = sections
            record["warnings"] = []
            if sections["missing_sections"]:
                record["warnings"].append(
                    "Secoes nao encontradas: "
                    + ", ".join(sections["missing_sections"])
                )

            record["text_output_path"] = str(save_text_artifact(output_dir, pdf_path, repaired_text))
            record["normalized_text_output_path"] = str(
                save_text_artifact(output_dir, pdf_path.with_stem(f"{pdf_path.stem}_normalized"), normalized_text)
            )
            record["sections_output_path"] = str(
                save_json_artifact(
                    output_dir,
                    "segmentation",
                    f"{pdf_path.stem}_sections.json",
                    sections,
                )
            )
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
        documents.append(record)
    return documents


def write_extraction_summary(output_dir: Path, documents: list[dict[str, object]]) -> Path:
    summary_path = output_dir / "extraction_summary.json"
    payload = {
        "document_count": len(documents),
        "processed_count": sum(1 for doc in documents if doc["status"] == "ok"),
        "error_count": sum(1 for doc in documents if doc["status"] == "error"),
        "documents": documents,
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    documents = run_extraction(args.input_dir, args.output_dir)
    summary_path = write_extraction_summary(args.output_dir, documents)
    alignment_path = run_alignment(documents, args.output_dir)

    print("Etapas de ingestao, pre-processamento, segmentacao e alinhamento concluidas.")
    print(f"Entrada: {args.input_dir}")
    print(f"Saida: {args.output_dir}")
    print(f"Documentos encontrados: {len(documents)}")
    print(f"Resumo salvo em: {summary_path}")
    print(f"Resultados de alinhamento salvos em: {alignment_path}")


if __name__ == "__main__":
    main()
