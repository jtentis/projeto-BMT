from __future__ import annotations

import argparse
import json
import re
import traceback
import unicodedata
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader


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

    print("Etapas de ingestao, pre-processamento e segmentacao concluidas.")
    print(f"Entrada: {args.input_dir}")
    print(f"Saida: {args.output_dir}")
    print(f"Documentos encontrados: {len(documents)}")
    print(f"Resumo salvo em: {summary_path}")


if __name__ == "__main__":
    main()
