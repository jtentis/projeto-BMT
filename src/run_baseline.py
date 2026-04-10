from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from pypdf import PdfReader


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
            record["text"] = text
            record["text_length"] = len(text)
            record["text_output_path"] = str(save_text_artifact(output_dir, pdf_path, text))
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

    print("Etapa de ingestao e extracao concluida.")
    print(f"Entrada: {args.input_dir}")
    print(f"Saida: {args.output_dir}")
    print(f"Documentos encontrados: {len(documents)}")
    print(f"Resumo salvo em: {summary_path}")


if __name__ == "__main__":
    main()
