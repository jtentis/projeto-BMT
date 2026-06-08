from __future__ import annotations

import argparse
import csv
import json
import math
import re
import traceback
import unicodedata
from pathlib import Path
from typing import Callable, TypedDict

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
    method: str
    status: str
    promise_section: str
    promise_doco_section: str
    delivery_sections: list[str]
    delivery_doco_sections: list[str]
    compared_pairs: list[dict[str, object]]
    alignment_score: float | None
    alignment_label: str
    notes: list[str]


class DatasetDocument(TypedDict):
    document_id: str
    pdf_path: str
    status: str
    page_count: int | None
    text_length: int
    normalized_text_length: int
    available_sections: list[str]
    missing_sections: list[str]
    warning_count: int


LEGACY_TO_DOCO = {
    "introduction": "doco:Introduction",
    "results": "doco:Results",
    "conclusion": "doco:Conclusion",
}

SCIBERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"


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
    parser.add_argument(
        "--method",
        choices=("tfidf", "scibert", "all"),
        default="tfidf",
        help="Metodo de similaridade a executar.",
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


def count_pdf_pages(pdf_path: Path) -> int:
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


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

def repair_text_encoding(text: str) -> str:
    if not any(marker in text for marker in ("Ã", "â", "ï", "ð")):
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text

    suspicious_chars = ("Ã", "â", "ï", "ð")
    original_noise = sum(text.count(char) for char in suspicious_chars)
    repaired_noise = sum(repaired.count(char) for char in suspicious_chars)
    if repaired_noise < original_noise:
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


def is_table_of_contents_line(raw_line: str) -> bool:
    stripped = raw_line.strip()
    return bool(
        re.search(r"\.{4,}\s*\d+\s*$", stripped)
        or re.search(r"(?:\.\s*){4,}\d+\s*$", stripped)
        or (stripped.count(".") >= 4 and re.search(r"\d+\s*$", stripped))
    )


def is_reference_heading(normalized_line: str) -> bool:
    return any(
        term in normalized_line
        for term in (
            "referencias",
            "references",
            "bibliografia",
            "apendices",
            "anexos",
        )
    )


def classify_heading_role(normalized_title: str) -> str | None:
    intro_patterns = [r"\bintrodu", r"\bintroduction\b"]
    conclusion_patterns = [
        r"\bconclus",
        r"\bconsideracoes finais\b",
        r"\bfinal considerations\b",
        r"\bconclusion\b",
    ]
    delivery_patterns = [
        r"\bresult",
        r"\bdiscuss",
        r"\bavaliac",
        r"\bexperimen",
        r"\banalis",
        r"\bestudo de caso\b",
        r"\baplicac",
        r"\bexemplos?\b",
        r"\bparte pratica\b",
        r"\bdesenvolv",
        r"\bimplementac",
        r"\bproposta\b",
        r"\bmodelo\b",
        r"\bferramenta\b",
        r"\bsistema\b",
        r"\bsimulador\b",
        r"\bmanual\b",
    ]

    if any(re.search(pattern, normalized_title) for pattern in intro_patterns):
        return "introduction"
    if any(re.search(pattern, normalized_title) for pattern in conclusion_patterns):
        return "conclusion"
    if any(re.search(pattern, normalized_title) for pattern in delivery_patterns):
        return "results"
    return None


def classify_doco_role(normalized_title: str) -> str:
    intro_patterns = [r"\bintrodu", r"\bintroduction\b"]
    conclusion_patterns = [
        r"\bconclus",
        r"\bconsideracoes finais\b",
        r"\bfinal considerations\b",
        r"\bconclusion\b",
    ]
    method_patterns = [
        r"\bmetod",
        r"\bmethod",
        r"\bmateriais e metodos\b",
        r"\bprocedimentos\b",
        r"\bprocesso de desenvolvimento\b",
    ]
    related_work_patterns = [
        r"\btrabalhos relacionados\b",
        r"\brelated work\b",
        r"\breferencial teorico\b",
        r"\breferencias teoricas\b",
        r"\bfundamentacao teorica\b",
        r"\bconceitos basicos\b",
        r"\brevisao bibliografica\b",
    ]
    discussion_patterns = [
        r"\bdiscuss",
        r"\bavaliac",
        r"\banalis",
        r"\banalise dos resultados\b",
        r"\banalise e discuss",
        r"\bavaliacao dos resultados\b",
    ]
    results_patterns = [
        r"\bresult",
        r"\bexperimen",
        r"\bestudo de caso\b",
        r"\baplicac",
        r"\bexemplos?\b",
        r"\bparte pratica\b",
        r"\bdesenvolv",
        r"\bimplementac",
        r"\bproposta\b",
        r"\bmodelo\b",
        r"\bferramenta\b",
        r"\bsistema\b",
        r"\bsimulador\b",
        r"\bmanual\b",
    ]

    if any(re.search(pattern, normalized_title) for pattern in intro_patterns):
        return "doco:Introduction"
    if any(re.search(pattern, normalized_title) for pattern in conclusion_patterns):
        return "doco:Conclusion"
    if any(re.search(pattern, normalized_title) for pattern in related_work_patterns):
        return "doco:RelatedWork"
    if any(re.search(pattern, normalized_title) for pattern in method_patterns):
        return "doco:Methods"
    if any(re.search(pattern, normalized_title) for pattern in discussion_patterns):
        return "doco:Discussion"
    if any(re.search(pattern, normalized_title) for pattern in results_patterns):
        return "doco:Results"
    return "doco:Other"


def heading_level(numbering: str) -> int:
    return numbering.count(".") + 1


def next_section_boundary(
    lines: list[str],
    headings: list[dict[str, object]],
    current_index: int,
) -> int:
    current_heading = headings[current_index]
    current_line = int(current_heading["line_index"])
    current_level = int(current_heading["level"])

    for candidate in headings[current_index + 1 :]:
        candidate_line = int(candidate["line_index"])
        candidate_level = int(candidate["level"])
        if candidate_line > current_line and candidate_level <= current_level:
            return candidate_line

    for line_index in range(current_line + 1, len(lines)):
        normalized_line = normalize_for_matching(lines[line_index])
        if is_reference_heading(normalized_line):
            return line_index

    return len(lines)


def segment_sections(normalized_text: str) -> SegmentationResult:
    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    headings: list[dict[str, object]] = []

    for line_index, raw_line in enumerate(lines):
        if raw_line.isdigit() or is_table_of_contents_line(raw_line):
            continue

        normalized_line = normalize_for_matching(raw_line)
        match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$", normalized_line)
        if not match:
            continue

        numbering = match.group(1)
        title = match.group(2).strip(" .")
        headings.append(
            {
                "line_index": line_index,
                "numbering": numbering,
                "level": heading_level(numbering),
                "header": raw_line,
                "normalized_title": title,
                "doco_role": classify_doco_role(title),
                "role": classify_heading_role(title),
            }
        )

    intro_idx = next(
        (idx for idx, heading in enumerate(headings) if heading["role"] == "introduction"),
        None,
    )
    conclusion_idx = next(
        (idx for idx in range(len(headings) - 1, -1, -1) if headings[idx]["role"] == "conclusion"),
        None,
    )

    results_idx = None
    if intro_idx is not None:
        search_end = conclusion_idx if conclusion_idx is not None else len(headings)
        for idx in range(intro_idx + 1, search_end):
            if headings[idx]["role"] == "results":
                results_idx = idx
                break

        if results_idx is None:
            excluded_titles = (
                "referencias teoricas",
                "fundamentacao teorica",
                "referencial teorico",
                "motivacao",
                "objetivo",
                "estrutura do texto",
                "estrutura do trabalho",
            )
            fallback_candidates = []
            for idx in range(intro_idx + 1, search_end):
                heading = headings[idx]
                title = str(heading["normalized_title"])
                if int(heading["level"]) != 1:
                    continue
                if any(title.startswith(excluded) for excluded in excluded_titles):
                    continue
                fallback_candidates.append(idx)
            if fallback_candidates:
                results_idx = fallback_candidates[-1]

    selected_indices = {
        "introduction": intro_idx,
        "results": results_idx,
        "conclusion": conclusion_idx,
    }

    sections: dict[str, SectionData] = {}
    for section_name, heading_idx in selected_indices.items():
        if heading_idx is None:
            continue

        heading = headings[heading_idx]
        start_line = int(heading["line_index"])
        end_line = next_section_boundary(lines, headings, heading_idx)
        content = " ".join(lines[start_line + 1 : end_line]).strip()
        sections[section_name] = {
            "header": str(heading["header"]),
            "start_char": start_line,
            "end_char": end_line,
            "content": content,
            "content_length": len(content),
        }

    doco_sections: dict[str, list[dict[str, object]]] = {}
    for heading_idx, heading in enumerate(headings):
        doco_role = str(heading["doco_role"])
        if doco_role == "doco:Other":
            continue
        start_line = int(heading["line_index"])
        end_line = next_section_boundary(lines, headings, heading_idx)
        content = " ".join(lines[start_line + 1 : end_line]).strip()
        doco_sections.setdefault(doco_role, []).append(
            {
                "header": str(heading["header"]),
                "legacy_role": heading["role"],
                "start_char": start_line,
                "end_char": end_line,
                "content": content,
                "content_length": len(content),
            }
        )

    result = {
        "available_sections": sorted(sections.keys()),
        "missing_sections": [
            section_name
            for section_name in ("introduction", "results", "conclusion")
            if section_name not in sections
        ],
        "sections": sections,
        "available_doco_labels": sorted(doco_sections.keys()),
        "missing_doco_labels": [
            label
            for label in (
                "doco:Introduction",
                "doco:Results",
                "doco:Discussion",
                "doco:Conclusion",
            )
            if label not in doco_sections
        ],
        "doco_sections": doco_sections,
    }
    return result


def save_json_artifact(output_dir: Path, relative_dir: str, name: str, payload: object) -> Path:
    artifact_dir = output_dir / relative_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifact_dir / name
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def section_presence(document: dict[str, object], section_name: str) -> bool:
    if document["status"] != "ok":
        return False
    sections = document.get("sections")
    if not isinstance(sections, dict):
        return False
    section_map = sections.get("sections")
    if not isinstance(section_map, dict):
        return False
    return section_name in section_map


def build_dataset_documents(documents: list[dict[str, object]]) -> list[DatasetDocument]:
    dataset_documents: list[DatasetDocument] = []
    for document in documents:
        sections = document.get("sections")
        available_sections: list[str] = []
        missing_sections: list[str] = []
        if isinstance(sections, dict):
            raw_available = sections.get("available_sections", [])
            raw_missing = sections.get("missing_sections", [])
            if isinstance(raw_available, list):
                available_sections = [str(section) for section in raw_available]
            if isinstance(raw_missing, list):
                missing_sections = [str(section) for section in raw_missing]

        warnings = document.get("warnings", [])
        warning_count = len(warnings) if isinstance(warnings, list) else 0
        page_count = document.get("page_count")
        dataset_documents.append(
            {
                "document_id": str(document["document_id"]),
                "pdf_path": str(document["pdf_path"]),
                "status": str(document["status"]),
                "page_count": page_count if isinstance(page_count, int) else None,
                "text_length": int(document.get("text_length", 0)),
                "normalized_text_length": int(document.get("normalized_text_length", 0)),
                "available_sections": available_sections,
                "missing_sections": missing_sections,
                "warning_count": warning_count,
            }
        )
    return dataset_documents


def write_dataset_summary(output_dir: Path, documents: list[dict[str, object]]) -> Path:
    dataset_documents = build_dataset_documents(documents)
    processed_documents = [doc for doc in dataset_documents if doc["status"] == "ok"]
    text_lengths = [doc["normalized_text_length"] for doc in processed_documents]
    page_counts = [doc["page_count"] for doc in processed_documents if doc["page_count"] is not None]
    section_counts = {
        section_name: sum(1 for document in documents if section_presence(document, section_name))
        for section_name in ("introduction", "results", "conclusion")
    }
    status_distribution: dict[str, int] = {}
    for document in dataset_documents:
        status_distribution[document["status"]] = status_distribution.get(document["status"], 0) + 1

    payload = {
        "dataset_name": "Corpus final consolidado BMT",
        "source": "Trabalhos academicos em PDF coletados do Pantheon/UFRJ e versionados no repositorio.",
        "inclusion_criteria": [
            "PDF academico completo",
            "texto extraivel com pypdf",
            "documento relacionado a computacao ou jogos digitais",
        ],
        "exclusion_criteria": [
            "arquivos duplicados",
            "PDFs sem texto extraivel",
            "documentos fora do formato de trabalho academico",
        ],
        "cleaning_and_normalization": [
            "extracao textual com pypdf",
            "reparo heuristico de codificacao quando detectado",
            "remocao de hifenizacao artificial",
            "normalizacao de espacos",
            "normalizacao sem acentos para busca de cabecalhos",
        ],
        "document_count": len(dataset_documents),
        "processed_count": len(processed_documents),
        "error_count": len(dataset_documents) - len(processed_documents),
        "page_count_total": sum(page_counts),
        "normalized_text_length_total": sum(text_lengths),
        "normalized_text_length_mean": round(sum(text_lengths) / len(text_lengths), 2) if text_lengths else 0,
        "section_detection_counts": section_counts,
        "status_distribution": status_distribution,
        "documents": dataset_documents,
    }
    return save_json_artifact(output_dir, "dataset", "summary.json", payload)


def write_experimental_protocol(output_dir: Path) -> Path:
    payload = {
        "protocol_name": "Avaliacao descritiva reproduzivel do baseline BMT",
        "data_split": "Corpus fixo completo usado como conjunto de avaliacao.",
        "training_data": "Nao aplicavel nesta entrega; nao ha treinamento supervisionado.",
        "validation_data": "Nao aplicavel nesta entrega; limiares do baseline sao fixos e documentados.",
        "test_data": "Os 10 PDFs consolidados em data/raw.",
        "leakage_control": [
            "O baseline nao usa rotulos manuais para ajustar parametros.",
            "O TF-IDF e ajustado somente nos dois trechos comparados dentro de cada documento.",
            "Nao ha transferencia de vocabulario, pesos ou estatisticas entre documentos.",
        ],
        "cross_validation": "Nao usada porque o corpus e pequeno e o objetivo atual e validar o desenho experimental antes de evoluir modelos.",
        "random_seeds": [],
        "rounds": 1,
        "baselines_compared": ["TF-IDF com similaridade do cosseno entre introducao e secoes de entrega"],
    }
    return save_json_artifact(output_dir, "protocol", "experimental_protocol.json", payload)


def compute_similarity(text_a: str, text_b: str) -> float:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([text_a, text_b])
    similarity_matrix = cosine_similarity(matrix, matrix)
    score = similarity_matrix[0, 1]
    return float(score)


def cosine_from_vectors(vector_a: list[float], vector_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def build_scibert_similarity() -> Callable[[str, str], float]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Dependencias do SciBERT ausentes. Execute pip install -r requirements.txt."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(SCIBERT_MODEL_NAME)
    model.eval()

    def embed(text: str) -> list[float]:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            output = model(**encoded)
        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked_embeddings = token_embeddings * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        embedding = summed / counts
        return embedding.squeeze(0).tolist()

    def compute(text_a: str, text_b: str) -> float:
        return cosine_from_vectors(embed(text_a), embed(text_b))

    return compute


def build_similarity_function(method: str) -> Callable[[str, str], float]:
    if method == "tfidf":
        return compute_similarity
    if method == "scibert":
        return build_scibert_similarity()
    raise ValueError(f"Metodo desconhecido: {method}")


def label_alignment(score: float | None) -> str:
    if score is None:
        return "insufficient_sections"
    if score >= 0.35:
        return "high"
    if score >= 0.15:
        return "medium"
    return "low"


def write_alignment_artifacts(
    output_dir: Path,
    results: list[AlignmentResult],
    json_name: str,
    write_legacy_metrics: bool,
) -> Path:
    json_path = save_json_artifact(output_dir, "alignment", json_name, {"documents": results})

    if write_legacy_metrics:
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


def run_alignment_for_method(
    documents: list[dict[str, object]],
    method: str,
) -> list[AlignmentResult]:
    similarity_function = build_similarity_function(method)
    results: list[AlignmentResult] = []

    for document in documents:
        document_id = str(document["document_id"])
        if document["status"] != "ok":
            results.append(
                {
                    "document_id": document_id,
                    "method": method,
                    "status": "error",
                    "promise_section": "introduction",
                    "promise_doco_section": LEGACY_TO_DOCO["introduction"],
                    "delivery_sections": [],
                    "delivery_doco_sections": [],
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

            score = similarity_function(introduction["content"], delivery_section["content"])
            compared_pairs.append(
                {
                    "method": method,
                    "promise_section": "introduction",
                    "promise_doco_section": LEGACY_TO_DOCO["introduction"],
                    "delivery_section": delivery_name,
                    "delivery_doco_section": LEGACY_TO_DOCO[delivery_name],
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
            "method": method,
            "status": "ok",
            "promise_section": "introduction",
            "promise_doco_section": LEGACY_TO_DOCO["introduction"],
            "delivery_sections": [str(pair["delivery_section"]) for pair in compared_pairs],
            "delivery_doco_sections": [
                str(pair["delivery_doco_section"]) for pair in compared_pairs
            ],
            "compared_pairs": compared_pairs,
            "alignment_score": alignment_score,
            "alignment_label": label_alignment(alignment_score),
            "notes": notes,
        }
        results.append(result)

    return results


def write_method_comparison(
    output_dir: Path,
    tfidf_results: list[AlignmentResult],
    scibert_results: list[AlignmentResult],
) -> Path:
    tfidf_by_doc = {result["document_id"]: result for result in tfidf_results}
    scibert_by_doc = {result["document_id"]: result for result in scibert_results}
    output_path = output_dir / "alignment" / "comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "document_id",
                "tfidf_score",
                "tfidf_label",
                "scibert_score",
                "scibert_label",
                "score_delta",
            ],
        )
        writer.writeheader()
        for document_id in sorted(tfidf_by_doc):
            tfidf_result = tfidf_by_doc[document_id]
            scibert_result = scibert_by_doc[document_id]
            tfidf_score = tfidf_result["alignment_score"]
            scibert_score = scibert_result["alignment_score"]
            score_delta = ""
            if isinstance(tfidf_score, float) and isinstance(scibert_score, float):
                score_delta = f"{scibert_score - tfidf_score:.4f}"
            writer.writerow(
                {
                    "document_id": document_id,
                    "tfidf_score": "" if tfidf_score is None else f"{tfidf_score:.4f}",
                    "tfidf_label": tfidf_result["alignment_label"],
                    "scibert_score": "" if scibert_score is None else f"{scibert_score:.4f}",
                    "scibert_label": scibert_result["alignment_label"],
                    "score_delta": score_delta,
                }
            )
    return output_path


def run_alignment(
    documents: list[dict[str, object]],
    output_dir: Path,
    method: str,
) -> tuple[Path, list[AlignmentResult]]:
    if method == "tfidf":
        tfidf_results = run_alignment_for_method(documents, "tfidf")
        tfidf_path = write_alignment_artifacts(
            output_dir,
            tfidf_results,
            "tfidf_results.json",
            write_legacy_metrics=True,
        )
        write_alignment_artifacts(
            output_dir,
            tfidf_results,
            "results.json",
            write_legacy_metrics=False,
        )
        return tfidf_path, tfidf_results

    if method == "scibert":
        scibert_results = run_alignment_for_method(documents, "scibert")
        scibert_path = write_alignment_artifacts(
            output_dir,
            scibert_results,
            "scibert_results.json",
            write_legacy_metrics=False,
        )
        return scibert_path, scibert_results

    tfidf_results = run_alignment_for_method(documents, "tfidf")
    scibert_results = run_alignment_for_method(documents, "scibert")
    tfidf_path = write_alignment_artifacts(
        output_dir,
        tfidf_results,
        "tfidf_results.json",
        write_legacy_metrics=True,
    )
    write_alignment_artifacts(
        output_dir,
        tfidf_results,
        "results.json",
        write_legacy_metrics=False,
    )
    write_alignment_artifacts(
        output_dir,
        scibert_results,
        "scibert_results.json",
        write_legacy_metrics=False,
    )
    write_method_comparison(output_dir, tfidf_results, scibert_results)
    return tfidf_path, tfidf_results


def write_coverage_metrics(
    output_dir: Path,
    documents: list[dict[str, object]],
    results: list[AlignmentResult],
) -> Path:
    label_distribution: dict[str, int] = {}
    for result in results:
        label = result["alignment_label"]
        label_distribution[label] = label_distribution.get(label, 0) + 1

    processed_count = sum(1 for document in documents if document["status"] == "ok")
    document_count = len(documents)
    evaluable_count = sum(1 for result in results if result["alignment_score"] is not None)
    metrics = {
        "document_count": document_count,
        "processed_count": processed_count,
        "error_count": document_count - processed_count,
        "extraction_success_rate": round(processed_count / document_count, 4) if document_count else 0,
        "has_introduction_count": sum(1 for document in documents if section_presence(document, "introduction")),
        "has_results_count": sum(1 for document in documents if section_presence(document, "results")),
        "has_conclusion_count": sum(1 for document in documents if section_presence(document, "conclusion")),
        "evaluable_count": evaluable_count,
        "evaluable_rate": round(evaluable_count / document_count, 4) if document_count else 0,
        "alignment_label_distribution": label_distribution,
        "alignment_metrics": [
            "tfidf_cosine_similarity",
            "best_introduction_to_delivery_score",
            "alignment_label",
        ],
        "future_bri_metrics": [
            "Precision@k",
            "Recall@k",
            "MAP",
            "NDCG",
        ],
        "future_mt_metrics": [
            "accuracy",
            "F1",
            "macro-F1",
            "RMSE",
        ],
    }
    json_path = save_json_artifact(output_dir, "metrics", "coverage_metrics.json", metrics)

    csv_path = output_dir / "metrics" / "coverage_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": json.dumps(value, ensure_ascii=False)})

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
            page_count = count_pdf_pages(pdf_path)
            text = extract_text_from_pdf(pdf_path)
            repaired_text = repair_text_encoding(fix_mojibake(text))
            normalized_text = normalize_whitespace(repaired_text)
            sections = segment_sections(normalized_text)

            record["page_count"] = page_count
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
            record["doco_sections_output_path"] = str(
                save_json_artifact(
                    output_dir,
                    "doco_sections",
                    f"{pdf_path.stem}_doco_sections.json",
                    {
                        "document_id": pdf_path.stem,
                        "available_doco_labels": sections["available_doco_labels"],
                        "missing_doco_labels": sections["missing_doco_labels"],
                        "doco_sections": sections["doco_sections"],
                    },
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
    dataset_summary_path = write_dataset_summary(args.output_dir, documents)
    protocol_path = write_experimental_protocol(args.output_dir)
    alignment_path, alignment_results = run_alignment(documents, args.output_dir, args.method)
    coverage_metrics_path = write_coverage_metrics(args.output_dir, documents, alignment_results)

    print("Etapas de ingestao, pre-processamento, segmentacao e alinhamento concluidas.")
    print(f"Entrada: {args.input_dir}")
    print(f"Saida: {args.output_dir}")
    print(f"Metodo: {args.method}")
    print(f"Documentos encontrados: {len(documents)}")
    print(f"Resumo salvo em: {summary_path}")
    print(f"Resumo do dataset salvo em: {dataset_summary_path}")
    print(f"Protocolo experimental salvo em: {protocol_path}")
    print(f"Resultados de alinhamento salvos em: {alignment_path}")
    print(f"Metricas de cobertura salvas em: {coverage_metrics_path}")


if __name__ == "__main__":
    main()
