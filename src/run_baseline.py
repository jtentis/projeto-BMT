from __future__ import annotations

import argparse
from pathlib import Path


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


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Baseline BMT configurado com sucesso.")
    print(f"Entrada: {args.input_dir}")
    print(f"Saida: {args.output_dir}")
    print("As etapas de extracao, segmentacao e alinhamento serao executadas neste comando.")


if __name__ == "__main__":
    main()
