"""Create a full-scale printable chessboard for the LINGYAN calibration workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib.pagesizes import A3, A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


PAPERS = {"A4": landscape(A4), "A3": landscape(A3)}


def draw_board(output: Path, paper: str, square_mm: float, inner_cols: int, inner_rows: int) -> None:
    if square_mm <= 0 or inner_cols < 2 or inner_rows < 2:
        raise ValueError("square size must be positive and inner-corner counts must be at least 2")
    page_width, page_height = PAPERS[paper]
    columns, rows = inner_cols + 1, inner_rows + 1
    square = square_mm * mm
    board_width, board_height = columns * square, rows * square
    if board_width > page_width - 30 * mm or board_height > page_height - 50 * mm:
        raise ValueError("board does not fit with print margins; use smaller squares or A3")

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=(page_width, page_height), pageCompression=0)
    pdf.setTitle(f"{inner_cols}x{inner_rows} inner corners, {square_mm:g} mm squares")
    left = (page_width - board_width) / 2
    bottom = (page_height - board_height) / 2 + 3 * mm

    pdf.setFillColorRGB(0, 0, 0)
    for row in range(rows):
        for column in range(columns):
            if (row + column) % 2 == 0:
                x = left + column * square
                y = bottom + (rows - row - 1) * square
                pdf.rect(x, y, square, square, fill=1, stroke=0)

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawCentredString(
        page_width / 2,
        bottom + board_height + 12 * mm,
        f"Camera calibration: {inner_cols} x {inner_rows} inner corners, {square_mm:g} mm squares",
    )
    bar_width = 100 * mm
    bar_x = (page_width - bar_width) / 2
    bar_y = 19 * mm
    pdf.setLineWidth(0.7)
    pdf.line(bar_x, bar_y, bar_x + bar_width, bar_y)
    pdf.line(bar_x, bar_y - 2 * mm, bar_x, bar_y + 2 * mm)
    pdf.line(bar_x + bar_width, bar_y - 2 * mm, bar_x + bar_width, bar_y + 2 * mm)
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(page_width / 2, 11 * mm, "100 mm reference - print at 100% / actual size")
    pdf.showPage()
    pdf.save()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a print-ready calibration chessboard PDF.")
    parser.add_argument("--paper", choices=PAPERS, default="A4")
    parser.add_argument("--square-mm", type=float, default=20.0)
    parser.add_argument("--inner-cols", type=int, default=9)
    parser.add_argument("--inner-rows", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path("output/pdf") / (
        f"lingyan_chessboard_{args.inner_cols}x{args.inner_rows}_"
        f"{args.paper}_{args.square_mm:g}mm.pdf"
    )
    draw_board(output, args.paper, args.square_mm, args.inner_cols, args.inner_rows)
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
