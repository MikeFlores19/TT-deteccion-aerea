"""
Genera el PDF de la revisión de mejoras del TT a partir del documento Markdown.

Convierte revision_6puntos.md -> revision_6puntos.pdf usando python-markdown
(para parsear el Markdown a HTML) y fpdf2 (para renderizar el PDF). Se usan las
fuentes DejaVu del sistema para soportar acentos y símbolos del español.

Uso:
    python3 docs/revision_mejoras/build_pdf.py
"""

from pathlib import Path
from datetime import date

import markdown
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "revision_6puntos.md"
PDF_PATH = ROOT / "revision_6puntos.pdf"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


class Documento(FPDF):
    """PDF A4 con pie de página numerado."""

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"TT2026-2_IA05  ·  Revisión de mejoras  ·  pág. {self.page_no()}",
                  align="C")
        self.set_text_color(0, 0, 0)


def registrar_fuentes(pdf: FPDF) -> None:
    """Registra la familia DejaVu (regular, bold, italic, bold-italic)."""
    pdf.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVu", "I", str(FONT_DIR / "DejaVuSans-Oblique.ttf"))
    pdf.add_font("DejaVu", "BI", str(FONT_DIR / "DejaVuSans-BoldOblique.ttf"))


def portada(pdf: FPDF) -> None:
    """Página de portada simple y centrada."""
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("DejaVu", "B", 22)
    pdf.multi_cell(0, 12, "Revisión de mejoras\ndel Trabajo Terminal", align="C")
    pdf.ln(8)
    pdf.set_font("DejaVu", "", 13)
    pdf.multi_cell(
        0, 8,
        "TT2026-2_IA05\nSistema de detección y seguimiento de objetos\n"
        "desde perspectiva aérea con UAV",
        align="C",
    )
    pdf.ln(16)
    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(
        0, 7,
        "Miguel Alejandro Flores Sotelo\nSergio de Jesús Castillo Molano",
        align="C",
    )
    pdf.ln(10)
    pdf.set_font("DejaVu", "I", 10)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 7, f"Generado el {date.today().isoformat()}", align="C")
    pdf.set_text_color(0, 0, 0)


def cuerpo(pdf: FPDF, md_text: str) -> None:
    """Renderiza el cuerpo del documento desde Markdown.

    Se omite el primer bloque (título y datos de portada del .md, hasta el
    primer separador horizontal) porque ya se muestra en la portada del PDF.
    """
    partes = md_text.split("\n---\n", 1)
    contenido = partes[1] if len(partes) == 2 else md_text
    html = markdown.markdown(
        contenido,
        extensions=["tables", "sane_lists", "fenced_code"],
    )
    pdf.add_page()
    pdf.set_font("DejaVu", "", 11)
    pdf.write_html(html)


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    pdf = Documento(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(20, 18, 20)
    registrar_fuentes(pdf)
    portada(pdf)
    cuerpo(pdf, md_text)
    pdf.output(str(PDF_PATH))
    print(f"PDF generado: {PDF_PATH}")


if __name__ == "__main__":
    main()
