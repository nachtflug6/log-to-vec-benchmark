"""
Convert comprehensive_report.md → comprehensive_report.pdf via pdflatex.
Run from the repo root: python3 reports/build_pdf.py
"""
import re, os, subprocess, sys, textwrap

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
MD_FILE    = os.path.join(REPORT_DIR, "comprehensive_report.md")
TEX_FILE   = os.path.join(REPORT_DIR, "comprehensive_report.tex")
PDF_FILE   = os.path.join(REPORT_DIR, "comprehensive_report.pdf")
FIG_DIR    = os.path.join(REPORT_DIR, "figures")


# ── helpers ───────────────────────────────────────────────────────────────────

def escape_latex(s):
    """Escape LaTeX special chars, but preserve already-escaped commands."""
    # order matters
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&",  "\\&")
    s = s.replace("%",  "\\%")
    s = s.replace("$",  "\\$")
    s = s.replace("#",  "\\#")
    s = s.replace("_",  "\\_")
    s = s.replace("{",  "\\{")
    s = s.replace("}",  "\\}")
    s = s.replace("~",  "\\textasciitilde{}")
    s = s.replace("^",  "\\textasciicircum{}")
    return s


def inline_format(s):
    """Apply inline markdown formatting after escaping."""
    s = escape_latex(s)
    # bold **...**
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    # italic *...*
    s = re.sub(r"\*(.+?)\*", r"\\textit{\1}", s)
    # inline code `...` — restore underscores (already escaped above) back to \_ inside \texttt
    s = re.sub(r"`([^`]+)`", lambda m: "\\texttt{" + m.group(1) + "}", s)
    # math / special unicode
    s = s.replace("±", "$\\pm$")
    s = s.replace("−", "$-$")          # U+2212 minus sign
    s = s.replace("≈", "$\\approx$")
    s = s.replace("≤", "$\\leq$")
    s = s.replace("≥", "$\\geq$")
    s = s.replace("σ", "$\\sigma$")
    s = s.replace("→", "$\\rightarrow$")
    s = s.replace("←", "$\\leftarrow$")
    s = s.replace("²", "$^2$")
    # arrows for metrics
    s = s.replace("↑", "$\\uparrow$")
    s = s.replace("↓", "$\\downarrow$")
    # dashes
    s = s.replace("—", "---")
    s = s.replace("–", "--")
    return s


def convert_table(lines):
    """Convert a markdown table block into a LaTeX longtable."""
    rows = []
    for line in lines:
        line = line.strip().strip("|")
        cells = [c.strip() for c in line.split("|")]
        rows.append(cells)

    if len(rows) < 2:
        return ""

    # row 0: headers, row 1: separator, rest: data
    headers = rows[0]
    data    = rows[2:]
    ncols   = len(headers)
    col_spec = "|" + "l|" * ncols

    out = []
    out.append(f"\\begin{{longtable}}{{{col_spec}}}")
    out.append("\\hline")

    header_cells = " & ".join(f"\\textbf{{{inline_format(h)}}}" for h in headers)
    out.append(header_cells + " \\\\")
    out.append("\\hline")
    out.append("\\endfirsthead")

    out.append("\\hline")
    out.append(header_cells + " \\\\")
    out.append("\\hline")
    out.append("\\endhead")

    out.append("\\hline")
    out.append("\\endfoot")

    for row in data:
        # pad/trim to ncols
        while len(row) < ncols:
            row.append("")
        row = row[:ncols]
        cells = " & ".join(inline_format(c) for c in row)
        out.append(cells + " \\\\")
        out.append("\\hline")

    out.append("\\end{longtable}")
    return "\n".join(out)


def figure_latex(alt, path):
    """Convert markdown image reference to LaTeX figure."""
    # resolve path relative to reports/figures/
    fname = os.path.basename(path)
    fig_path = os.path.join(FIG_DIR, fname)
    if not os.path.exists(fig_path):
        return f"% [missing figure: {fname}]"
    rel = os.path.relpath(fig_path, REPORT_DIR).replace("\\", "/")
    alt_tex = escape_latex(alt)
    return (
        "\\begin{figure}[H]\n"
        "\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{{rel}}}\n"
        f"\\caption{{{alt_tex}}}\n"
        "\\end{figure}"
    )


# ── main converter ────────────────────────────────────────────────────────────

def md_to_latex(md_text):
    lines  = md_text.splitlines()
    output = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # ── blank line
        if not line.strip():
            output.append("")
            i += 1
            continue

        # ── ATX headings
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            title = inline_format(m.group(2))
            cmds  = ["section", "subsection", "subsubsection", "paragraph"]
            cmd   = cmds[min(level - 1, 3)]
            if level <= 3:
                output.append(f"\\{cmd}{{{title}}}")
            else:
                output.append(f"\\{cmd}{{{title}}}\\mbox{{}}")
            i += 1
            continue

        # ── horizontal rule
        if re.match(r"^---+\s*$", line):
            output.append("\\vspace{4pt}\\hrule\\vspace{4pt}")
            i += 1
            continue

        # ── fenced code block ```
        if line.strip().startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            code = "\n".join(code_lines)
            output.append("\\begin{lstlisting}")
            output.append(code)
            output.append("\\end{lstlisting}")
            continue

        # ── markdown table
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s\-\|:]+\|\s*$", lines[i + 1]):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            output.append(convert_table(table_lines))
            continue

        # ── image ![alt](path)
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line.strip())
        if m:
            output.append(figure_latex(m.group(1), m.group(2)))
            i += 1
            continue

        # ── unordered list
        if re.match(r"^(\s*)-\s+", line):
            output.append("\\begin{itemize}")
            while i < len(lines) and re.match(r"^(\s*)[-*]\s+", lines[i]):
                item = re.sub(r"^\s*[-*]\s+", "", lines[i])
                output.append(f"  \\item {inline_format(item)}")
                i += 1
            output.append("\\end{itemize}")
            continue

        # ── ordered list
        if re.match(r"^\d+\.\s+", line):
            output.append("\\begin{enumerate}")
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i]):
                item = re.sub(r"^\d+\.\s+", "", lines[i])
                output.append(f"  \\item {inline_format(item)}")
                i += 1
            output.append("\\end{enumerate}")
            continue

        # ── normal paragraph line
        output.append(inline_format(line))
        i += 1

    return "\n".join(output)


# ── LaTeX document wrapper ─────────────────────────────────────────────────────

PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[a4paper, margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{float}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{parskip}
\usepackage{microtype}
\usepackage{titlesec}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!60!black,
    urlcolor=blue!60!black,
}

\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    backgroundcolor=\color{gray!10},
    frame=single,
    framesep=4pt,
    xleftmargin=4pt,
    xrightmargin=4pt,
}

\titleformat{\section}{\large\bfseries}{}{0em}{\thesection\quad}
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{\thesubsection\quad}
\titlespacing{\section}{0pt}{16pt}{6pt}
\titlespacing{\subsection}{0pt}{10pt}{4pt}

\begin{document}
"""

POSTAMBLE = r"""
\end{document}
"""


def build_tex(md_text):
    body = md_to_latex(md_text)
    return PREAMBLE.strip() + "\n\n" + body + "\n\n" + POSTAMBLE.strip()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Reading {MD_FILE}")
    with open(MD_FILE, encoding="utf-8") as f:
        md = f.read()

    tex = build_tex(md)

    print(f"Writing {TEX_FILE}")
    with open(TEX_FILE, "w", encoding="utf-8") as f:
        f.write(tex)

    print("Running pdflatex (pass 1)...")
    cmd = ["pdflatex", "-interaction=nonstopmode", "-output-directory", REPORT_DIR,
           TEX_FILE]
    r = subprocess.run(cmd, capture_output=True, cwd=REPORT_DIR)
    r.stdout = r.stdout.decode("latin-1")
    log_lines = r.stdout.splitlines()
    hard_errors = [l for l in log_lines if l.startswith("! ")]
    if hard_errors and not os.path.exists(PDF_FILE):
        print("pdflatex hard errors:")
        for e in hard_errors[:20]:
            print(" ", e)
        print("Last 20 lines of log:")
        for l in log_lines[-20:]:
            print(" ", l)
        sys.exit(1)
    if hard_errors:
        print(f"  ({len(hard_errors)} minor LaTeX warnings, PDF still produced)")

    print("Running pdflatex (pass 2 — resolve refs)...")
    subprocess.run(cmd, capture_output=True, cwd=REPORT_DIR)

    if os.path.exists(PDF_FILE):
        size = os.path.getsize(PDF_FILE) // 1024
        print(f"\nDone: {PDF_FILE}  ({size} KB)")
    else:
        print("ERROR: PDF not produced.")
        sys.exit(1)
