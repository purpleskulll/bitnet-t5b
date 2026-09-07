#!/usr/bin/env python3
r"""Generate paper/arxiv.md from paper/bitnet-baremetal-t5b.tex.

WHY THIS EXISTS
---------------
The paper was maintained as two hand-written files, and they drifted four times.
Three of those a reviewer caught, and on the third occasion the .tex -- the file
that gets submitted -- was missing the paper's most important section entirely.
Their advice, given three times, was to stop maintaining two: *generate the
Markdown from the LaTeX, or drop it, and never touch it by hand.*

Two gates were built instead (numeric-claim parity and path resolution). They
caught drift; they did not stop it, because two hand-maintained files remain two
hand-maintained files. This removes the second one: `arxiv.md` is now OUTPUT.

`pandoc` is not available on this host and installing packages was declined, so
this is purpose-built for the LaTeX this document actually uses -- nine
environments and about thirty macros, inventoried rather than guessed. It is not
a general converter and does not pretend to be.

HOW IT BEHAVES ON SOMETHING IT DOES NOT KNOW, stated precisely because an
earlier version of this paragraph claimed it "fails loudly" and that was too
strong. Three things raise and stop: an unbalanced brace group, an siunitx unit
macro not in the table, and a \ref to a label that does not exist. Everything
else -- an unknown macro, an unknown environment -- is passed through VERBATIM.
That is deliberate and it is what maths depends on: \Sigma, \times, \sqrt and
\frac must reach the Markdown untouched, and a converter that raised on them
would be useless here.

The consequence is that an unknown construct does not stop the run; it appears
in the output and `--check` reports the difference. That is a weaker guarantee
than an exception and it is the honest one: nothing is silently DROPPED, which
is the failure that would matter.

    tools/tex_to_md.py            regenerate paper/arxiv.md
    tools/tex_to_md.py --check    regenerate into memory and diff; exit 1 on
                                  any difference. This is the gate: it fails
                                  when someone edits arxiv.md by hand.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper/bitnet-baremetal-t5b.tex"
MD = ROOT / "paper/arxiv.md"


# --------------------------------------------------------------- numbering ---
def number_sections(tex: str):
    """Map every \\label to the number LaTeX would assign it.

    The Markdown has to write "§7.6" where the LaTeX writes \\ref{sec:tq10}, so
    the counters are replayed here. Equations are counted document-wide, which
    is what the article class does.
    """
    labels, sec, sub, eq = {}, 0, 0, 0
    pending = None
    for m in re.finditer(
            r"\\(section|subsection)\*?\{|\\begin\{equation\}|\\label\{([^}]*)\}",
            tex):
        tok = m.group(0)
        if tok.startswith(r"\section*"):
            pending = None            # starred: unnumbered, and never referenced
        elif tok.startswith(r"\section"):
            sec += 1; sub = 0; pending = f"{sec}"
        elif tok.startswith(r"\subsection*"):
            pending = None
        elif tok.startswith(r"\subsection"):
            sub += 1; pending = f"{sec}.{sub}"
        elif tok.startswith(r"\begin{equation}"):
            eq += 1; pending = f"{eq}"
        elif m.group(2) is not None and pending is not None:
            labels[m.group(2)] = pending
    return labels


# ------------------------------------------------------------------ macros ---
def si_unit(u: str) -> str:
    """siunitx unit macros, spelled out. Only those this document uses."""
    parts, out = re.findall(r"\\([a-zA-Z]+)", u), []
    # Every unit macro this document uses, collected from the .tex rather than
    # guessed. An unknown one RAISES: silently emitting a number without its
    # unit is the kind of quiet wrong that this whole file exists to stop, and
    # the first run of the fixed converter duly stopped on \mega.
    MAP = {"percent": "%", "milli": "m", "micro": "µ", "nano": "n",
           "giga": "G", "mega": "M", "kibi": "Ki", "mebi": "Mi", "gibi": "Gi",
           "second": "s", "byte": "B", "hertz": "Hz", "per": "/"}
    for p in parts:
        if p not in MAP:
            raise ValueError(f"unknown siunitx unit macro: \\{p}")
        out.append(MAP[p])
    s = "".join(out)
    return s if s == "%" else " " + s


def group(text: str, i: int):
    """Contents of the brace group starting at text[i] == '{', and the index
    after its close. Brace-counting, not a regular expression: these groups
    nest, and a non-greedy match would end at the first inner brace."""
    assert text[i] == "{"
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{" and (j == i or text[j-1] != "\\"):
            depth += 1
        elif text[j] == "}" and text[j-1] != "\\":
            depth -= 1
            if depth == 0:
                return text[i+1:j], j + 1
        j += 1
    raise ValueError("unbalanced brace group")


def expand(text: str, labels: dict, math: bool = False) -> str:
    """One pass of macro expansion, innermost-first via repeated application."""
    def num(s):
        s = s.strip()
        if re.fullmatch(r"-?\d+", s) and len(s.lstrip("-")) > 4:
            return f"{int(s):,}"
        return s

    out, i = [], 0
    while i < len(text):
        m = re.compile(r"\\([a-zA-Z]+)\*?").match(text, i)
        if not m:
            out.append(text[i]); i += 1; continue
        name, j = m.group(1), m.end()

        def arg():
            nonlocal j
            while j < len(text) and text[j] in " \n":
                j += 1
            if j < len(text) and text[j] == "{":
                g, j2 = group(text, j); j = j2
                return g
            return ""

        if name == "code":
            body = arg()
            if math:
                # Inside $$ ... $$ a backtick is literal text, not code: the
                # first version emitted \mathrm{PPL}_{`i2_s`} and broke the
                # formula. \texttt is what a maths renderer understands.
                out.append("\\texttt{" + body + "}")
            else:
                out.append("`" + body.replace("\\_", "_").replace("\\#", "#")
                                     .replace("\\%", "%").replace("\\&", "&") + "`")
        elif name == "num":
            out.append(num(arg()))
        elif name == "numrange":
            a, b = num(arg()), num(arg()); out.append(f"{a}–{b}")
        elif name == "SI":
            v, u = num(arg()), arg(); out.append(v + si_unit(u))
        elif name == "SIrange":
            a, b, u = num(arg()), num(arg()), arg()
            out.append(f"{a}–{b}{si_unit(u)}")
        elif name in ("textbf", "emph", "textit"):
            body = expand(arg(), labels, math)
            out.append(("**" if name == "textbf" else "*") + body
                       + ("**" if name == "textbf" else "*"))
        elif name in ("ref", "eqref"):
            key = arg()
            if key not in labels:
                raise ValueError(f"reference to unknown label: {key}")
            out.append(labels[key] if name == "ref" else f"({labels[key]})")
        elif name == "cite":
            out.append("[" + arg() + "]")
        elif name == "label":
            arg()
        elif name == "S":
            out.append("§")
        elif name in ("percent",):
            out.append("%")
        elif name in ("allowbreak", "sloppy", "small", "toprule", "midrule",
                      "bottomrule", "centering", "noindent", "protect",
                      "arraybackslash", "raggedright"):
            pass
        elif name == "paragraph":
            out.append("**" + expand(arg(), labels, math).rstrip(".") + ".** ")
        else:
            # Maths and anything else stays verbatim: the Markdown renders it
            # with the same LaTeX the .tex uses, so \Sigma, \times, \sqrt and
            # friends need no translation and must not be mangled.
            out.append(m.group(0))
        i = j
    return "".join(out)


# -------------------------------------------------------------- structures ---
def convert_tabular(body: str, labels: dict) -> str:
    rows = []
    for line in body.split("\\\\"):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        line = re.sub(r"\\(top|mid|bottom)rule", "", line).strip()
        if not line:
            continue
        cells = [expand(c.strip(), labels) for c in line.split("&")]
        rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "|" + "---|" * width]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def convert(tex: str) -> str:
    labels = number_sections(tex)
    body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}")[0]

    # Title block and abstract, which the Markdown carries as front matter.
    title = re.search(r"\\title\{(.*?)\}\s*\n", tex, re.S)
    out = []
    if title:
        # The pieces are joined with "" at the end, so every line the front
        # matter needs must carry its own newline. Without them the first run
        # produced "---title: ...abstract: |" on one line, which is not YAML.
        t = " ".join(expand(title.group(1).replace("\\\\", " "), labels).split())
        out.append("---\n")
        out.append("title: " + t + "\n")
        ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
        if ab:
            out.append("abstract: |\n")
            for para in [q.strip() for q in ab.group(1).strip().split("\n\n") if q.strip()]:
                out.append("  " + expand(para, labels).replace("\n", "\n  ") + "\n")
        out.append("---\n")
        body = body[body.find(r"\end{abstract}") + len(r"\end{abstract}"):] \
            if r"\end{abstract}" in body else body

    sec = sub = 0
    i = 0
    # Plain paragraph text is BUFFERED and expanded when a structural element
    # ends the run. The first version appended it character by character and
    # never expanded it, so 114 \code, 106 \num and 36 \ref survived raw into
    # the Markdown -- and the numeric-parity check did not see it, because that
    # check resolves raw \num{} in the Markdown too. Structure was converted and
    # prose was not.
    plain = []

    def flush():
        if plain:
            out.append(expand("".join(plain), labels))
            plain.clear()

    while i < len(body):
        # Section headings
        m = re.compile(r"\\(section|subsection)(\*?)\{").match(body, i)
        if m:
            flush()
            g, j = group(body, m.end() - 1)
            starred = m.group(2) == "*"
            if m.group(1) == "section":
                if not starred:
                    sec += 1; sub = 0
                out.append(f"\n---\n\n# {'' if starred else str(sec) + '. '}"
                           + expand(g, labels).strip() + "\n")
            else:
                if not starred:
                    sub += 1
                out.append(f"\n## {'' if starred else f'{sec}.{sub} '}"
                           + expand(g, labels).strip() + "\n")
            i = j; continue

        # Environments
        m = re.compile(r"\\begin\{(equation|center|tabular|enumerate|proposition"
                       r"|proof|thebibliography)\}").match(body, i)
        if m:
            flush()
            env = m.group(1)
            end = f"\\end{{{env}}}"
            k = body.find(end, m.end())
            if k < 0:
                raise ValueError(f"unclosed environment: {env}")
            inner = body[m.end():k]
            if env == "equation":
                # expand() too: an equation can hold \num{1.58496} and
                # \code{vpmaddubsw}, which no Markdown maths renderer knows.
                # It is safe because expand() passes anything it does not handle
                # through verbatim, so \log, \sum, \frac and \text survive.
                inner = re.sub(r"\\label\{[^}]*\}", "", inner).strip()
                out.append("\n$$\n" + expand(inner, labels, math=True) + "\n$$\n")
            elif env == "tabular":
                spec, rest = group(inner, inner.index("{"))
                out.append("\n" + convert_tabular(inner[inner.index("{") + len(spec) + 2:],
                                                  labels) + "\n")
            elif env in ("center", "proof", "proposition"):
                lead = {"proof": "\n*Proof.* ", "proposition": "\n**Proposition.** "}
                out.append(lead.get(env, ""))
                out.append(convert(f"\\begin{{document}}{inner}\\end{{document}}")
                           if False else convert_inner(inner, labels))
            elif env == "enumerate":
                for item in [x.strip() for x in inner.split(r"\item") if x.strip()]:
                    out.append("\n" + str(len(out)) + ". " if False else "")
                items = [x.strip() for x in inner.split(r"\item") if x.strip()]
                out.append("\n" + "\n".join(
                    f"{n}. " + expand(t, labels).replace("\n", "\n   ")
                    for n, t in enumerate(items, 1)) + "\n")
            elif env == "thebibliography":
                out.append("\n# References\n")
                for b in re.finditer(r"\\bibitem\{[^}]*\}(.*?)(?=\\bibitem|\Z)",
                                     inner, re.S):
                    # Each on its own line: `out` is joined with "", so an
                    # entry without a trailing newline runs into the next and
                    # thirteen references become one.
                    out.append("- " + " ".join(expand(b.group(1), labels).split())
                               + "\n")
            i = k + len(end); continue

        # Display maths written \[ ... \]. Markdown renderers know $$, not
        # \[, so an unconverted one shows as literal brackets around the
        # formula -- which is how it first shipped.
        if body.startswith("\\[", i):
            k = body.find("\\]", i)
            if k < 0:
                raise ValueError("unclosed display maths \\[")
            flush()
            out.append("\n$$\n" + expand(body[i+2:k], labels, math=True).strip()
                       + "\n$$\n")
            i = k + 2; continue

        plain.append(body[i]); i += 1
    flush()

    text = "".join(out)
    # Paragraph whitespace: collapse the runs LaTeX ignores.
    # LaTeX's non-breaking space. It is punctuation in TeX and a literal tilde
    # in Markdown, where GitHub also reads a doubled one as strikethrough.
    text = text.replace("~", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def convert_inner(inner: str, labels: dict) -> str:
    r"""A center, proposition or proof body.

    It must handle EQUATIONS as well as tabulars: a proposition here states its
    claim in a nested equation environment, and the first version of this
    function knew only tables, so that equation reached the Markdown as literal
    \begin{equation}. Display maths written \[ ... \] is converted too --
    Markdown renderers know $$, not \[."""
    out, i = [], 0
    while i < len(inner):
        m = re.compile(r"\\begin\{(tabular|equation)\}").match(inner, i)
        if m:
            env = m.group(1)
            end = f"\\end{{{env}}}"
            k = inner.find(end, m.end())
            if k < 0:
                raise ValueError(f"unclosed {env} inside a nested environment")
            seg = inner[m.end():k]
            if env == "tabular":
                spec, _ = group(seg, seg.index("{"))
                out.append("\n" + convert_tabular(
                    seg[seg.index("{") + len(spec) + 2:], labels) + "\n")
            else:
                seg = re.sub(r"\\label\{[^}]*\}", "", seg).strip()
                out.append("\n$$\n" + expand(seg, labels, math=True) + "\n$$\n")
            i = k + len(end); continue
        if inner.startswith("\\[", i):
            k = inner.find("\\]", i)
            if k < 0:
                raise ValueError("unclosed display maths \\[")
            out.append("\n$$\n" + expand(inner[i+2:k], labels, math=True).strip() + "\n$$\n")
            i = k + 2; continue
        out.append(inner[i]); i += 1
    return expand("".join(out), labels)


def main() -> int:
    if not TEX.exists():
        print(f"missing: {TEX}", file=sys.stderr); return 2
    try:
        got = convert(TEX.read_text())
    except ValueError as e:
        print(f"tex_to_md: {e}", file=sys.stderr)
        print("  The converter refuses to guess. Teach it the construct, or", file=sys.stderr)
        print("  change the .tex to one it knows.", file=sys.stderr)
        return 2

    if "--check" in sys.argv:
        have = MD.read_text() if MD.exists() else ""
        if have == got:
            print("PAPER SOURCE: arxiv.md is the generated form of the .tex.")
            return 0
        print("PAPER SOURCE: arxiv.md differs from what the .tex generates.")
        import difflib
        d = list(difflib.unified_diff(have.splitlines(), got.splitlines(),
                                      "arxiv.md", "generated", lineterm=""))
        for line in d[:40]:
            print("  " + line)
        if len(d) > 40:
            print(f"  ... and {len(d) - 40} more lines")
        print("\nFAIL -- arxiv.md is GENERATED and was edited by hand, or the")
        print("  .tex changed without regenerating. Run tools/tex_to_md.py.")
        return 1

    MD.write_text(got)
    print(f"wrote {MD} ({len(got.splitlines())} lines) from {TEX.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
