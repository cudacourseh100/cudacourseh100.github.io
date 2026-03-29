#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
except ImportError as exc:  # pragma: no cover - friendly failure path
    raise SystemExit("beautifulsoup4 is required to run scripts/export-lesson-markdown.py") from exc


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_ROOT = ROOT / "markdown"
LESSONS_ROOT = MARKDOWN_ROOT / "lessons"
SITE_BASE_URL = "https://cudacourseh100.github.io"
STATIC_MARKDOWN_DOCS = [
    ("seo/seo-strategy.md", "SEO strategy"),
    ("seo/indexing-checklist.md", "SEO indexing checklist"),
]


@dataclass(frozen=True)
class LessonSource:
    number: str
    slug: str
    html_path: str


LESSONS = [
    LessonSource("1", "introduction-to-h100s", "pages/lesson-1.html"),
    LessonSource("2", "clusters-data-types-inline-ptx-pointers", "pages/lesson-2.html"),
    LessonSource("3", "asynchronicity-and-barriers", "pages/lesson-3.html"),
    LessonSource("4", "cutensormap", "pages/lesson-4.html"),
    LessonSource("5", "cp-async-bulk", "pages/lesson-5.html"),
    LessonSource("6", "wgmma-part-1", "pages/lesson-6.html"),
    LessonSource("7", "wgmma-part-2", "pages/lesson-7.html"),
    LessonSource("8", "kernel-design", "pages/lesson-8.html"),
    LessonSource("8.1", "stream-k", "pages/lesson-8.1.html"),
    LessonSource("9", "multi-gpu-part-1", "pages/lesson-9.html"),
    LessonSource("10", "multi-gpu-part-2", "pages/lesson-10.html"),
]


TEXT_REPLACEMENTS = {
    "\u00a0": " ",
    "\u2009": " ",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2192": "->",
    "\u00d7": "x",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2260": "!=",
    "\ufeff": "",
}


def ascii_sanitize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    for source, target in TEXT_REPLACEMENTS.items():
        value = value.replace(source, target)
    value = value.encode("ascii", "ignore").decode("ascii")
    return value


def normalize_inline(text: str) -> str:
    value = ascii_sanitize(text).replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:?!])", r"\1", value)
    value = re.sub(r"\(\s+", "(", value)
    value = re.sub(r"\s+\)", ")", value)
    value = re.sub(r"\[\s+", "[", value)
    value = re.sub(r"\s+\]", "]", value)
    return value.strip()


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def repo_relative_path(base_file: str, href: str) -> str:
    resolved = (ROOT / base_file).parent.joinpath(href).resolve()
    return resolved.relative_to(ROOT.resolve()).as_posix()


def render_inline_node(node: NavigableString | Tag) -> str:
    if isinstance(node, NavigableString):
        return ascii_sanitize(str(node))

    if not isinstance(node, Tag):
        return ""

    if node.name == "code":
        text = normalize_inline(node.get_text(" ", strip=True))
        return f"`{text}`" if text else ""

    if node.name in {"strong", "b"}:
        text = render_inline_children(node)
        return f"**{text}**" if text else ""

    if node.name in {"em", "i"}:
        text = render_inline_children(node)
        return f"*{text}*" if text else ""

    if node.name == "a":
        text = render_inline_children(node) or normalize_inline(node.get_text(" ", strip=True))
        href = ascii_sanitize(node.get("href", "")).strip()
        return f"[{text}]({href})" if href else text

    if node.name == "br":
        return "  \n"

    return render_inline_children(node)


def render_inline_children(node: Tag) -> str:
    return normalize_inline("".join(render_inline_node(child) for child in node.children))


def render_list(tag: Tag, ordered: bool) -> str:
    lines: list[str] = []
    items = tag.find_all("li", recursive=False)
    for index, item in enumerate(items, start=1):
        prefix = f"{index}. " if ordered else "- "
        content = render_inline_children(item)
        if content:
            lines.append(f"{prefix}{content}")
    return "\n".join(lines)


def render_pre(tag: Tag) -> str:
    code = tag.find("code")
    text = code.get_text("\n", strip=False) if code else tag.get_text("\n", strip=False)
    text = ascii_sanitize(text).strip("\n")
    return f"```text\n{text}\n```" if text else ""


def render_note(tag: Tag) -> str:
    parts = [render_block(child) for child in tag.children]
    content = "\n\n".join(part for part in parts if part.strip())
    if not content:
        return ""
    return "\n".join("> " + line if line else ">" for line in content.splitlines())


def render_fact_list(tag: Tag) -> str:
    lines: list[str] = []
    for row in tag.find_all("div", recursive=False):
        dt = row.find("dt")
        dd = row.find("dd")
        if not dt or not dd:
            continue
        term = render_inline_children(dt)
        detail = render_inline_children(dd)
        lines.append(f"- **{term}:** {detail}")
    return "\n".join(lines)


def render_table(tag: Tag) -> str:
    rows = tag.find_all("tr")
    if not rows:
        return ""

    parsed_rows: list[list[str]] = []
    first_cells_are_headers = True
    for row in rows:
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        parsed_rows.append([render_inline_children(cell).replace("|", "\\|") for cell in cells])
        if cells[0].name != "th":
            first_cells_are_headers = False

    if not parsed_rows:
        return ""

    header_row: list[str]
    body_rows: list[list[str]]
    if tag.find("thead"):
        header_row = parsed_rows[0]
        body_rows = parsed_rows[1:]
    elif len(parsed_rows[0]) == 2 and first_cells_are_headers:
        header_row = ["Term", "Definition"]
        body_rows = parsed_rows
    else:
        header_row = parsed_rows[0]
        body_rows = parsed_rows[1:]

    width = len(header_row)
    normalized_body = [row + [""] * (width - len(row)) for row in body_rows]
    normalized_header = header_row + [""] * (width - len(header_row))

    lines = [
        "| " + " | ".join(normalized_header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in normalized_body:
        lines.append("| " + " | ".join(row[:width]) + " |")
    return "\n".join(lines)


def render_card(tag: Tag, heading_level: int) -> str:
    title = tag.find(["h3", "h4"])
    parts: list[str] = []
    if title:
        parts.append(f"{'#' * min(heading_level, 6)} {render_inline_children(title)}")

    for child in tag.children:
        if child == title:
            continue
        rendered = render_block(child)
        if rendered:
            parts.append(rendered)
    return "\n\n".join(parts)


def render_children(tag: Tag) -> str:
    parts = [render_block(child) for child in tag.children]
    return "\n\n".join(part for part in parts if part.strip())


def render_block(node: NavigableString | Tag) -> str:
    if isinstance(node, NavigableString):
        return ""

    if not isinstance(node, Tag):
        return ""

    classes = set(node.get("class", []))

    if node.name in {"section", "article"}:
        return render_children(node)

    if node.name == "div" and "lesson-note" in classes:
        return render_note(node)

    if node.name == "div" and "lesson-card-grid" in classes:
        cards = [
            render_card(card, 4)
            for card in node.find_all("article", class_="lesson-card", recursive=False)
        ]
        return "\n\n".join(card for card in cards if card.strip())

    if node.name == "div" and "lesson-facts" in classes:
        return render_fact_list(node)

    if node.name in {"div", "aside"}:
        return render_children(node)

    if node.name in {"h2", "h3", "h4"}:
        base = int(node.name[1])
        return f"{'#' * min(base + 1, 6)} {render_inline_children(node)}"

    if node.name == "p":
        return render_inline_children(node)

    if node.name == "ul":
        return render_list(node, ordered=False)

    if node.name == "ol":
        return render_list(node, ordered=True)

    if node.name == "pre":
        return render_pre(node)

    if node.name == "table":
        return render_table(node)

    if node.name == "dl":
        return render_fact_list(node)

    return ""


def normalize_slide_lines(page_text: str) -> list[str]:
    raw_lines = [ascii_sanitize(line.rstrip()) for line in page_text.splitlines()]
    lines: list[str] = []
    for raw_line in raw_lines:
        line = re.sub(r"^\s*-\s+", "- ", raw_line)
        line = re.sub(r"^\s+\.", "", line)
        line = line.rstrip()
        if not line.strip():
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line.lstrip())

    collapsed: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line == "-" and index + 1 < len(lines):
            next_line = lines[index + 1].lstrip("- ").strip()
            collapsed.append(f"- {next_line}")
            index += 2
            continue
        if re.fullmatch(r"\d+\.", line) and index + 1 < len(lines):
            collapsed.append(f"{line} {lines[index + 1].strip()}")
            index += 2
            continue
        collapsed.append(line)
        index += 1

    while collapsed and collapsed[0] == "":
        collapsed.pop(0)
    while collapsed and collapsed[-1] == "":
        collapsed.pop()
    return collapsed


def extract_slide_markdown(pdf_path: str) -> tuple[int, str]:
    pdf_file = ROOT / pdf_path
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_file), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    raw_pages = [page for page in result.stdout.split("\f") if page.strip()]
    sections: list[str] = []
    for index, page in enumerate(raw_pages, start=1):
        lines = normalize_slide_lines(page)
        if not lines:
            continue
        heading = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if body:
            sections.append(f"### Slide {index}: {heading}\n\n{body}")
        else:
            sections.append(f"### Slide {index}: {heading}")
    return len(raw_pages), "\n\n".join(sections)


def frontmatter_list(key: str, values: list[str]) -> str:
    if not values:
        return f"{key}: []"
    lines = [f"{key}:"]
    lines.extend(f"  - {json.dumps(value)}" for value in values)
    return "\n".join(lines)


def number_token(number: str) -> str:
    if "." in number:
        major, minor = number.split(".", maxsplit=1)
        return f"{int(major):02d}-{minor}"
    return f"{int(number):02d}"


def build_markdown(lesson: LessonSource) -> tuple[str, str]:
    html_file = ROOT / lesson.html_path
    soup = BeautifulSoup(html_file.read_text(encoding="utf8"), "html.parser")

    title = render_inline_children(soup.select_one(".lesson-hero-title"))
    summary = render_inline_children(soup.select_one(".lesson-hero-summary"))
    chips = dedupe([render_inline_children(tag) for tag in soup.select(".lesson-chip")])
    meta_heading = render_inline_children(soup.select_one(".lesson-meta-card h2"))
    meta_paragraph = render_inline_children(soup.select_one(".lesson-meta-card > p:not(.lesson-meta-eyebrow)"))

    facts_block = soup.select_one(".lesson-facts")
    facts = render_fact_list(facts_block) if facts_block else ""

    takeaway_list = ""
    for card in soup.select(".lesson-sidebar-card"):
        kicker = render_inline_children(card.select_one(".section-kicker")) if card.select_one(".section-kicker") else ""
        if kicker == "Key Takeaways":
            items = [render_inline_children(li) for li in card.select("li")]
            takeaway_list = "\n".join(f"- {item}" for item in items if item)
            break

    source_slide_href = soup.select_one(".hero-actions a.button-primary")["href"]
    slide_pdf = repo_relative_path(lesson.html_path, source_slide_href)
    published_page = "/" + Path(lesson.html_path).as_posix()
    file_token = number_token(lesson.number)
    published_markdown = f"/markdown/lessons/lesson-{file_token}-{lesson.slug}.md"

    code_refs = dedupe(
        re.findall(r"[A-Za-z0-9_./-]+\.(?:cu|cuh|hpp|inl|md)", facts)
    )

    article = soup.select_one("article.lesson-article")
    article_markdown = render_children(article) if article else ""

    slide_count, slide_markdown = extract_slide_markdown(slide_pdf)

    frontmatter = "\n".join(
        [
            "---",
            f"title: {json.dumps(f'Lesson {lesson.number} - {title}')}",
            f"lesson_number: {json.dumps(lesson.number)}",
            f"lesson_slug: {json.dumps(lesson.slug)}",
            f"instructor: {json.dumps('Prateek Shukla')}",
            f"course: {json.dumps('CUDA Programming for NVIDIA H100s')}",
            f"source_page: {json.dumps(lesson.html_path)}",
            f"source_slide_pdf: {json.dumps(slide_pdf)}",
            f"published_lesson_page: {json.dumps(published_page)}",
            f"published_markdown_path: {json.dumps(published_markdown)}",
            frontmatter_list("topics", chips),
            frontmatter_list("code_refs", code_refs),
            frontmatter_list("generated_from", [lesson.html_path, slide_pdf]),
            "---",
        ]
    )

    sources_section = "\n".join(
        [
            f"- Lesson page: `{lesson.html_path}`",
            f"- Slide deck: `{slide_pdf}`",
            f"- Published lesson URL: `{SITE_BASE_URL}{published_page}`",
            f"- Published markdown URL: `{SITE_BASE_URL}{published_markdown}`",
        ]
    )

    sections = [
        frontmatter,
        f"# Lesson {lesson.number} - {title}",
        "This file combines the published lesson page with full slide-deck text so agents can fetch and search it directly.",
        "## Sources",
        sources_section,
        "## Lesson Summary",
        summary,
        "## Why This Lesson Matters",
        f"**{meta_heading}**",
        meta_paragraph,
        "## Topics",
        "\n".join(f"- {chip}" for chip in chips),
    ]

    if facts:
        sections.extend(["## Lesson Facts", facts])

    if takeaway_list:
        sections.extend(["## Key Takeaways", takeaway_list])

    if article_markdown:
        sections.extend(["## Web Lesson Content", article_markdown])

    sections.extend(
        [
            "## Full Slide Deck Text",
            f"Extracted from `{slide_pdf}` with `pdftotext -layout`. Total slides: {slide_count}.",
            slide_markdown,
        ]
    )

    markdown = "\n\n".join(section for section in sections if section.strip()).strip() + "\n"
    filename = f"lesson-{file_token}-{lesson.slug}.md"
    return filename, markdown


def write_index(entries: list[tuple[LessonSource, str, str]]) -> None:
    lines = [
        "# AI-Fetchable Lesson Markdown",
        "",
        "These files combine the published lesson pages with full extracted slide text so an AI agent can pull searchable lesson content with `curl` or `wget`.",
        "",
        "## Fetch Examples",
        "",
        "```sh",
        f"curl -L {SITE_BASE_URL}/markdown/lessons/lesson-01-introduction-to-h100s.md",
        f"wget -O lesson-06-wgmma-part-1.md {SITE_BASE_URL}/markdown/lessons/lesson-06-wgmma-part-1.md",
        "```",
        "",
        "## Files",
        "",
    ]

    for lesson, filename, title in entries:
        lines.append(f"- `lessons/{filename}` - Lesson {lesson.number}: {title}")

    available_static_docs = [
        (path, label) for path, label in STATIC_MARKDOWN_DOCS if (MARKDOWN_ROOT / path).exists()
    ]
    if available_static_docs:
        lines.extend(["", "## Strategy Docs", ""])
        for path, label in available_static_docs:
            lines.append(f"- `{path}` - {label}")

    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "```sh",
            "python3 scripts/export-lesson-markdown.py",
            "```",
            "",
        ]
    )

    (MARKDOWN_ROOT / "README.md").write_text("\n".join(lines), encoding="utf8")


def main() -> None:
    LESSONS_ROOT.mkdir(parents=True, exist_ok=True)
    for existing_file in LESSONS_ROOT.glob("lesson-*.md"):
        existing_file.unlink()

    index_entries: list[tuple[LessonSource, str, str]] = []
    for lesson in LESSONS:
        filename, markdown = build_markdown(lesson)
        (LESSONS_ROOT / filename).write_text(markdown, encoding="utf8")
        title_match = re.search(r"^# Lesson [^\n]+ - (.+)$", markdown, re.MULTILINE)
        title = title_match.group(1) if title_match else lesson.slug
        index_entries.append((lesson, filename, title))
        print(f"Wrote markdown/lessons/{filename}")

    write_index(index_entries)
    print("Wrote markdown/README.md")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        command = " ".join(exc.cmd) if isinstance(exc.cmd, list) else str(exc.cmd)
        raise SystemExit(f"Command failed: {command}") from exc
