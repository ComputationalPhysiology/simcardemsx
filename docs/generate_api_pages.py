#!/usr/bin/env python3
"""Write the API reference pages from the package's own source.

The pages under `docs/api/` are generated at build time rather than written by hand, so the
reference is always the code in the repository. One page per module, mirroring the package,
because that is the structure a reader already has in their imports:

    python3 docs/generate_api_pages.py docs/api

Adapted from sknm's generator (github.com/ComputationalPhysiology/sknm). The docstrings here
are written with Sphinx roles and reST literals, which MyST would show verbatim, so every text
chunk passes through :func:`_rst_to_markdown` first.

`griffe` reads the source without importing it, so generating the reference needs none of
the package's runtime dependencies and cannot be affected by import side effects.

Requires the `docs` extra:  pip install -e ".[docs]"
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import griffe

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "simcardemsx"

# What a page documents. Attributes are included because a module can be almost nothing else.
DOCUMENTED_KINDS = ("class", "function", "attribute")
ATTRIBUTE_HEADERS = ("Name", "Type", "Value", "Description")


def load(name: str = PACKAGE, search_paths: Sequence[Path] | None = None) -> Any:
    """Read a package from source, with its docstrings parsed as numpydoc.

    Parameters
    ----------
    name : str, optional
        The package or module to read. Defaults to `simcardemsx`.
    search_paths : sequence of Path, optional
        Where to look for it. Defaults to this repository's `src/`.

    Returns
    -------
    griffe.Module
        The module, with submembers and parsed docstrings attached.
    """
    paths = [ROOT / "src"] if search_paths is None else list(search_paths)
    return griffe.load(name, search_paths=[str(path) for path in paths], docstring_parser="numpy")


def documented_members(module: Any) -> list[Any]:
    """The members a module's page documents, in the order they are written in.

    A member belongs to the module that defines it, not to any module that imports it, so
    imported names are left out: they are documented on their own page. A module that
    declares `__all__` narrows the list further to what it names.

    Parameters
    ----------
    module : griffe.Module
        The module to list.

    Returns
    -------
    list of griffe.Object
        Its classes, functions and attributes.
    """
    declared = set(module.exports) if module.exports is not None else None
    members = [
        member
        for name, member in module.members.items()
        if not member.is_alias
        and member.kind.value in DOCUMENTED_KINDS
        and not name.startswith("_")
        and (declared is None or name in declared)
    ]
    return sorted(members, key=lambda member: member.lineno or 0)


def _summary(module: Any) -> str:
    """A module's first docstring line, which a list of modules shows beside its name."""
    if module.docstring is None:
        return ""
    return _rst_to_markdown(module.docstring.value.strip().splitlines()[0])


def submodules(module: Any) -> list[Any]:
    """The modules directly beneath one, in alphabetical order.

    Parameters
    ----------
    module : griffe.Module
        The module to look under.

    Returns
    -------
    list of griffe.Module
        Its submodules, which may be empty.
    """
    found = [
        member
        for member in module.members.values()
        if not member.is_alias and member.kind.value == "module"
    ]
    return sorted(found, key=lambda child: child.name)


def modules(package: Any) -> Iterator[Any]:
    """Walk a package and its subpackages, parents before children.

    Parameters
    ----------
    package : griffe.Module
        The package to walk.

    Yields
    ------
    griffe.Module
        Every module in it, including the package itself.
    """
    yield package
    for child in submodules(package):
        yield from modules(child)


_ROLE = re.compile(r":(?:py:)?(?:class|func|meth|attr|mod|data|exc|obj|const):`~?([^`]+)`")
_LITERAL = re.compile(r"``([^`]+)``")
_MATH_ROLE = re.compile(r":math:`([^`]+)`")


def _rst_to_markdown(text: str) -> str:
    """Rewrite the reST a docstring uses into the MyST markdown a page is written in.

    Cross-reference roles become code spans (a `~` prefix keeps only the last name, as in
    Sphinx), double-backtick literals become single ones, `:math:` becomes `$...$`, and a
    `.. math::` directive becomes a `$$` block.
    """

    def role(match: re.Match[str]) -> str:
        target = match.group(1)
        shown = target.rsplit(".", 1)[-1] if match.group(0).count("~") else target
        return f"`{shown}`"

    text = _MATH_ROLE.sub(lambda m: f"${m.group(1)}$", text)
    text = _ROLE.sub(role, text)
    text = _LITERAL.sub(lambda m: f"`{m.group(1)}`", text)

    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == ".. math::":
            indent = len(line) - len(line.lstrip())
            i += 1
            block: list[str] = []
            while i < len(lines) and (
                not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent
            ):
                block.append(lines[i].strip())
                i += 1
            while block and not block[-1]:
                block.pop()
            body = [entry for entry in block if entry]
            out += ["$$", *body, "$$", ""]
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _table(rows: Sequence[tuple[str, ...]], headers: tuple[str, ...]) -> list[str]:
    """Render rows as a markdown table, with cell contents made safe for one.

    A column no row fills is dropped, so a table of undocumented, unannotated names does not
    carry two empty columns down its whole length.
    """

    def cell(text: str) -> str:
        return " ".join(text.split()).replace("|", "\\|")

    kept = [index for index in range(len(headers)) if any(row[index].strip() for row in rows)]
    lines = ["| " + " | ".join(headers[index] for index in kept) + " |"]
    lines += ["|" + "|".join([" --- "] * len(kept)) + "|"]
    lines += ["| " + " | ".join(cell(row[index]) for index in kept) + " |" for row in rows]
    return lines


def _definition(term: str, description: str) -> list[str]:
    """Render one entry of a markdown definition list, indenting what runs on."""
    body = _rst_to_markdown(description).strip().splitlines() or [""]
    return [term, f": {body[0]}", *[f"  {line}" for line in body[1:]], ""]


def _annotated(name: str, annotation: Any) -> str:
    """A definition term naming something and, where it is known, its type."""
    if annotation is None:
        return f"`{name}`" if name else ""
    return f"`{name}` : *{annotation}*" if name else f"*{annotation}*"


def _sections(docstring: Any) -> list[str]:
    """Render a parsed numpydoc docstring as markdown."""
    if docstring is None:
        return []
    lines: list[str] = []
    for section in docstring.parsed:
        kind = section.kind.value
        if kind == "text":
            lines += [_rst_to_markdown(section.value), ""]
        elif kind in ("parameters", "other_parameters", "attributes"):
            lines += [f"**{kind.replace('_', ' ').title()}**", ""]
            for item in section.value:
                lines += _definition(_annotated(item.name, item.annotation), item.description)
        elif kind in ("returns", "yields"):
            lines += [f"**{kind.title()}**", ""]
            for item in section.value:
                lines += _definition(_annotated(item.name or "", item.annotation), item.description)
        elif kind == "raises":
            lines += ["**Raises**", ""]
            for item in section.value:
                lines += _definition(f"`{item.annotation}`", item.description)
        elif kind == "examples":
            lines += ["**Examples**", ""]
            for example_kind, text in section.value:
                if example_kind.value == "examples":
                    lines += ["```pycon", text, "```", ""]
                else:
                    lines += [text, ""]
        elif kind == "admonition":
            lines += [f"**{section.title}**", "", _rst_to_markdown(section.value.description), ""]
    return lines


def _signature(member: Any) -> str | None:
    """How the thing is called, or nothing for something that is not called.

    A class that writes no `__init__`, such as a protocol or an exception that only names
    itself, has no signature to show, and griffe renders it as the bare name. A fence holding a
    name and no parentheses reads as a constructor that takes no arguments, which is a
    different claim, so there is nothing to render.
    """
    try:
        signature = str(member.signature())
    except Exception:
        return None
    return signature if "(" in signature else None


def _documented_names(docstring: Any) -> set[str]:
    """The names a docstring's own Parameters and Attributes sections already cover."""
    if docstring is None:
        return set()
    return {
        item.name
        for section in docstring.parsed
        if section.kind.value in ("parameters", "attributes")
        for item in section.value
    }


def _attribute_rows(members: list[Any]) -> list[tuple[str, ...]]:
    return [
        (
            f"`{member.name}`",
            f"`{member.annotation}`" if member.annotation is not None else "",
            f"`{member.value}`" if member.value is not None else "",
            _rst_to_markdown(member.docstring.value) if member.docstring else "",
        )
        for member in members
    ]


def _render_member(member: Any, qualifier: str, level: int) -> list[str]:
    """Render one class, function or property, and a class's own members beneath it."""
    name = f"{qualifier}{member.name}"
    lines = ["#" * level + f" `{name}`", ""]
    if member.kind.value == "class" and member.bases:
        lines += ["Bases: " + ", ".join(f"`{base}`" for base in member.bases), ""]
    signature = _signature(member)
    if signature is not None:
        lines += ["```python", f"{name}{signature[len(member.name) :]}", "```", ""]
    lines += _sections(member.docstring)

    if member.kind.value != "class":
        return lines

    covered = _documented_names(member.docstring)
    children = documented_members(member)
    attributes = [
        child for child in children if child.kind.value == "attribute" and child.name not in covered
    ]
    if attributes:
        lines += ["**Attributes**", "", *_table(_attribute_rows(attributes), ATTRIBUTE_HEADERS), ""]
    for child in children:
        if child.kind.value != "attribute":
            lines += _render_member(child, f"{name}.", level + 1)
    return lines


def render(module: Any) -> str:
    """Write one module's reference page.

    Parameters
    ----------
    module : griffe.Module
        The module to document.

    Returns
    -------
    str
        The page, as MyST markdown.
    """
    lines = ["---", f"title: {module.path}", "---", "", f"# `{module.path}`", ""]
    lines += _sections(module.docstring)

    children = submodules(module)
    if children:
        rows = [(f"[`{child.path}`]({child.path}.md)", _summary(child)) for child in children]
        lines += ["## Modules", "", *_table(rows, ("Module", "Summary")), ""]

    members = documented_members(module)
    for member in members:
        if member.kind.value != "attribute":
            lines += _render_member(member, "", 2)

    attributes = [member for member in members if member.kind.value == "attribute"]
    if attributes:
        lines += ["## Attributes", "", *_table(_attribute_rows(attributes), ATTRIBUTE_HEADERS), ""]

    return "\n".join(lines).rstrip("\n") + "\n"


def write_pages(output_dir: Path, package: str = PACKAGE) -> list[Path]:
    """Write a page for every module in a package.

    Parameters
    ----------
    output_dir : Path
        Where the pages go. Created if it does not exist.
    package : str, optional
        The package to document. Defaults to `simcardemsx`.

    Returns
    -------
    list of Path
        The pages written, parents before children.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for module in modules(load(package)):
        path = output_dir / f"{module.path}.md"
        path.write_text(render(module))
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Write the reference pages to the directory named on the command line."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print(f"usage: {Path(__file__).name} <output directory>", file=sys.stderr)
        return 2
    for path in write_pages(Path(arguments[0])):
        print(f"wrote {path}  ({len(path.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
