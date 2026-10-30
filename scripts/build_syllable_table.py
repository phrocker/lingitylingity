"""Derive lingity's vendored syllable table from a pinned CMUdict release.

This runs once, by a maintainer, when the pinned dictionary changes. Analysis
never runs it and never fetches anything: ``lingity/readability.py`` reads the
table this script writes, and refuses a table whose hash differs from the one
it pins.

Usage::

    python scripts/build_syllable_table.py path/to/cmudict.dict

The input must be ``cmudict.dict`` from ``https://github.com/cmusphinx/cmudict``
at the commit named in ``CMUDICT_COMMIT``; any other bytes are refused, so the
table can always be rebuilt from a named upstream file and compared.

Derivation, for every headword:

* the syllable count of a pronunciation is the number of its phones that carry
  a stress digit (``0``, ``1`` or ``2``), which in ARPAbet marks exactly the
  vowels;
* when a word has several pronunciations, the first one listed wins. CMUdict
  lists its primary pronunciation first and numbers the variants ``word(2)``,
  ``word(3)`` and so on, so the choice is fixed by the file and needs no
  judgement here;
* trailing ``# comment`` text is ignored.

The output is UTF-8, one ``word<TAB>count`` line per headword, sorted by word,
with a final newline.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

CMUDICT_COMMIT = "74790861f652b15e4ac49015a90074ad62a27690"
CMUDICT_SHA256 = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
OUTPUT = Path(__file__).resolve().parents[1] / "lingity" / "data" / "cmudict-syllables.tsv"
_VARIANT = re.compile(r"^(?P<word>.+)\((?P<number>\d+)\)$")


def derive(source: str) -> dict[str, int]:
    table: dict[str, int] = {}
    for line in source.splitlines():
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        headword, *phones = content.split()
        if not phones:
            raise ValueError(f"CMUdict line has no phones: {line!r}")
        variant = _VARIANT.match(headword)
        word = variant.group("word") if variant else headword
        if word in table:
            continue
        if variant is not None:
            raise ValueError(f"CMUdict lists variant {headword!r} before its primary entry")
        table[word] = sum(1 for phone in phones if phone[-1] in "012")
    return table


def render(table: dict[str, int]) -> str:
    return "".join(f"{word}\t{table[word]}\n" for word in sorted(table))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmudict", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    raw = Path(args.cmudict).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CMUDICT_SHA256:
        raise SystemExit(
            f"{args.cmudict} has sha256 {digest}, not the pinned CMUdict "
            f"{CMUDICT_SHA256} (commit {CMUDICT_COMMIT}). Refusing to derive a table "
            "from unpinned input."
        )
    rendered = render(derive(raw.decode("utf-8")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"{output}: {rendered.count(chr(10))} entries, sha256 {hashlib.sha256(rendered.encode('utf-8')).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
