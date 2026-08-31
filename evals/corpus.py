"""The eval corpus: four documents of deliberately different shape, ten
hand-labeled questions each (40 total).

This module is the single source of truth for the question set. Both
eval_modes.py (RAG vs. stuff) and eval_chunk_sizes.py (chunk-size sweep)
import DOCUMENTS from here, so the README's numbers for both come from
running those two scripts over exactly this set.

Why these four -- the spread is the point, each one stresses a different
part of the extract -> chunk -> retrieve -> cite pipeline:

    notes_primer.pdf        synthetic, 10 pp -- dense informal prose with
                            section headers, one topic per page. The
                            "study notes" shape the original
                            scratch/sample.pdf had; kept so results stay
                            comparable to the old 9-question set.

    thermostat_manual.pdf   synthetic, 10 pp -- imperative voice,
                            numbered procedures, spec and troubleshooting
                            tables. Structurally the opposite of flowing
                            prose.

    apache_license_2.0.pdf  real text, typeset, 6 pp -- the verbatim
                            Apache License 2.0. Legal register, numbered
                            clauses 1-9, defined terms ("Licensor",
                            "Derivative Works", "Legal Entity").

    nist_sp800-63-3.pdf     real published PDF, 76 pp -- NIST Digital
                            Identity Guidelines, committed as downloaded.
                            Normative "SHALL/SHOULD" language, deep
                            section nesting, genuine extraction quirks
                            (running headers/footers pulled inline,
                            printed folios != PDF sheet numbers), and
                            big enough that stuff mode's per-question
                            token cost is no longer close to RAG's.

Provenance and licensing for the two real documents: see corpus/SOURCES.md.

Each question is labeled with the set of PDF sheet numbers (1-indexed,
matching extraction.Page.number) where the answer legitimately lives --
usually one, a few span two. Scoring stays the cheap heuristic
eval_modes.py always used: a hit is any labeled page turning up in the
answer's citations. It catches "cited the wrong page" or "cited nothing";
it says nothing about whether the prose is right, so read the printed
answers.
"""

from __future__ import annotations

import pathlib
from typing import NamedTuple


class Document(NamedTuple):
    name: str
    path: str  # relative to repo root -- run the eval scripts from there
    shape: str  # one-line description for the eval report
    questions: list[tuple[str, set[int]]]


NOTES_PRIMER = Document(
    name="notes_primer",
    path="evals/corpus/notes_primer.pdf",
    shape="synthetic study notes, 10 pp, dense prose",
    questions=[
        ("How many bits is a MAC address?", {2}),
        ("Which three IPv4 address ranges are reserved for private use?", {3}),
        ("How many usable host addresses are in a /24 subnet, and why not 256?", {4}),
        ("When several routes match a packet's destination, how does a router choose?", {5}),
        ("What happens to a packet's TTL at each hop, and what happens when it hits zero?", {5}),
        ("What are the three messages of the TCP three-way handshake?", {6}),
        ("In TCP, what is the difference between flow control and congestion control?", {7}),
        ("Which DNS record type maps a name to an IPv6 address?", {8}),
        ("What does the first digit of an HTTP status code mean, and what does 4xx indicate?", {9}),
        ("In the TLS handshake, what is asymmetric cryptography used for?", {10}),
    ],
)

THERMOSTAT_MANUAL = Document(
    name="thermostat_manual",
    path="evals/corpus/thermostat_manual.pdf",
    shape="synthetic product manual, 10 pp, procedures + tables",
    questions=[
        ("What voltage powers the T-200, and which wire terminal supplies it?", {2}),
        ("Which Wi-Fi band does the thermostat support?", {2, 5}),
        ("What backup batteries does the T-200 use, and how many?", {2, 9}),
        ("What should you do before disconnecting the wires from the old thermostat?", {3}),
        ("What is the setpoint temperature range?", {2}),
        ("How many schedule periods can each day have, and what are they named?", {7}),
        ("What does Adaptive Recovery do?", {8}),
        ("What does Eco mode change?", {8}),
        ("The thermostat screen is blank -- what does the guide say to check?", {9}),
        ("How do you factory reset the T-200?", {10}),
    ],
)

APACHE_LICENSE = Document(
    name="apache_license_2.0",
    path="evals/corpus/apache_license_2.0.pdf",
    shape="real license text, typeset, 6 pp, numbered clauses",
    questions=[
        ('How does the license define "Derivative Works"?', {2}),
        ("Does the Apache License grant patent rights, and on what terms?", {3}),
        (
            "Under Section 3, what happens to your patent license if you start patent "
            "litigation alleging the Work infringes?",
            {3},
        ),
        ("What conditions must you meet when redistributing the Work or Derivative Works?", {3, 4}),
        ("What must you do about a NOTICE text file when distributing Derivative Works?", {4}),
        ("On what basis does the Licensor provide the Work, per the warranty disclaimer?", {5}),
        ("When redistributing, may you offer paid support or a warranty, and on whose behalf?", {6}),
        ('What does "Legal Entity" mean, and what ownership percentage counts as control?', {1}),
        ("Does the license grant permission to use the Licensor's trademarks?", {5}),
        ("Under Section 8, when can a Contributor be held liable for damages?", {5}),
    ],
)

NIST_SP_800_63_3 = Document(
    name="nist_sp800-63-3",
    path="evals/corpus/nist_sp800-63-3.pdf",
    shape="real 76 pp government standard, normative language",
    questions=[
        ("What three assurance levels do agencies select as distinct options under SP 800-63-3?", {8, 16}),
        ("What does AAL2 provide, and how many authentication factors does it require?", {10}),
        ("How is the SP 800-63 suite organized into companion volumes?", {9}),
        ("How does the Introduction define a digital identity?", {15}),
        ("What is the stated purpose of SP 800-63-3?", {6, 14}),
        ("In enrollment and identity proofing, what does an applicant become after successful proofing?", {24, 25}),
        ("On what two kinds of secrets are authenticators based?", {26}),
        ("What biometric characteristics does the document give as examples for verifying identity?", {27}),
        ("Why does SP 800-63-3 say identity federation is preferred over siloed identity systems?", {28}),
        ("Does SP 800-63-3 establish new risk management processes for agencies?", {30}),
    ],
)

DOCUMENTS: list[Document] = [
    NOTES_PRIMER,
    THERMOSTAT_MANUAL,
    APACHE_LICENSE,
    NIST_SP_800_63_3,
]

QUESTION_COUNT = sum(len(d.questions) for d in DOCUMENTS)
assert QUESTION_COUNT == 40, QUESTION_COUNT


def require_pdfs() -> None:
    """Fail early, with a fix, if a corpus PDF is missing."""
    missing = [d.path for d in DOCUMENTS if not pathlib.Path(d.path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing corpus PDF(s): "
            + ", ".join(missing)
            + ". The synthetic/typeset ones are produced by "
            "`python evals/corpus/build_corpus.py`; nist_sp800-63-3.pdf is "
            "committed as-is (see evals/corpus/SOURCES.md)."
        )
