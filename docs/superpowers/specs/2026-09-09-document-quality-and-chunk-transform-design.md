# Document Quality Gate and Chunk Transform Design

## Goal

Prevent malformed or unreadable PDFs from entering the knowledge base, expose a
structured rejection result for a future Dashboard, and optionally clean and enrich
ordinary prose chunks with an LLM after deterministic splitting.

## Scope

This change covers the offline ingestion path from PDF parsing through chunk creation.
It does not build a Dashboard, add OCR fallback, or change retrieval, reranking, and
answer generation. The command-line workflow remains the primary entry point.

## Architecture

The ingestion flow becomes:

```text
PDF
  -> quality gate (always on by default)
  -> MinerU parsing
  -> normalization and table/image serialization
  -> deterministic splitter
  -> optional prose chunk transform (off by default)
  -> indexing
```

Quality inspection remains independent of MinerU and uses PyMuPDF for a fast sample.
Chunk transformation is implemented in a separate processing module so it can be
invoked by the splitter or rerun independently without parsing the PDF again.

## PDF Quality Model

The quality checker samples the first five physical pages, or every page when the
document has fewer than five pages. Metrics are accumulated per page so separator
characters inserted by the application cannot make an empty document appear valid.

The result contains:

- `valid_char_ratio`: recognizable non-whitespace characters divided by all extracted
  non-whitespace characters.
- `recognizable_chars`: number of recognizable non-whitespace characters.
- `extracted_chars`: number of extracted non-whitespace characters.
- `average_chars_per_page`: recognizable characters divided by sampled pages.
- `text_page_ratio`: sampled pages containing at least one recognizable character,
  divided by sampled pages.
- `sample_pages`: number of pages inspected.
- `passed`, `reason`, and a stable failure code.

Recognizable characters include Unicode letters and numbers, ordinary punctuation,
and common currency or mathematical symbols. Control, surrogate, unassigned,
private-use, formatting, and replacement characters are invalid. Whitespace is ignored
in the ratio. This is a text-layer health heuristic, not a language detector; printable
mojibake composed entirely of ordinary letters cannot be detected reliably without a
language model.

A document is rejected when:

- no recognizable text is extracted from the sample; or
- `valid_char_ratio < 0.80`.

Density and text-page ratio are reported for diagnostics but are not hard rejection
thresholds in this iteration. This avoids rejecting annual reports whose cover and
front-matter pages are intentionally sparse.

## Parsing Result Contract

`parse_and_export()` continues processing a batch when one file fails and returns one
structured result per input PDF. Each result has a status from `accepted`, `rejected`,
`failed`, or `skipped`, plus the source path, user-facing message, optional stable error
code, optional quality metrics, and optional output path.

Quality rejection uses error code `DOCUMENT_QUALITY_REJECTED` and the exact user-facing
message:

```text
该文档质量不达标，请检查后重新上传
```

This contract can be consumed directly by a future Dashboard. The CLI prints the same
information. Existing parsed outputs remain `skipped` to preserve restart behavior;
they are not silently revalidated or overwritten. `--no-quality-check` remains an
explicit escape hatch.

## Optional Chunk Transform

`enterprise_rag.processing.chunk_transformer` processes only chunks whose type is
`content`. Serialized tables and images already have specialized LLM processing and
are copied unchanged.

For each prose chunk, the model receives the current text and bounded excerpts from
the previous and next prose chunks. It must return only a cleaned form of the current
chunk, with these constraints:

- remove residual headers, footers, control artifacts, and obvious garbling;
- repair a sentence cut by a physical boundary only when adjacent context supports it;
- retain the existing heading breadcrumb or other useful context;
- preserve facts, names, dates, monetary values, percentages, and other numbers;
- do not import unrelated facts from adjacent chunks or invent missing content.

The transformer validates that output is non-empty and that normalized numeric tokens
from the original chunk remain present. Validation failure or an exhausted API retry
keeps the original text and records transform status and reason in chunk metadata.
Successful transformations preserve the original text in `original_text`, replace
`text`, recompute `length_tokens`, and record model/status metadata. This makes the
operation auditable and idempotent.

Transformation is disabled by default. It can be requested with `text_splitter
--transform` or run independently over an existing chunked directory. Independent
runs support restart behavior by skipping chunks already marked successfully
transformed. Concurrency and retry behavior follow the existing table serializer
patterns, and one failed chunk never aborts the document.

## Error Handling

- Unopenable, empty, or textless PDFs are rejected before MinerU.
- Quality-check exceptions become structured `failed` results rather than terminating
  the entire batch.
- A missing LLM API key is reported before an explicitly requested transform starts.
- LLM response parse errors, empty responses, numeric loss, and API failures fall back
  to the original chunk and are recorded per chunk.
- Output files are written only after a document-level operation has produced a valid
  result.

## Testing Strategy

Tests create small PDFs locally with PyMuPDF and exercise the public APIs. Coverage
includes ordinary text, blank multi-page PDFs, replacement/control characters, the
80-percent boundary, missing files, and structured parser rejection. MinerU is replaced
at the parser boundary only where invoking the real GPU parser would be inappropriate.

Chunk-transform tests use a deterministic fake OpenAI-compatible client, with cases
for successful cleanup, unchanged table/image chunks, numeric-loss fallback, API
failure isolation, restart skipping, and default-off splitter behavior. No test sends a
network request or consumes LLM credits.

## Compatibility

The existing parsed JSON and chunked JSON remain readable. New result and transform
metadata are additive. The current default pipeline performs the quality gate but does
not incur new LLM calls unless transformation is explicitly enabled.
