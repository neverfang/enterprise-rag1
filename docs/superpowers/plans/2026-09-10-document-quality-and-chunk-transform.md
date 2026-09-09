# Document Quality Gate and Chunk Transform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject PDFs with unhealthy text layers before MinerU, return Dashboard-ready ingestion statuses, and provide an explicitly enabled LLM cleanup stage for prose chunks.

**Architecture:** A pure quality-metrics core feeds a PyMuPDF-backed checker and the parser batch contract. A separate chunk transformer operates on splitter output, preserves original text, validates numeric fidelity, and can be called either by `text_splitter --transform` or as a standalone resumable command.

**Tech Stack:** Python 3.10+, PyMuPDF, OpenAI-compatible client, Pydantic-free dataclasses, pytest, Ruff, tiktoken.

---

## File Map

- Modify `src/enterprise_rag/quality/quality_checker.py`: robust character classification, per-page metrics, result serialization, stable failure codes.
- Modify `src/enterprise_rag/quality/__init__.py`: export the quality API.
- Modify `src/enterprise_rag/parsing/pdf_parser.py`: import the checker and return structured per-document ingestion results.
- Create `src/enterprise_rag/processing/chunk_transformer.py`: optional LLM prose cleanup, validation, retries, restart support, and CLI.
- Modify `src/enterprise_rag/processing/text_splitter.py`: add an explicit `--transform` integration while retaining default-off behavior.
- Modify `README.md`: document the quality gate and optional transform commands.
- Create `tests/test_quality_checker.py`: metric and real-PDF regression tests.
- Create `tests/test_pdf_parser.py`: quality rejection and parser status-contract tests.
- Create `tests/test_chunk_transformer.py`: fake-client transform and validation tests.
- Create `tests/test_text_splitter.py`: default-off and explicitly enabled integration tests.

### Task 1: Quality metric contract

**Files:**
- Create: `tests/test_quality_checker.py`
- Modify: `src/enterprise_rag/quality/quality_checker.py`
- Modify: `src/enterprise_rag/quality/__init__.py`

- [ ] **Step 1: Write failing tests for character classification and metric boundaries**

Add tests that call a public `analyze_page_texts()` API directly:

```python
from enterprise_rag.quality.quality_checker import analyze_page_texts


def test_blank_pages_are_rejected_instead_of_counting_join_newlines():
    result = analyze_page_texts(["", "", ""], threshold=0.80)
    assert result.passed is False
    assert result.error_code == "NO_RECOGNIZABLE_TEXT"
    assert result.recognizable_chars == 0
    assert result.average_chars_per_page == 0


def test_replacement_and_control_characters_are_invalid():
    result = analyze_page_texts(["abcd\ufffd\x03"], threshold=0.80)
    assert result.valid_char_ratio == pytest.approx(4 / 6)
    assert result.passed is False


def test_exactly_eighty_percent_passes():
    result = analyze_page_texts(["abcd\ufffd"], threshold=0.80)
    assert result.valid_char_ratio == pytest.approx(0.80)
    assert result.passed is True
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_quality_checker.py -q`

Expected: FAIL because `analyze_page_texts` and the new result fields do not exist.

- [ ] **Step 3: Implement the minimal metric model**

Use `unicodedata.category()` and ignore whitespace in the denominator. Accept categories
starting with `L`, `N`, or `P`, and common symbol categories `Sc`, `Sm`, `Sk`, and `So`;
reject all `C*` categories and U+FFFD. Add these fields to `QualityCheckResult`:

```python
error_code: str | None
recognizable_chars: int
extracted_chars: int
average_chars_per_page: float
text_page_ratio: float

def as_dict(self) -> dict:
    return dataclasses.asdict(self)
```

Implement `analyze_page_texts(page_texts, threshold)` without joining pages. Reject no
recognizable text with `NO_RECOGNIZABLE_TEXT`; reject a ratio below the threshold with
`LOW_VALID_CHAR_RATIO`. Preserve `total_chars` and `valid_chars` as compatibility
properties if existing callers need them.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_quality_checker.py -q`

Expected: all current quality tests PASS.

- [ ] **Step 5: Add real-PDF and input-validation tests**

Create PDFs in pytest's `tmp_path` with PyMuPDF. Cover readable text, three blank pages,
a missing file, a zero-page/invalid file, `sample_pages <= 0`, and a threshold outside
`[0, 1]`. Assert invalid configuration raises `ValueError`, while document problems
return failed `QualityCheckResult` values.

- [ ] **Step 6: Verify RED, implement PyMuPDF delegation, then verify GREEN**

Run the focused test after adding cases and confirm the expected failures. Update
`check_pdf_quality()` to validate arguments, extract one string per sampled page, call
`analyze_page_texts()`, and always close the document. Re-run:

`.venv\Scripts\python.exe -m pytest tests/test_quality_checker.py -q`

Expected: PASS.

- [ ] **Step 7: Export the public API and checkpoint**

Export `QualityCheckResult`, `analyze_page_texts`, and `check_pdf_quality` from the
package `__init__.py`. Run Ruff on the touched files and commit only Task 1 files.

### Task 2: Structured parser ingestion results

**Files:**
- Create: `tests/test_pdf_parser.py`
- Modify: `src/enterprise_rag/parsing/pdf_parser.py`

- [ ] **Step 1: Write failing rejection-contract test**

Patch `check_pdf_quality` at the parser module boundary to return a rejected result,
call `parse_and_export()` on a temporary PDF, and assert:

```python
assert results[0].status == "rejected"
assert results[0].error_code == "DOCUMENT_QUALITY_REJECTED"
assert results[0].message == "该文档质量不达标，请检查后重新上传"
assert results[0].quality["passed"] is False
assert not output_json.exists()
```

Also assert `_run_mineru` was not called.

- [ ] **Step 2: Run the parser test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pdf_parser.py -q`

Expected: FAIL because the checker import is missing and `parse_and_export()` returns
`None`.

- [ ] **Step 3: Add `IngestionResult` and fix the quality-check import**

Define a dataclass with `source`, `status`, `message`, `error_code`, `quality`, and
`output_path`. Import `check_pdf_quality` explicitly. Append exactly one result per
input file and return the list. Keep the existing CLI messages, using the stable
Dashboard message for quality rejection.

- [ ] **Step 4: Verify the rejection test is GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pdf_parser.py -q`

Expected: PASS for the rejection case.

- [ ] **Step 5: Add tests for accepted, skipped, failed, and disabled-check paths**

Use temporary files and patch only MinerU/metadata boundaries. Assert accepted writes
JSON and includes its path, skipped does not call the checker, parser exceptions become
`failed`, and `quality_check=False` bypasses inspection.

- [ ] **Step 6: Verify RED, implement all statuses, and verify GREEN**

Run the focused tests before implementation to observe failures, implement the minimal
status branches, then run:

`.venv\Scripts\python.exe -m pytest tests/test_pdf_parser.py -q`

Expected: PASS.

### Task 3: Prose chunk transformer core

**Files:**
- Create: `tests/test_chunk_transformer.py`
- Create: `src/enterprise_rag/processing/chunk_transformer.py`

- [ ] **Step 1: Write failing tests for numeric extraction and validation**

Specify pure APIs:

```python
assert extract_numeric_tokens("FY2023 revenue was $1,234.50, up 8%") == {
    "2023", "1,234.50", "8%"
}
assert validate_transformed_text("Revenue was $1,234.50", "Revenue improved") \
    == (False, "numeric_tokens_missing: 1,234.50")
assert validate_transformed_text("Revenue was $1,234.50", "Revenue was $1,234.50") \
    == (True, None)
```

- [ ] **Step 2: Run and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chunk_transformer.py -q`

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement pure helpers and verify GREEN**

Implement normalized numeric-token extraction for integers, decimals, grouped numbers,
years, and percentages. Reject empty transformed text and missing original numeric
tokens. Run the focused tests and expect PASS.

- [ ] **Step 4: Write failing fake-client transform tests**

Use a small fake object implementing `client.chat.completions.create`. Cover:

- successful prose cleanup preserves `original_text`, updates `length_tokens`, and
  writes `transform.status == "transformed"`;
- table/image chunks are unchanged and never call the client;
- numeric loss falls back to the original text with status `validation_failed`;
- an API exception after retries keeps the original with status `failed`;
- a previously transformed chunk is skipped on restart;
- adjacent prose excerpts are included in the request without becoming output.

- [ ] **Step 5: Run and verify RED**

Run the new transform cases and confirm failure because `transform_document()` is
missing.

- [ ] **Step 6: Implement the transformer with bounded context and retries**

Add a strict system prompt and request JSON of the form `{"transformed_text": "..."}`.
Implement `transform_document(doc, client, model, max_workers=4, retries=3)` using
`ThreadPoolExecutor`. Operate on copies of chunks, use at most 800 characters from each
adjacent prose chunk, validate outputs, and record per-chunk audit metadata. A failed
future must not abort other chunks.

- [ ] **Step 7: Run transformer tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chunk_transformer.py -q`

Expected: PASS with zero network calls.

### Task 4: Standalone and splitter-integrated transform

**Files:**
- Modify: `src/enterprise_rag/processing/chunk_transformer.py`
- Create: `tests/test_text_splitter.py`
- Modify: `src/enterprise_rag/processing/text_splitter.py`

- [ ] **Step 1: Write failing default-off splitter test**

Call `split_all()` with its existing arguments and assert no client factory or transform
function is called. Add an explicit enabled case using dependency injection and assert
the saved JSON contains transform metadata.

- [ ] **Step 2: Run and verify RED for the enabled case**

Run: `.venv\Scripts\python.exe -m pytest tests/test_text_splitter.py -q`

Expected: enabled case FAIL because `split_all()` has no transform option.

- [ ] **Step 3: Add explicit integration without changing defaults**

Extend `split_all()` with keyword-only `transform=False`, `client=None`, and
`model=None`. Only create a configured client when transformation is explicitly true.
Add CLI flags `--transform` and `--transform-workers`. Keep all old positional calls
working.

- [ ] **Step 4: Add standalone resumable CLI**

Implement chunk transformer arguments `input`, `--out`, `--workers`, and `--force`.
Load OpenAI-compatible settings using the same environment variables as
`table_serializer`. Write each completed document atomically through a sibling
temporary file followed by `Path.replace()`.

- [ ] **Step 5: Run focused integration tests and verify GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_text_splitter.py tests/test_chunk_transformer.py -q
```

Expected: PASS.

### Task 5: Documentation, compatibility, and full verification

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Test: `tests/test_quality_checker.py`
- Test: `tests/test_pdf_parser.py`
- Test: `tests/test_chunk_transformer.py`
- Test: `tests/test_text_splitter.py`

- [ ] **Step 1: Ensure the development environment contains declared dependencies**

Install the project with development extras into `.venv` if needed:

`.venv\Scripts\python.exe -m pip install -e ".[dev]"`

Confirm PyMuPDF imports as `pymupdf`. Do not add a second PDF library.

- [ ] **Step 2: Document the new flow and commands**

Update the README architecture and usage sections with the default quality gate,
structured rejection behavior, `--no-quality-check`, default-off `--transform`, and the
standalone transformer command. Explicitly mention LLM cost and that only prose chunks
are transformed.

- [ ] **Step 3: Run the full test suite**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS with no network access.

- [ ] **Step 4: Run focused Ruff checks and compilation**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check src/enterprise_rag/quality src/enterprise_rag/parsing/pdf_parser.py src/enterprise_rag/processing/chunk_transformer.py src/enterprise_rag/processing/text_splitter.py tests
.venv\Scripts\python.exe -m compileall -q src tests
```

Expected: both commands exit 0. Existing unrelated Ruff findings outside these paths do
not block this feature, but no new finding may be introduced in touched files.

- [ ] **Step 5: Run a local smoke check on one existing development PDF**

Invoke `check_pdf_quality()` on one file from `data/raw/dev` and print its serialized
metrics. Do not run MinerU or a real LLM. Confirm the result includes ratio, density,
text-page ratio, and sampled-page count.

- [ ] **Step 6: Review the final diff against the approved design**

Verify every design requirement has a corresponding implementation or test, ensure no
API key or generated corpus artifact is present, and inspect `git diff --check`.

- [ ] **Step 7: Commit implementation checkpoints without unrelated files**

Stage only files listed in this plan. Preserve unrelated user changes and report any
pre-existing dirty files separately in the handoff.
