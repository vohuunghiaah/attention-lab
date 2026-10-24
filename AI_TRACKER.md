# AI Task & Interaction Tracker (AI_TRACKER)

This document tracks all AI-assisted operations, architectural decisions, completed tasks, and session logs for `attention-lab`.

---

## 📋 Session Log

| Date | Agent / Model | Task / Objective | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 2026-10-22 | Antigravity | Initialize project directory structure, AGENTS.md, AI_TRACKER | ✅ Completed | Created empty directory structure with `src/attention_lab/`, `tests/`, `notebooks/`, `app/`, `AGENTS.md`, and `AI_TRACKER`. |

---

## 🎯 Project Milestones

- [x] **Milestone 1: Repository Foundation**
  - [x] Standard folder structure initialized
  - [x] Config files added (`requirements.txt`, `.gitignore`, `LICENSE`)
  - [x] Agent rules (`AGENTS.md`) and tracking (`AI_TRACKER.md`) established
- [x] **Milestone 2: Attention Implementations & Testing**
  - [x] Scaled Dot-Product Attention (`attention.py`)
  - [x] Multi-Head Attention (`multi_head.py`)
  - [x] Masking Utilities (`masking.py`)
  - [x] Automated unit test suite (`tests/`)
- [ ] **Milestone 3: Interactive Visualizations & Demos**
  - [x] Interactive Streamlit app for attention map inspection (`app/`)
  - [ ] Pedagogical walkthrough notebook (`notebooks/`)

## 2026-10-23 — Complete interactive laboratory implementation

- Inspected existing skeleton and retained original import paths through small adapters.
- Added independent NumPy/PyTorch engines, causal/padding masks, bias-free multi-head
  projections, inspectable intermediate results, and deterministic SHA-256 token inputs.
- Mask convention: True = allowed. Fully masked rows return zero weights/output with
  finite gradients; rectangular causal masks are upper-left aligned. Leading batch
  dimensions must match rather than implicitly broadcast.
- Built four Vietnamese Streamlit tabs with live Plotly matrices, query selection,
  scaling comparison, causal comparison, custom tokens and configurable multi-head results.
- Added pyproject packaging, dependency compatibility entry point, GitHub Actions,
  Vietnamese README, mathematical/API notes, and real screenshots under assets/.
- Existing notebook was inadvertently processed by repository-wide Ruff. Restored its
  contents; final git diff for test.ipynb is empty. Excluded notebooks from Ruff.
  A new walkthrough notebook remains an optional future task, not part of this delivery.

### Executed verification

- Created a clean Python 3.11.9 .venv; installed all dependencies using uv, then verified
  the documented `python -m pip install -e ".[dev]"` command successfully.
- `python -m pytest tests/ -q`: **21 passed, 1 skipped** (12.89 seconds).
  CUDA was unavailable; the optional device test was skipped. CPU float32/64 tested.
- Reference tests compare with PyTorch SDPA and nn.MultiheadAttention; autograd gradcheck,
  hand-worked scores, mask broadcasting and fully masked rows all passed.
- AppTest exercised causal toggle, query selection, seed, custom input, embedding width,
  token count, head count, invalid divisibility, empty input and excessive input length.
  Fixed an initial test assertion that inspected probability sums instead of weights.
- Ruff check passed; formatting was applied to Python files only.
- Streamlit launched on 127.0.0.1:8501 and was inspected in the in-app browser.
  Verified real causal matrices and zero future probability; saved two screenshots.
- Sandbox DNS initially blocked dependency installation, and sandbox loopback sockets
  hung Streamlit/AppTest before app execution. Approved unsandboxed execution resolved
  both; no application workaround or security configuration change was needed.
- Versions installed: NumPy 2.4.6, PyTorch 2.14.1, Streamlit 1.66.1, Plotly 6.9.0,
  pytest 9.1.1, Ruff 0.17.1. CI workflow authored but not run on GitHub.
- No commit, remote push, deployment, pretrained model or external AI service used.

## 2026-10-23 — Improve matrix indices and mathematical presentation

- Addressed all three browser comments: Q/K/V explicitly show row index i = 0,1,2
  and feature columns 0,1; all four intermediate matrices have indexed numeric tables.
- Replaced the crowded inline formula chain with individual equation panels and four
  stages: S (raw scores), S_scaled (before mask), A (probabilities), O (weighted values).
- Used bold matrix notation, upright operators/subscripts, proper fractions and sums;
  distinguished masked scores Z from S_scaled and documented row/column meanings.
- Updated chart titles and split Multi-Head equations. Added explicit pandas dependency.
- Verification: pytest tests/ — 21 passed, 1 CUDA skip; Ruff lint and format checks passed.
  AppTest checks Q/K/V indices plus causal probabilities and existing interactive controls.
- Inspected the running browser at the user's 894px viewport. Saved indexed-qkv.png
  as evidence of the revised tables. Mathematical engines were unchanged.

## 2026-10-23 — Remove matrix index header label

- Removed the index name "i" from the shared matrix table renderer for Q, K, V,
  S, S_scaled, A and O. Numeric row and column indices remain visible.
- Updated the existing UI assertion to expect an unnamed row index.
- Verification: 21 tests passed, 1 optional CUDA test skipped; Ruff lint and format passed.
