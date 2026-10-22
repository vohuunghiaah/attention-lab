# AGENTS.md - Repository Guidelines for AI Agents

Welcome, AI Agent. This document defines the rules, engineering standards, and architectural conventions for working on the `attention-lab` repository.

---

## 1. Project Overview & Mission

`attention-lab` is a pedagogical, modular repository providing clear, mathematically accurate implementations of attention mechanisms from scratch using PyTorch.

Key goals:
- Clarity and readable code over over-engineering.
- Strict adherence to standard Transformer mathematical formulations (e.g., Vaswani et al., 2017).
- Full test coverage for tensor shapes, masking invariants, and gradient flow.

---

## 2. Directory Conventions

- `src/attention_lab/`: Primary Python package source code.
  - `attention.py`: Core Scaled Dot-Product Attention (functional + module).
  - `multi_head.py`: Multi-Head Attention mechanism.
  - `masking.py`: Causal (autoregressive) and padding masks.
- `tests/`: Automated test suite using `pytest`.
- `notebooks/`: Jupyter notebooks for visual walkthroughs and step-by-step intuition.
- `app/`: Interactive demos (e.g. Streamlit visualizers for attention weights).

---

## 3. Code Standards & Guidelines

- **Python Version**: >= 3.10
- **Type Annotations**: All functions and methods must have complete type hints (`torch.Tensor`, `Optional`, `Tuple`, etc.).
- **Docstrings**: Google or NumPy style docstrings detailing input shapes, output shapes, and mathematical intuition.
- **Tensor Shape Comments**: Annotate intermediate tensor operations with shapes (e.g., `# (batch_size, num_heads, seq_len, d_k)`).
- **Naming Conventions**:
  - `d_model`: Embedding dimension.
  - `d_k`, `d_v`: Head projection dimensions.
  - `seq_len_q`, `seq_len_k`: Query and Key sequence lengths.

---

## 4. Verification & Testing

Before completing any task, ensure:
1. `pytest tests/` passes with zero errors.
2. Tensor shape compatibility across single-head and multi-head attention.
3. Attention weights sum to 1 across the key dimension (`dim=-1`).
4. Proper masking behavior: masked positions must have zero attention weight after softmax.

---

## 5. Agent Workflow & Tracking

- Always record notable changes, design decisions, and session milestones in `AI_TRACKER.md`.
- Keep modifications minimal and clean.
