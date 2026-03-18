# AGENTS.md

## General Rules

- All source code, comments, and docstrings must be written in English.
- Target Python version: **3.12 or later**.
- Follow PEP 8.
- Maximum line length: 88 characters.
- Use explicit imports only. No wildcard imports.
- Avoid deprecated features.

---

## Type Hints

- All functions and methods must include explicit type hints.
- All class attributes must have type annotations.
- Do not use `Any` unless strictly necessary and justified.
- Use `from __future__ import annotations`.
- Prefer:
  - `dataclasses` for structured data
  - `TypedDict` for dictionary-like structured objects
  - `Protocol` for interface-style abstractions
- Avoid untyped containers such as `list` or `dict`. Use `list[int]`, `dict[str, float]`, etc.

---

## Documentation

- Every public function, method, and class must include a docstring.
- Use NumPy-style or Google-style docstrings consistently.
- Each docstring must include:
  - Clear description of purpose
  - Parameter descriptions with types
  - Return value description
  - Raised exceptions
  - Shape information for tensors or arrays where applicable
- Nontrivial algorithms must briefly describe:
  - Core idea
  - Computational complexity

---

## Testing

- Provide meaningful unit tests using `pytest`.
- Cover:
  - Normal cases
  - Edge cases
  - Failure modes
- Avoid global state in tests.
- Tests must be deterministic.
- Use fixed random seeds where randomness is involved.
Pytest is installed in conda intronmodel.

---

## Design Principles

- Prefer pure functions where possible.
- Avoid hidden global state.
- Separate:
  - Data processing
  - Model definition
  - Training logic
  - Evaluation logic
- Do not mix core logic with I/O.
- Avoid side effects unless explicitly required.
- Keep functions small and composable.

---

## Input Validation and Error Handling

- Validate all public inputs explicitly.
- Raise appropriate exceptions (`ValueError`, `TypeError`, etc.).
- Do not silently coerce invalid inputs.
- Fail early and clearly.

---

## Machine Learning Specific Guidelines

- Clearly specify:
  - Input tensor shapes
  - Output tensor shapes
  - Expected data types
- Do not hardcode hyperparameters inside functions.
- Pass configuration explicitly via arguments or configuration objects.
- Avoid implicit device placement (CPU/GPU). Make it explicit.
- Ensure reproducibility:
  - Set random seeds explicitly.
  - Allow seed injection via configuration.

---

## Code Quality Expectations

- Avoid overly clever code.
- Prioritize clarity over brevity.
- Avoid premature optimization.
- Document assumptions clearly.
- Keep dependencies minimal and justified.
- If introducing a new dependency, explain why it is necessary.

---

## Cross-Model Change Policy

- Treat runtime-default changes (for example compile/AMP/device behavior) as
  cross-model changes by default.
- Prefer implementing shared defaults in common infrastructure (for example
  `run/lib/common.sh`) instead of patching only one model script.
- If a requested change is first made in one model, explicitly ask whether to
  apply it to all relevant models before finishing the task.

---

This file defines mandatory coding standards.  
All generated code must comply strictly with these rules.
