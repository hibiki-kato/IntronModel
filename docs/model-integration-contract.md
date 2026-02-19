# Model Integration Contract

Any model connected to `src/run_model.py` must be registered in
`src/models/registry.py` and must expose this API:

- `add_train_args(parser: argparse.ArgumentParser) -> None`
- `add_infer_args(parser: argparse.ArgumentParser) -> None`
- `train(common_args: argparse.Namespace, model_args: argparse.Namespace) -> dict[str, object]`
- `infer_site(common_args: argparse.Namespace, model_args: argparse.Namespace) -> list[dict[str, object]]`

## Required `infer_site` Output Rows

Each row must contain:

- `transcript_id`: `str`
- `intron_index`: `int`
- `site_type`: `str` (`donor` or `acceptor`)
- `score`: `float`

## Pipeline Compatibility Rules

- Do not bypass checkpoint naming logic in `run_model.py`.
- Keep model-specific arguments additive and explicit.
- Avoid hardcoded absolute paths.
- Keep device handling explicit (`auto|cuda|mps|cpu`).
- Validate public inputs and fail early with clear exceptions.

## Recommended Validation Before Registry Connection

1. `python3 src/run_model.py --model <new_model> --help` works.
2. Model module passes contract validation in `load_model_module`.
3. `infer_site` rows aggregate correctly via `util.transcript_eval`.
4. Wrapper scripts and README examples are updated if new options are exposed.
