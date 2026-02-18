Repository layout and usage

Top-level directories (current):

- `src/`: training and scoring Python scripts (ML code)
- `model/dirosophila/`: saved model files organized per-role (`acceptor`, `donar`) and per-model subdirs with `best.pt`
- `data/Dmel/`: dataset area. Subfolders:
	- `raw/` : raw genome / GTF files
	- `train/` : processed training inputs
	- `trans_score/` : transcript-level score files (flat names include window length, e.g. `cnn30bp.tsv`)
- `evaluate/`: evaluation scripts and `evaluate/data/` for eval-only data (e.g. `transcript_class.txt`, `eval_score/`)

Design decisions

- Keep a single repository but separate concerns: `src/` for ML code, `evaluate/` for analysis/plots. Common data kept under `data/`.
- Score files are flattened and named with the window length (e.g. `trans_score/cnn30bp.tsv`) rather than nested 30bp/50bp folders.
- Models are provided per role and per architecture as `model/dirosophila/<role>/<arch>/best.pt` (symlinks created where needed).

Quick run examples

Run CNN training + scoring (auto device selection):

```bash
# Train + score (writes transcript scores to data/Dmel/trans_score/...)
uv run python3 src/cnn_splice_scoring.py
```

Score only (use existing model dirs):

```bash
uv run python3 src/cnn_splice_scoring.py --skip_training --donor_model_dir model/dirosophila/donar/cnn --acceptor_model_dir model/dirosophila/acceptor/cnn
```

Run evaluation scripts (in `evaluate/src`):

```bash
uv run python3 evaluate/src/visualize.py
```

Notes & troubleshooting

- If a script expects `best.pt` inside a model directory, use the per-model directory layout above, or pass the exact checkpoint path via the CLI.
- Use `uv run` to run scripts inside the project's virtual environment (provides `torch`).
- If you change data layout, update defaults in `src/*` or pass explicit CLI paths.

If you want, I can:
- create small compatibility helpers (symlinks) for old `outputs/` paths, or
- update all scripts to accept either a model checkpoint file or a model directory.


