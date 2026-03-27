from __future__ import annotations

import argparse
import sys
from pathlib import Path

ANALYSIS_SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "script"
if str(ANALYSIS_SCRIPT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT))

from score.l1_notebook_utils import (
    build_snpr_args,
    read_tsv_head,
    resolve_repo_root,
)


def test_resolve_repo_root_finds_repository_root(tmp_path: Path) -> None:
    """The helper should locate the repository root from a nested path."""

    repo_root = tmp_path / "repo"
    (repo_root / ".git").mkdir(parents=True)
    (repo_root / "analysis").mkdir()
    nested = repo_root / "analysis" / "notebooks" / "nested"
    nested.mkdir(parents=True)

    resolved = resolve_repo_root(nested)

    assert resolved == repo_root


def test_build_snpr_args_returns_expected_namespace(tmp_path: Path) -> None:
    """The SN-PR argument bundle should mirror the notebook configuration."""

    args = build_snpr_args(
        repo_root=tmp_path,
        train_species="Mmus",
        snpr_eval_species="all",
        score_model="dnabert2",
        precision_target=0.85,
        recall_target=0.85,
        logreg_c=1.0,
        random_state=7,
        test_size=0.25,
        valid_size=0.2,
        max_transcripts=123,
        sn_denominator="reference_all",
    )

    assert isinstance(args, argparse.Namespace)
    assert args.data_root == tmp_path / "data"
    assert args.train_species == "Mmus"
    assert args.eval_species == "all"
    assert args.score_model == "dnabert2"
    assert args.random_state == 7
    assert args.test_size == 0.25
    assert args.valid_size == 0.2
    assert args.max_transcripts == 123
    assert args.sn_denominator == "reference_all"


def test_read_tsv_head_prints_first_rows(tmp_path: Path, capsys) -> None:
    """The TSV preview helper should print only the requested rows."""

    path = tmp_path / "sample.tsv"
    path.write_text(
        "a\tb\n1\t2\n3\t4\n5\t6\n",
        encoding="utf-8",
    )

    read_tsv_head(path, n=2)

    captured = capsys.readouterr()
    assert "{'a': '1', 'b': '2'}" in captured.out
    assert "{'a': '3', 'b': '4'}" in captured.out
    assert "{'a': '5', 'b': '6'}" not in captured.out
