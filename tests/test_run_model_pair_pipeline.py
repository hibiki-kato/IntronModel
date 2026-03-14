from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluate_scores
import run_model
from util.unique_intron import UniqueMapMember


class _DummyPairModelModule:
    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        del parser

    def add_infer_args(self, parser: argparse.ArgumentParser) -> None:
        del parser

    def train(
        self,
        common_args: argparse.Namespace,
        model_args: argparse.Namespace,
    ) -> dict[str, object]:
        del model_args
        checkpoint_path = Path(str(common_args.pair_checkpoint_path))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"pair")
        return {
            "pair": {
                "checkpoint": str(checkpoint_path),
                "best_metric": "pr_auc",
                "best_score": 0.93,
            }
        }

    def infer_site(
        self,
        common_args: argparse.Namespace,
        model_args: argparse.Namespace,
    ) -> list[dict[str, object]]:
        del model_args
        assert str(common_args.pair_checkpoint_path).endswith(".pt")
        return [
            {
                "transcript_id": "tx1",
                "intron_index": 1,
                "site_type": "pair",
                "score": 0.25,
            },
            {
                "transcript_id": "tx1",
                "intron_index": 2,
                "site_type": "pair",
                "score": 0.75,
            },
        ]


def test_expand_unique_site_rows_maps_back_members() -> None:
    """Expand one unique site row into multiple original transcript introns."""
    unique_rows: list[dict[str, object]] = [
        {
            "transcript_id": "uintron_00000001",
            "intron_index": 1,
            "site_type": "pair",
            "score": 0.42,
        }
    ]
    unique_map = {
        ("uintron_00000001", 1): [
            UniqueMapMember(transcript_id="txA", intron_index=2),
            UniqueMapMember(transcript_id="txB", intron_index=7),
        ]
    }

    mapped = run_model._expand_unique_site_rows(
        site_score_rows=unique_rows,
        unique_map=unique_map,
    )

    assert len(mapped) == 2
    assert {str(row["transcript_id"]) for row in mapped} == {"txA", "txB"}
    assert {int(row["intron_index"]) for row in mapped} == {2, 7}


def test_expand_unique_site_rows_raises_when_mapping_missing() -> None:
    """Raise when unique site rows cannot be mapped back to originals."""
    unique_rows: list[dict[str, object]] = [
        {
            "transcript_id": "uintron_99999999",
            "intron_index": 1,
            "site_type": "pair",
            "score": 0.42,
        }
    ]

    with pytest.raises(ValueError, match="unmapped introns"):
        _ = run_model._expand_unique_site_rows(
            site_score_rows=unique_rows,
            unique_map={},
        )


def test_expand_unique_site_rows_allows_original_keyed_rows() -> None:
    """Pass through rows that are already keyed by original introns."""
    site_rows: list[dict[str, object]] = [
        {
            "transcript_id": "txA",
            "intron_index": 2,
            "site_type": "pair",
            "score": 0.42,
        }
    ]
    unique_map = {
        ("uintron_00000001", 1): [
            UniqueMapMember(transcript_id="txA", intron_index=2),
        ]
    }

    mapped = run_model._expand_unique_site_rows(
        site_score_rows=site_rows,
        unique_map=unique_map,
    )

    assert mapped == site_rows


def test_run_pipeline_pair_model_writes_compatible_transcript_tsv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_json = tmp_path / "train_summary.json"
    site_output_tsv = tmp_path / "site.tsv"
    intron_output_tsv = tmp_path / "intron.tsv"
    transcript_output_tsv = tmp_path / "transcript.tsv"
    eval_output_txt = tmp_path / "eval.txt"
    class_file = tmp_path / "class.txt"
    class_file.write_text("tx1\t1\n", encoding="utf-8")
    test_tsv = tmp_path / "transcripts.tsv"
    test_tsv.write_text(
        "transcript_id\tsite_type\tintron_index\tseq\n",
        encoding="utf-8",
    )
    ref_gff = tmp_path / "ref.gff"
    ref_gff.write_text("##gff-version 3\n", encoding="utf-8")

    parser = run_model._build_parser(
        selected_model="cnn_pair",
        skip_model_import_error=True,
    )
    args = parser.parse_args(
        [
            "--model",
            "cnn_pair",
            "--species",
            "Dmel",
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
            "--train_target",
            "pair",
            "--epochs",
            "1",
            "--metrics_json",
            str(metrics_json),
            "--test_tsv",
            str(test_tsv),
            "--class_file",
            str(class_file),
            "--site_output_tsv",
            str(site_output_tsv),
            "--intron_output_tsv",
            str(intron_output_tsv),
            "--transcript_output_tsv",
            str(transcript_output_tsv),
            "--eval_output_txt",
            str(eval_output_txt),
            "--ref_gff",
            str(ref_gff),
        ]
    )

    monkeypatch.setattr(
        run_model,
        "load_model_module",
        lambda model_name: _DummyPairModelModule(),
    )
    monkeypatch.setattr(
        run_model,
        "prune_species_model_checkpoints",
        lambda **_: SimpleNamespace(
            total_candidates=0,
            kept_count=0,
            deleted_count=0,
            dry_run=False,
        ),
    )
    monkeypatch.setattr(
        evaluate_scores,
        "evaluate_score_file",
        lambda **_: ["ok"],
    )
    monkeypatch.setattr(
        evaluate_scores,
        "plot_eval_scores",
        lambda **_: None,
    )
    monkeypatch.setattr(
        run_model,
        "_load_required_unique_intron_map",
        lambda species: {
            ("tx1", 1): [
                UniqueMapMember(transcript_id="tx1", intron_index=1),
            ],
            ("tx1", 2): [
                UniqueMapMember(transcript_id="tx1", intron_index=2),
            ],
        },
    )

    run_model.run_pipeline(args)

    summary = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert summary["pair"]["best_metric"] == "pr_auc"
    assert float(summary["pair"]["best_score"]) == pytest.approx(0.93)

    intron_lines = intron_output_tsv.read_text(encoding="utf-8").strip().splitlines()
    assert intron_lines[0].split("\t") == [
        "transcript_id",
        "intron_index",
        "score",
        "label",
    ]
    assert len(intron_lines) == 3

    lines = transcript_output_tsv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split("\t") == [
        "transcript_id",
        "min_intron_index",
        "Score_donor",
        "Score_acceptor",
        "min_donor_plus_acceptor",
    ]
    assert len(lines[1].split("\t")) == 5
