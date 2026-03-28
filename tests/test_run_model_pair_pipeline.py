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


class _DummyPairModelModuleDuplicateMembers(_DummyPairModelModule):
    def infer_site(
        self,
        common_args: argparse.Namespace,
        model_args: argparse.Namespace,
    ) -> list[dict[str, object]]:
        del model_args
        assert str(common_args.pair_checkpoint_path).endswith(".pt")
        return [
            {
                "transcript_id": "txA",
                "intron_index": 1,
                "site_type": "pair",
                "score": 0.25,
            },
            {
                "transcript_id": "txB",
                "intron_index": 3,
                "site_type": "pair",
                "score": 0.25,
            },
        ]


class _DummyPairModelModuleCompileRetry(_DummyPairModelModule):
    def __init__(self) -> None:
        self.infer_calls: list[tuple[object, object]] = []

    def infer_site(
        self,
        common_args: argparse.Namespace,
        model_args: argparse.Namespace,
    ) -> list[dict[str, object]]:
        del model_args
        self.infer_calls.append(
            (
                getattr(common_args, "infer_compile", None),
                getattr(common_args, "infer_compile_mode", None),
            )
        )
        if len(self.infer_calls) == 1:
            raise RuntimeError(
                "torch._inductor.exc.InductorError: PermissionError: "
                "[Errno 13] Permission denied: '/afs/glue/.triton'"
            )
        return [
            {
                "transcript_id": "tx1",
                "intron_index": 1,
                "site_type": "pair",
                "score": 0.25,
            }
        ]


class _DummyCnnV2PairRowsModule:
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
        donor_path = Path(str(common_args.donor_checkpoint_path))
        acceptor_path = Path(str(common_args.acceptor_checkpoint_path))
        donor_path.parent.mkdir(parents=True, exist_ok=True)
        acceptor_path.parent.mkdir(parents=True, exist_ok=True)
        donor_path.write_bytes(b"donor")
        acceptor_path.write_bytes(b"acceptor")
        return {
            "donor": {
                "checkpoint": str(donor_path),
                "best_metric": "pr_auc",
                "best_score": 0.81,
            },
            "acceptor": {
                "checkpoint": str(acceptor_path),
                "best_metric": "pr_auc",
                "best_score": 0.82,
            },
        }

    def infer_site(
        self,
        common_args: argparse.Namespace,
        model_args: argparse.Namespace,
    ) -> list[dict[str, object]]:
        del common_args, model_args
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


def test_collapse_site_rows_to_unique_maps_original_members() -> None:
    """Collapse original-member rows to one unique intron key per site type."""
    site_rows: list[dict[str, object]] = [
        {
            "transcript_id": "txA",
            "intron_index": 3,
            "site_type": "donor",
            "score": 0.85,
        },
        {
            "transcript_id": "txA",
            "intron_index": 3,
            "site_type": "acceptor",
            "score": 0.45,
        },
        {
            "transcript_id": "txB",
            "intron_index": 7,
            "site_type": "donor",
            "score": 0.85,
        },
        {
            "transcript_id": "txB",
            "intron_index": 7,
            "site_type": "acceptor",
            "score": 0.45,
        },
    ]
    unique_map = {
        ("uintron_00000002", 1): [
            UniqueMapMember(transcript_id="txA", intron_index=3),
            UniqueMapMember(transcript_id="txB", intron_index=7),
        ]
    }

    collapsed = run_model._collapse_site_rows_to_unique(
        site_score_rows=site_rows,
        unique_map=unique_map,
    )

    assert len(collapsed) == 2
    assert {
        (
            str(row["transcript_id"]),
            int(row["intron_index"]),
            str(row["site_type"]),
            float(row["score"]),
        )
        for row in collapsed
    } == {
        ("uintron_00000002", 1, "acceptor", 0.45),
        ("uintron_00000002", 1, "donor", 0.85),
    }


def test_collapse_site_rows_to_unique_raises_on_conflicting_scores() -> None:
    """Raise when one unique key receives conflicting collapsed site scores."""
    site_rows: list[dict[str, object]] = [
        {
            "transcript_id": "txA",
            "intron_index": 3,
            "site_type": "donor",
            "score": 0.85,
        },
        {
            "transcript_id": "txB",
            "intron_index": 7,
            "site_type": "donor",
            "score": 0.32,
        },
    ]
    unique_map = {
        ("uintron_00000002", 1): [
            UniqueMapMember(transcript_id="txA", intron_index=3),
            UniqueMapMember(transcript_id="txB", intron_index=7),
        ]
    }

    with pytest.raises(ValueError, match="Conflicting scores"):
        _ = run_model._collapse_site_rows_to_unique(
            site_score_rows=site_rows,
            unique_map=unique_map,
        )


def test_uses_default_unique_test_tsv_true_for_processed_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return true when test_tsv points to processed/transcripts.unique.tsv."""
    base_dir = tmp_path / "data" / "Dmel"
    processed_dir = base_dir / "processed"
    processed_dir.mkdir(parents=True)
    unique_path = processed_dir / "transcripts.unique.tsv"
    unique_path.write_text("header\n", encoding="utf-8")

    monkeypatch.setattr(
        run_model,
        "species_data_dirs",
        lambda species: {
            "base": str(base_dir),
            "raw": str(base_dir / "raw"),
            "learning_metric": str(base_dir / "learning_metric"),
            "eval_score": str(base_dir / "eval_score"),
        },
    )

    assert run_model._uses_default_unique_test_tsv(
        species="Dmel",
        test_tsv=str(unique_path),
    )


def test_uses_default_unique_test_tsv_false_for_non_default_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return false when test_tsv points to a non-default TSV path."""
    base_dir = tmp_path / "data" / "Dmel"
    processed_dir = base_dir / "processed"
    processed_dir.mkdir(parents=True)
    default_unique_path = processed_dir / "transcripts.unique.tsv"
    default_unique_path.write_text("header\n", encoding="utf-8")
    non_default_path = tmp_path / "custom.tsv"
    non_default_path.write_text("header\n", encoding="utf-8")

    monkeypatch.setattr(
        run_model,
        "species_data_dirs",
        lambda species: {
            "base": str(base_dir),
            "raw": str(base_dir / "raw"),
            "learning_metric": str(base_dir / "learning_metric"),
            "eval_score": str(base_dir / "eval_score"),
        },
    )

    assert not run_model._uses_default_unique_test_tsv(
        species="Dmel",
        test_tsv=str(non_default_path),
    )


def test_run_pipeline_pair_model_writes_compatible_transcript_tsv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    monkeypatch.setattr(
        run_model,
        "_load_optional_intron_labels",
        lambda species: {("tx1", 1): 0, ("tx1", 2): 1},
    )

    run_model.run_pipeline(args)
    captured = capsys.readouterr()
    assert "[pipeline] inference stage elapsed:" in captured.out

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
    assert intron_lines[1].split("\t")[3] == "0"
    assert intron_lines[2].split("\t")[3] == "1"

    site_lines = site_output_tsv.read_text(encoding="utf-8").splitlines()
    assert site_lines[0].split("\t") == [
        "transcript_id",
        "intron_index",
        "donor_score",
        "acceptor_score",
        "label",
    ]
    assert site_lines[1].split("\t")[4] == "0"
    assert site_lines[2].split("\t")[4] == "1"

    lines = transcript_output_tsv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split("\t") == [
        "transcript_id",
        "min_intron_index",
        "Score_donor",
        "Score_acceptor",
        "trans_score",
    ]
    assert len(lines[1].split("\t")) == 5


def test_run_pipeline_cnn_pair_v2_rows_aggregate_from_pair_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep cnn_v2 transcript scores non-zero when inference yields pair rows."""
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
        selected_model="cnn_v2",
        skip_model_import_error=True,
    )
    args = parser.parse_args(
        [
            "--model",
            "cnn_v2",
            "--species",
            "Dmel",
            "--donor_len",
            "100",
            "--acceptor_len",
            "100",
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
        lambda model_name: _DummyCnnV2PairRowsModule(),
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
    monkeypatch.setattr(
        run_model,
        "_load_optional_intron_labels",
        lambda species: {("tx1", 1): 1, ("tx1", 2): 0},
    )

    run_model.run_pipeline(args)

    transcript_lines = (
        transcript_output_tsv.read_text(encoding="utf-8").strip().splitlines()
    )
    assert transcript_lines[0].split("\t") == [
        "transcript_id",
        "min_intron_index",
        "Score_donor",
        "Score_acceptor",
        "trans_score",
    ]
    assert transcript_lines[1].split("\t") == [
        "tx1",
        "1",
        "-0.602060",
        "-0.602060",
        "-0.602060",
    ]


def test_run_pipeline_pair_model_retries_infer_without_compile_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
            "--infer_compile",
            "1",
            "--infer_compile_mode",
            "on",
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

    retry_module = _DummyPairModelModuleCompileRetry()
    monkeypatch.setattr(
        run_model,
        "load_model_module",
        lambda model_name: retry_module,
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
        },
    )
    monkeypatch.setattr(
        run_model,
        "_load_optional_intron_labels",
        lambda species: {("tx1", 1): 1},
    )

    run_model.run_pipeline(args)
    captured = capsys.readouterr()

    assert "Retry once with infer_compile=0" in captured.out
    assert retry_module.infer_calls == [(1, "on"), (0, "off")]


def test_run_pipeline_pair_model_uses_unique_intron_scores_and_maps_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_json = tmp_path / "train_summary.json"
    site_output_tsv = tmp_path / "site.tsv"
    intron_output_tsv = tmp_path / "intron.tsv"
    transcript_output_tsv = tmp_path / "transcript.tsv"
    eval_output_txt = tmp_path / "eval.txt"
    class_file = tmp_path / "class.txt"
    class_file.write_text("txA\t1\ntxB\t0\n", encoding="utf-8")
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
        lambda model_name: _DummyPairModelModuleDuplicateMembers(),
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
            ("uintron_00000001", 1): [
                UniqueMapMember(transcript_id="txA", intron_index=1),
                UniqueMapMember(transcript_id="txB", intron_index=3),
            ],
        },
    )

    run_model.run_pipeline(args)

    intron_lines = intron_output_tsv.read_text(encoding="utf-8").strip().splitlines()
    assert len(intron_lines) == 2
    assert intron_lines[1].split("\t")[0:2] == ["uintron_00000001", "1"]

    transcript_lines = (
        transcript_output_tsv.read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(transcript_lines) == 3
    assert {line.split("\t")[0] for line in transcript_lines[1:]} == {"txA", "txB"}
