"""Sequence transform helpers shared by independent and pair models."""

from __future__ import annotations

from dataclasses import dataclass

SEQUENCE_TRANSFORM_CHOICES: tuple[str, ...] = (
    "none",
    "mask_outside_intron_n",
)


@dataclass(frozen=True)
class PairSequenceRecord:
    """Pair sequence container used by transform helpers.

    Attributes
    ----------
    donor_seq : str
        Donor-side sequence in transcript orientation.
    acceptor_seq : str
        Acceptor-side sequence in transcript orientation.
    """

    donor_seq: str
    acceptor_seq: str


def validate_sequence_transform(mode: str) -> str:
    """Validate one transform mode string.

    Parameters
    ----------
    mode : str
        Requested sequence transform mode.

    Returns
    -------
    str
        Normalized mode string.

    Raises
    ------
    ValueError
        If mode is unsupported.
    """
    normalized = mode.strip().lower()
    if normalized not in SEQUENCE_TRANSFORM_CHOICES:
        raise ValueError(
            "Unsupported sequence_transform: "
            f"{mode}. Supported: {SEQUENCE_TRANSFORM_CHOICES}"
        )
    return normalized


def _mask_outside_boundary_window(
    sequence: str,
    *,
    site_type: str,
    intron_half_length: int,
    exon_context_bp: int,
) -> str:
    """Mask sequence regions outside a boundary-local window with ``N``.

    Parameters
    ----------
    sequence : str
        Input sequence.
    site_type : str
        ``donor`` or ``acceptor``.
    intron_half_length : int
        Intron half length used to derive keep span.
    exon_context_bp : int
        Exonic context length near boundary.

    Returns
    -------
    str
        Sequence with outside positions replaced by ``N``.

    Raises
    ------
    ValueError
        If arguments are invalid.
    """
    if intron_half_length < 0:
        raise ValueError(
            "intron_half_length must be >= 0 for masking, "
            f"got: {intron_half_length}"
        )
    if exon_context_bp <= 0:
        raise ValueError("exon_context_bp must be > 0.")

    keep = min(len(sequence), intron_half_length + exon_context_bp)
    seq_upper = sequence.upper()
    if site_type == "donor":
        return seq_upper[:keep] + ("N" * max(0, len(seq_upper) - keep))
    if site_type == "acceptor":
        return ("N" * max(0, len(seq_upper) - keep)) + seq_upper[-keep:]
    raise ValueError(f"Unsupported site_type for masking: {site_type}")


def apply_site_sequence_transform(
    sequence: str,
    *,
    site_type: str,
    transform_mode: str,
    intron_half_length: int | None,
    exon_context_bp: int = 3,
) -> str:
    """Apply sequence transform to one site sequence.

    Parameters
    ----------
    sequence : str
        Input sequence in transcript orientation.
    site_type : str
        ``donor`` or ``acceptor``.
    transform_mode : str
        Transform mode from ``SEQUENCE_TRANSFORM_CHOICES``.
    intron_half_length : int | None
        Intron half length. Required for masking mode.
    exon_context_bp : int, default=3
        Exonic context length near splice boundary.

    Returns
    -------
    str
        Transformed sequence.

    Raises
    ------
    ValueError
        If mode or required metadata is invalid.
    """
    mode = validate_sequence_transform(transform_mode)
    if mode == "none":
        return sequence.upper()

    if intron_half_length is None:
        raise ValueError(
            "sequence_transform mask_outside_intron_n requires "
            "intron_half_length metadata."
        )

    return _mask_outside_boundary_window(
        sequence=sequence,
        site_type=site_type,
        intron_half_length=intron_half_length,
        exon_context_bp=exon_context_bp,
    )


def apply_pair_sequence_transform(
    pair: PairSequenceRecord,
    *,
    transform_mode: str,
    intron_half_length: int | None,
    exon_context_bp: int = 3,
) -> PairSequenceRecord:
    """Apply sequence transform to donor/acceptor pair.

    Parameters
    ----------
    pair : PairSequenceRecord
        Donor and acceptor sequences.
    transform_mode : str
        Transform mode from ``SEQUENCE_TRANSFORM_CHOICES``.
    intron_half_length : int | None
        Intron half length. Required for masking mode.
    exon_context_bp : int, default=3
        Exonic context length near splice boundary.

    Returns
    -------
    PairSequenceRecord
        Transformed pair sequences.
    """
    donor_seq = apply_site_sequence_transform(
        pair.donor_seq,
        site_type="donor",
        transform_mode=transform_mode,
        intron_half_length=intron_half_length,
        exon_context_bp=exon_context_bp,
    )
    acceptor_seq = apply_site_sequence_transform(
        pair.acceptor_seq,
        site_type="acceptor",
        transform_mode=transform_mode,
        intron_half_length=intron_half_length,
        exon_context_bp=exon_context_bp,
    )
    return PairSequenceRecord(donor_seq=donor_seq, acceptor_seq=acceptor_seq)
