from __future__ import annotations

import pytest

from tools.run_wrapper_pipeline import _parse_species_list


def test_parse_species_list_supports_commas_and_spaces() -> None:
    """Parse mixed comma/space separated species values."""
    parsed = _parse_species_list("Athal, Dmel Mmus")
    assert parsed == ["Athal", "Dmel", "Mmus"]


def test_parse_species_list_deduplicates_preserving_order() -> None:
    """Keep first occurrence order while dropping duplicates."""
    parsed = _parse_species_list("Athal,Dmel,Athal  Dmel")
    assert parsed == ["Athal", "Dmel"]


def test_parse_species_list_rejects_empty_input() -> None:
    """Reject empty or separator-only input."""
    with pytest.raises(ValueError, match="at least one species"):
        _parse_species_list(" ,  ")
