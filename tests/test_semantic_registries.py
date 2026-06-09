"""Reproducibility tests for the externalized semantic registries (WS-4).

Asserts that:
* each ``data/*.json`` registry loads,
* the data covers the full 34-asset registry,
* the in-code fallback dicts are byte-identical to the historical hard-coded
  dicts (behavior preservation when a data file is absent),
* the loaded values are a coverage-superset of the fallback (and identical for
  the historical entries — no silent drift),
* the three previously-parallel category dicts are now harmonized to one
  source of truth.
"""

from __future__ import annotations

import importlib.util

import pytest

from libero_infinity import bddl_preprocessor as bp
from libero_infinity import perturbation_policy_helpers as pph
from libero_infinity import semantic_registries as sr
from libero_infinity.asset_registry import ALL_LIBERO_CLASSES


# --------------------------------------------------------------------------- #
# Load rate_cf_bddls.py as a module so we can read its (now harmonized)
# _CATEGORY without invoking main().
# --------------------------------------------------------------------------- #
def _load_rate_cf():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "rate_cf_bddls", root / "scripts" / "rate_cf_bddls.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# 1. Data files load
# --------------------------------------------------------------------------- #
class TestDataFilesLoad:
    def test_cf_category_groups_loads(self):
        groups = sr.load_cf_category_groups()
        assert groups is not None and len(groups) > 0

    def test_surface_compatibility_loads(self):
        result = sr.load_surface_compatibility()
        assert result is not None
        physical_category, incompatible = result
        assert len(physical_category) > 0
        assert len(incompatible) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in incompatible)

    def test_support_scale_factors_loads(self):
        scales = sr.load_support_scale_factors()
        assert scales is not None and len(scales) > 0
        assert all(len(v) == 2 for v in scales.values())


# --------------------------------------------------------------------------- #
# 2. Coverage of the full 34-asset registry
# --------------------------------------------------------------------------- #
class TestCoverage:
    def test_cf_category_covers_all_assets(self):
        missing = [c for c in ALL_LIBERO_CLASSES if c not in bp._CF_CATEGORY]
        assert missing == [], f"cf_category missing assets: {missing}"

    def test_physical_category_covers_all_assets(self):
        missing = [c for c in ALL_LIBERO_CLASSES if c not in bp._VISUAL_CATEGORY]
        assert missing == [], f"physical_category missing assets: {missing}"

    def test_newer_assets_have_rules(self):
        # The assets the RCA flagged as uncovered must now be categorized.
        for asset in ("popcorn", "corn", "cherries", "mayo", "bowl_drainer"):
            assert asset in bp._CF_CATEGORY
            assert asset in bp._VISUAL_CATEGORY

    def test_support_scale_covers_all_infer_types(self):
        # Every value infer_support_type can return must have a scale entry,
        # otherwise support_local_envelope raises KeyError at runtime.
        for support_type in (
            "contained",
            "cook_surface",
            "shelf_surface",
            "workspace",
            "object_surface",
        ):
            assert support_type in pph._SUPPORT_SCALE_BY_TYPE


# --------------------------------------------------------------------------- #
# 3. Fallback == historical in-code dict (behavior preservation)
# --------------------------------------------------------------------------- #
class TestFallbackBehaviorPreservation:
    _HISTORICAL_VISUAL = {
        "akita_black_bowl": "bowl",
        "white_bowl": "bowl",
        "glazed_rim_porcelain_ramekin": "bowl",
        "plate": "bowl",
        "chefmate_8_frypan": "cookware",
        "red_coffee_mug": "mug",
        "white_yellow_mug": "mug",
        "porcelain_mug": "mug",
        "moka_pot": "mug",
        "black_book": "book",
        "yellow_book": "book",
        "wine_bottle": "bottle",
        "ketchup": "bottle",
        "milk": "bottle",
        "orange_juice": "bottle",
        "tomato_sauce": "bottle",
        "bbq_sauce": "bottle",
        "salad_dressing": "bottle",
        "new_salad_dressing": "bottle",
        "cream_cheese": "carton",
        "butter": "carton",
        "chocolate_pudding": "carton",
        "alphabet_soup": "carton",
        "cookies": "carton",
        "basket": "container",
        "wooden_tray": "container",
        "desk_caddy": "container",
    }
    _HISTORICAL_INCOMPATIBLE = {
        ("container", "bowl"),
        ("bowl", "bowl"),
        ("bowl", "stove"),
        ("carton", "stove"),
        ("book", "stove"),
        ("mug", "stove"),
        ("book", "rack"),
    }
    _HISTORICAL_CF = {
        "alphabet_soup": "food_can",
        "tomato_sauce": "food_can",
        "bbq_sauce": "food_can",
        "ketchup": "food_can",
        "salad_dressing": "food_can",
        "new_salad_dressing": "food_can",
        "cream_cheese": "food_box",
        "butter": "food_box",
        "chocolate_pudding": "food_box",
        "red_coffee_mug": "mug",
        "white_yellow_mug": "mug",
        "black_book": "book",
        "akita_black_bowl": "bowl_plate",
        "white_bowl": "bowl_plate",
        "plate": "bowl_plate",
        "wine_bottle": "bottle",
    }
    _HISTORICAL_SUPPORT = {
        "contained": (0.45, 0.45),
        "cook_surface": (0.75, 0.75),
        "shelf_surface": (0.60, 0.60),
        "workspace": (0.60, 0.60),
        "object_surface": (0.60, 0.60),
    }

    def test_visual_fallback_matches_history(self):
        assert bp._VISUAL_CATEGORY_FALLBACK == self._HISTORICAL_VISUAL

    def test_incompatible_fallback_matches_history(self):
        assert bp._INCOMPATIBLE_FALLBACK == self._HISTORICAL_INCOMPATIBLE

    def test_cf_fallback_matches_history(self):
        assert bp._CF_CATEGORY_FALLBACK == self._HISTORICAL_CF

    def test_support_fallback_matches_history(self):
        assert pph._SUPPORT_SCALE_BY_TYPE_FALLBACK == self._HISTORICAL_SUPPORT


# --------------------------------------------------------------------------- #
# 4. Data is a coverage-superset of the fallback, identical for historical keys
# --------------------------------------------------------------------------- #
class TestDataSupersetOfFallback:
    def test_visual_data_preserves_fallback_entries(self):
        for key, value in bp._VISUAL_CATEGORY_FALLBACK.items():
            assert bp._VISUAL_CATEGORY[key] == value

    def test_cf_data_preserves_fallback_entries(self):
        for key, value in bp._CF_CATEGORY_FALLBACK.items():
            assert bp._CF_CATEGORY[key] == value

    def test_incompatible_data_superset_of_fallback(self):
        assert bp._INCOMPATIBLE_FALLBACK <= bp._INCOMPATIBLE

    def test_support_data_matches_fallback(self):
        # Support scales have no coverage additions — identical content.
        assert pph._SUPPORT_SCALE_BY_TYPE == pph._SUPPORT_SCALE_BY_TYPE_FALLBACK


# --------------------------------------------------------------------------- #
# 5. Absent / malformed data file -> None -> fallback (no crash)
# --------------------------------------------------------------------------- #
class TestFallbackPath:
    def test_absent_file_yields_none(self, monkeypatch):
        monkeypatch.setattr(sr.pkgutil, "get_data", lambda *a, **k: None)
        assert sr.load_cf_category_groups() is None
        assert sr.load_surface_compatibility() is None
        assert sr.load_support_scale_factors() is None

    def test_absent_file_resolves_to_fallback(self, monkeypatch):
        monkeypatch.setattr(sr.pkgutil, "get_data", lambda *a, **k: None)
        physical, incompatible, cf = bp._resolve_registries()
        assert physical == bp._VISUAL_CATEGORY_FALLBACK
        assert incompatible == bp._INCOMPATIBLE_FALLBACK
        assert cf == bp._CF_CATEGORY_FALLBACK

    def test_malformed_file_yields_none(self, monkeypatch):
        monkeypatch.setattr(sr.pkgutil, "get_data", lambda *a, **k: b"{ not json")
        assert sr.load_cf_category_groups() is None
        assert sr.load_surface_compatibility() is None
        assert sr.load_support_scale_factors() is None


# --------------------------------------------------------------------------- #
# 6. Harmonization: the three parallel category dicts are now one source
# --------------------------------------------------------------------------- #
class TestHarmonization:
    def test_cf_category_single_source_of_truth(self):
        rate_cf = _load_rate_cf()
        # rate_cf_bddls.py:_CATEGORY and bddl_preprocessor._CF_CATEGORY now
        # resolve from the same data file, so they must be identical.
        assert rate_cf._CATEGORY == bp._CF_CATEGORY

    def test_cf_category_partition_is_finer_than_legacy_rate_cf(self):
        # Harmonization splits condiments (food_can) from tall bottles (bottle):
        # in the live partition wine_bottle and ketchup are NOT in the same group.
        assert bp._CF_CATEGORY["wine_bottle"] != bp._CF_CATEGORY["ketchup"]
        # ...and soup-cans (food_can) are split from rectangular cartons (food_box).
        assert bp._CF_CATEGORY["alphabet_soup"] != bp._CF_CATEGORY["cream_cheese"]

    def test_physical_and_cf_axes_are_intentionally_distinct(self):
        # Documented intentional difference: wine_bottle and ketchup share a
        # PHYSICAL category (placement) but differ in cf_category (grounding).
        assert bp._VISUAL_CATEGORY["wine_bottle"] == bp._VISUAL_CATEGORY["ketchup"]
        assert bp._CF_CATEGORY["wine_bottle"] != bp._CF_CATEGORY["ketchup"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
