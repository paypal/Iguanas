import polars as pl
import pytest

from iguanas.rule_registry import RuleRegistry, filter_rule_pairs_by_overlap


class TestRuleRegistry:
    @pytest.fixture
    def registry(self):
        return RuleRegistry()  # in-memory

    @pytest.fixture
    def metrics_df(self):
        return pl.DataFrame(
            {"rule": ["rule_A", "rule_B"], "precision": [0.9, 0.7], "recall": [0.6, 0.8]}
        )

    def test_save_and_list(self, registry):
        registry.save("v1", rules=['(X["age"] > 30)'])
        assert "v1" in registry.list()

    def test_load_rules(self, registry):
        rules = ['(X["age"] > 30)', '(X["income"] < 50000)']
        registry.save("v1", rules=rules)
        entry = registry.load("v1")
        assert entry["rules"] == rules

    def test_load_metrics_as_dataframe(self, registry, metrics_df):
        registry.save("v1", rules=["r1"], metrics=metrics_df)
        entry = registry.load("v1")
        assert isinstance(entry["metrics"], pl.DataFrame)
        assert entry["metrics"].shape == metrics_df.shape

    def test_load_no_metrics(self, registry):
        registry.save("v1", rules=["r1"])
        entry = registry.load("v1")
        assert entry["metrics"] is None

    def test_load_metadata(self, registry):
        meta = {"threshold": 0.5, "note": "baseline"}
        registry.save("v1", rules=["r1"], metadata=meta)
        assert registry.load("v1")["metadata"] == meta

    def test_saved_at_set(self, registry):
        registry.save("v1", rules=["r1"])
        assert registry.load("v1")["saved_at"] is not None

    def test_overwrite_snapshot(self, registry):
        registry.save("v1", rules=["r1"])
        registry.save("v1", rules=["r2"])
        assert registry.load("v1")["rules"] == ["r2"]

    def test_delete(self, registry):
        registry.save("v1", rules=["r1"])
        registry.delete("v1")
        assert "v1" not in registry.list()

    def test_delete_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.delete("nonexistent")

    def test_load_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.load("nonexistent")

    def test_list_sorted(self, registry):
        registry.save("beta", rules=["r1"])
        registry.save("alpha", rules=["r2"])
        assert registry.list() == ["alpha", "beta"]

    def test_compare_returns_dataframe(self, registry, metrics_df):
        registry.save("v1", rules=["r1", "r2"], metrics=metrics_df)
        m2 = metrics_df.with_columns(pl.col("precision") + 0.05)
        registry.save("v2", rules=["r1", "r2"], metrics=m2)
        result = registry.compare("v1", "v2")
        assert isinstance(result, pl.DataFrame)
        assert "precision_v1" in result.columns
        assert "precision_v2" in result.columns
        assert "rule" in result.columns

    def test_compare_raises_if_no_metrics(self, registry):
        registry.save("v1", rules=["r1"])
        registry.save("v2", rules=["r1"])
        with pytest.raises(ValueError, match="no saved metrics"):
            registry.compare("v1", "v2")

    def test_persistence(self, tmp_path):
        path = tmp_path / "registry.json"
        reg1 = RuleRegistry(path)
        reg1.save("v1", rules=['(X["age"] > 30)'])
        # Load from disk in a new instance
        reg2 = RuleRegistry(path)
        assert "v1" in reg2.list()
        assert reg2.load("v1")["rules"] == ['(X["age"] > 30)']

    def test_persistence_delete_syncs(self, tmp_path):
        path = tmp_path / "registry.json"
        reg1 = RuleRegistry(path)
        reg1.save("v1", rules=["r1"])
        reg1.delete("v1")
        reg2 = RuleRegistry(path)
        assert reg2.list() == []


class TestFilterRulePairsByOverlap:
    def test_disjoint_rules_detected(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, True, False, False],
                "rule_B": [False, False, True, True],
            }
        )
        result = filter_rule_pairs_by_overlap(R, max_overlap=0.0)
        assert result.shape[0] == 1
        assert result["jaccard"][0] == pytest.approx(0.0)
        assert result["flagged_by_both"][0] == 0

    def test_overlapping_rules_excluded_by_max_overlap(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, True, False, False],
                "rule_B": [True, False, True, False],
            }
        )
        result = filter_rule_pairs_by_overlap(R, max_overlap=0.0)
        assert result.is_empty()

    def test_default_returns_all_pairs(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, True, False, False],
                "rule_B": [True, False, True, False],
            }
        )
        result = filter_rule_pairs_by_overlap(R)
        assert result.shape[0] == 1  # all pairs returned with defaults

    def test_max_overlap_threshold(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, True, True, True, False],
                "rule_B": [False, False, False, True, True],
            }
        )
        # jaccard = 1/5 = 0.2
        result = filter_rule_pairs_by_overlap(R, max_overlap=0.2)
        assert result.shape[0] == 1

    def test_min_overlap_threshold(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, True, False, False],
                "rule_B": [True, True, False, False],  # jaccard=1.0 with A
                "rule_C": [True, False, True, False],  # jaccard=0.33 with A
            }
        )
        result = filter_rule_pairs_by_overlap(R, min_overlap=0.9)
        assert result.shape[0] == 1
        assert result["jaccard"][0] == pytest.approx(1.0)

    def test_min_and_max_overlap_range(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, True, False, False],
                "rule_B": [True, False, True, False],  # jaccard ≈ 0.33
                "rule_C": [True, True, False, False],  # jaccard = 1.0 with A
            }
        )
        result = filter_rule_pairs_by_overlap(R, min_overlap=0.2, max_overlap=0.5)
        assert all(0.2 <= j <= 0.5 for j in result["jaccard"].to_list())

    def test_single_rule_returns_empty(self):
        R = pl.DataFrame({"rule_A": [True, False, True]})
        result = filter_rule_pairs_by_overlap(R)
        assert result.is_empty()

    def test_returns_correct_schema_when_empty(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, True, False],
                "rule_B": [True, False, True],
            }
        )
        result = filter_rule_pairs_by_overlap(R, max_overlap=0.0)
        assert result.is_empty()
        assert "rule_a" in result.columns
        assert "jaccard" in result.columns

    def test_sorted_ascending_by_jaccard(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, False, False, False],
                "rule_B": [False, True, False, False],  # disjoint: jaccard=0
                "rule_C": [True, True, False, False],   # overlaps A: jaccard=0.5
            }
        )
        result = filter_rule_pairs_by_overlap(R, max_overlap=0.5)
        jaccards = result["jaccard"].to_list()
        assert jaccards == sorted(jaccards)


class TestRuleRegistryCompareBranches:
    def test_compare_raises_if_second_snapshot_has_no_metrics(self):
        """compare() raises ValueError when the second snapshot has no metrics DataFrame."""
        registry = RuleRegistry()
        metrics_df = pl.DataFrame({"rule": ["r1"], "precision": [0.9]})
        registry.save("v1", rules=["r1"], metrics=metrics_df)
        registry.save("v2", rules=["r1"])  # no metrics
        with pytest.raises(ValueError, match="no saved metrics"):
            registry.compare("v1", "v2")
