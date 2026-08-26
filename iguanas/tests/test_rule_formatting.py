from iguanas.rule_formatting import (
    add_missing_value_conditions,
    decode_discretized_bins,
    decode_null_indicators,
    decode_numeric_encodings,
    decode_onehot_encodings,
    decode_scaled_thresholds,
    decode_string_imputation,
    drop_not_null_conditions,
    drop_null_clauses,
    format_as_boolean_conditions,
    format_floats_as_integers,
    prettify_rules,
    quote_string_values,
    round_thresholds,
    simplify_rule,
)


class TestSimplifyRule:
    """Test cases for simplify_rule function."""

    def test_greater_equal_and_greater_same_value(self):
        """Test that >= is removed when > exists with same value."""
        rule = '(X["amount"] >= 100.0) & (X["amount"] > 100.0)'
        result = simplify_rule(rule)
        assert result == '(X["amount"] > 100.0)'

    def test_less_equal_and_less_same_value(self):
        """Test that <= is removed when < exists with same value."""
        rule = '(X["amount"] < 100.0) & (X["amount"] <= 100.0)'
        result = simplify_rule(rule)
        assert result == '(X["amount"] < 100.0)'

    def test_greater_conditions_different_values(self):
        """Test that lower >= threshold is removed when higher > exists."""
        rule = '(X["a"] >= 50) & (X["b"] < 10) & (X["a"] > 100)'
        result = simplify_rule(rule)
        assert result == '(X["a"] > 100) & (X["b"] < 10)'

    def test_less_conditions_different_values(self):
        """Test that higher <= threshold is removed when lower < exists."""
        rule = '(X["a"] <= 100) & (X["b"] > 10) & (X["a"] < 50)'
        result = simplify_rule(rule)
        assert result == '(X["a"] < 50) & (X["b"] > 10)'

    def test_multiple_columns_with_redundancy(self):
        """Test simplification on multiple columns simultaneously."""
        rule = '(X["a"] >= 50) & (X["a"] > 50) & (X["b"] <= 20) & (X["b"] < 10)'
        result = simplify_rule(rule)
        assert result == '(X["a"] > 50) & (X["b"] < 10)'

    def test_no_redundant_conditions(self):
        """Test that rules without redundancy are unchanged."""
        rule = '(X["a"] > 50) & (X["b"] < 100)'
        result = simplify_rule(rule)
        assert result == '(X["a"] > 50) & (X["b"] < 100)'

    def test_only_greater_equal_no_simplification(self):
        """Test that >= alone is kept when no > exists."""
        rule = '(X["a"] >= 50) & (X["b"] < 100)'
        result = simplify_rule(rule)
        assert result == '(X["a"] >= 50) & (X["b"] < 100)'

    def test_only_less_equal_no_simplification(self):
        """Test that <= alone is kept when no < exists."""
        rule = '(X["a"] > 50) & (X["b"] <= 100)'
        result = simplify_rule(rule)
        assert result == '(X["a"] > 50) & (X["b"] <= 100)'

    def test_multiple_greater_conditions(self):
        """Test that only highest > threshold is kept."""
        rule = '(X["a"] > 10) & (X["a"] > 50) & (X["a"] > 30)'
        result = simplify_rule(rule)
        assert result == '(X["a"] > 50)'

    def test_multiple_less_conditions(self):
        """Test that only lowest < threshold is kept."""
        rule = '(X["a"] < 100) & (X["a"] < 30) & (X["a"] < 50)'
        result = simplify_rule(rule)
        assert result == '(X["a"] < 30)'

    def test_mixed_operators_same_threshold(self):
        """Test preference for strict operators over non-strict."""
        rule = '(X["a"] >= 50) & (X["a"] > 50) & (X["b"] <= 100) & (X["b"] < 100)'
        result = simplify_rule(rule)
        assert result == '(X["a"] > 50) & (X["b"] < 100)'

    def test_empty_rule(self):
        """Test that empty string is returned unchanged."""
        rule = ""
        result = simplify_rule(rule)
        assert result == ""

    def test_single_condition(self):
        """Test that single condition is unchanged."""
        rule = '(X["amount"] >= 100.0)'
        result = simplify_rule(rule)
        assert result == '(X["amount"] >= 100.0)'

    def test_equality_operators_preserved(self):
        """Test that == and != operators are preserved."""
        rule = '(X["a"] == 50) & (X["a"] > 30)'
        result = simplify_rule(rule)
        assert result == '(X["a"] == 50) & (X["a"] > 30)'

    def test_non_numeric_values_preserved(self):
        """Test that non-numeric comparisons are preserved."""
        rule = '(X["name"] == "John") & (X["age"] > 30)'
        result = simplify_rule(rule)
        assert result == '(X["name"] == "John") & (X["age"] > 30)'

    def test_no_double_ampersands_in_output(self):
        """Test that double ampersands never appear in output and column order is preserved."""
        rule = '(X["a"] >= 50) & (X["a"] > 100) & (X["b"] < 10) & (X["a"] > 75)'
        result = simplify_rule(rule)
        assert " & & " not in result
        assert result == '(X["a"] > 100) & (X["b"] < 10)'

    def test_column_order_preservation(self):
        """Test that column order is preserved based on first appearance."""
        rule = '(X["z"] > 1) & (X["a"] < 10) & (X["m"] >= 5) & (X["a"] < 8) & (X["z"] > 3)'
        result = simplify_rule(rule)
        # z appears first, then a, then m - this order should be preserved
        assert result == '(X["z"] > 3) & (X["a"] < 8) & (X["m"] >= 5)'
        # Verify column z comes before a, and a comes before m
        z_pos = result.index('"z"')
        a_pos = result.index('"a"')
        m_pos = result.index('"m"')
        assert z_pos < a_pos < m_pos


class TestConvertFloatToInt:
    """Test cases for format_floats_as_integers function."""

    def test_user_example(self):
        """Test the exact user example."""
        rule = '(X["a"] >= 0.1) & (X["b"] >= 9.1)'
        result = format_floats_as_integers(rule, ["a"])
        assert result == '(X["a"] >= 1) & (X["b"] >= 9.1)'

    def test_greater_equal_operator(self):
        """Test >= operator uses ceiling."""
        rule = '(X["a"] >= 0.1)'
        result = format_floats_as_integers(rule, ["a"])
        assert result == '(X["a"] >= 1)'

    def test_greater_operator(self):
        """Test > operator uses floor."""
        rule = '(X["a"] > 0.9)'
        result = format_floats_as_integers(rule, ["a"])
        assert result == '(X["a"] > 0)'

    def test_less_equal_operator(self):
        """Test <= operator uses floor."""
        rule = '(X["a"] <= 9.9)'
        result = format_floats_as_integers(rule, ["a"])
        assert result == '(X["a"] <= 9)'

    def test_less_operator(self):
        """Test < operator uses ceiling."""
        rule = '(X["a"] < 9.1)'
        result = format_floats_as_integers(rule, ["a"])
        assert result == '(X["a"] < 10)'

    def test_multiple_columns(self):
        """Test conversion on multiple columns."""
        rule = '(X["a"] > 0.1) & (X["b"] < 10.9)'
        result = format_floats_as_integers(rule, ["a", "b"])
        assert result == '(X["a"] > 0) & (X["b"] < 11)'

    def test_selective_conversion(self):
        """Test conversion only happens for specified columns."""
        rule = '(X["a"] >= 0.5) & (X["b"] >= 0.5) & (X["c"] >= 0.5)'
        result = format_floats_as_integers(rule, ["a", "c"])
        assert result == '(X["a"] >= 1) & (X["b"] >= 0.5) & (X["c"] >= 1)'

    def test_already_integer_values(self):
        """Test that integer values remain unchanged."""
        rule = '(X["a"] >= 1) & (X["b"] < 10)'
        result = format_floats_as_integers(rule, ["a", "b"])
        assert result == '(X["a"] >= 1) & (X["b"] < 10)'

    def test_empty_int_columns(self):
        """Test with empty int_columns list."""
        rule = '(X["a"] >= 0.5)'
        result = format_floats_as_integers(rule, [])
        assert result == '(X["a"] >= 0.5)'

    def test_non_numeric_values_unchanged(self):
        """Test that non-numeric values are not converted."""
        rule = '(X["name"] == "test") & (X["a"] >= 0.5)'
        result = format_floats_as_integers(rule, ["name", "a"])
        assert result == '(X["name"] == "test") & (X["a"] >= 1)'

    def test_negative_values(self):
        """Test conversion with negative values."""
        rule = '(X["a"] >= -0.5) & (X["b"] < -2.3)'
        result = format_floats_as_integers(rule, ["a", "b"])
        assert result == '(X["a"] >= 0) & (X["b"] < -2)'


class TestAddNullConditions:
    """Test cases for add_missing_value_conditions function."""

    def test_user_example(self):
        """Test the exact user example."""
        mapping = {"a": 0, "b": 0.3, "c": 100}
        rule = '(X["a"] < 1) & (X["b"] >= 3) & (X["c"] > 10)'
        result = add_missing_value_conditions(rule, mapping)
        assert (
            result
            == '(X["a"] < 1 | X["a"].is_null()) & (X["b"] >= 3) & (X["c"] > 10 | X["c"].is_null())'
        )

    def test_single_quotes(self):
        """Test that single quotes are handled correctly."""
        mapping = {"a": 0}
        rule = "(X['a'] < 1)"
        result = add_missing_value_conditions(rule, mapping)
        assert result == '(X["a"] < 1 | X["a"].is_null())'

    def test_greater_equal_condition_satisfied(self):
        """Test >= condition satisfied by nan value."""
        mapping = {"a": 5}
        rule = '(X["a"] >= 3)'
        result = add_missing_value_conditions(rule, mapping)
        assert result == '(X["a"] >= 3 | X["a"].is_null())'

    def test_greater_equal_condition_not_satisfied(self):
        """Test >= condition not satisfied by nan value."""
        mapping = {"a": 5}
        rule = '(X["a"] >= 10)'
        result = add_missing_value_conditions(rule, mapping)
        assert result == '(X["a"] >= 10)'

    def test_greater_condition_satisfied(self):
        """Test > condition satisfied by nan value."""
        mapping = {"a": 5}
        rule = '(X["a"] > 3)'
        result = add_missing_value_conditions(rule, mapping)
        assert result == '(X["a"] > 3 | X["a"].is_null())'

    def test_less_equal_condition_satisfied(self):
        """Test <= condition satisfied by nan value."""
        mapping = {"a": 5}
        rule = '(X["a"] <= 10)'
        result = add_missing_value_conditions(rule, mapping)
        assert result == '(X["a"] <= 10 | X["a"].is_null())'

    def test_less_condition_satisfied(self):
        """Test < condition satisfied by nan value."""
        mapping = {"a": 5}
        rule = '(X["a"] < 10)'
        result = add_missing_value_conditions(rule, mapping)
        assert result == '(X["a"] < 10 | X["a"].is_null())'

    def test_equality_condition_satisfied(self):
        """Test == condition satisfied by nan value."""
        mapping = {"a": 0}
        rule = '(X["a"] == 0)'
        result = add_missing_value_conditions(rule, mapping)
        assert result == '(X["a"] == 0 | X["a"].is_null())'

    def test_not_equal_condition_satisfied(self):
        """Test != condition satisfied by nan value."""
        mapping = {"a": 0}
        rule = '(X["a"] != 5)'
        result = add_missing_value_conditions(rule, mapping)
        assert result == '(X["a"] != 5 | X["a"].is_null())'

    def test_multiple_conditions_mixed(self):
        """Test multiple conditions with some satisfied and some not."""
        mapping = {"a": 5, "b": 10}
        rule = '(X["a"] <= 10) & (X["b"] > 5)'
        result = add_missing_value_conditions(rule, mapping)
        assert result == '(X["a"] <= 10 | X["a"].is_null()) & (X["b"] > 5 | X["b"].is_null())'

    def test_empty_mapping(self):
        """Test with empty mapping."""
        rule = '(X["a"] < 1)'
        result = add_missing_value_conditions(rule, {})
        assert result == '(X["a"] < 1)'

    def test_column_not_in_mapping(self):
        """Test column not in mapping is unchanged."""
        mapping = {"a": 0}
        rule = '(X["a"] < 1) & (X["b"] < 5)'
        result = add_missing_value_conditions(rule, mapping)
        assert result == '(X["a"] < 1 | X["a"].is_null()) & (X["b"] < 5)'

    def test_non_numeric_value_unchanged(self):
        """Test non-numeric comparison is unchanged."""
        mapping = {"a": 0, "name": "test"}
        rule = '(X["a"] < 1) & (X["name"] == "John")'
        result = add_missing_value_conditions(rule, mapping)
        assert result == '(X["a"] < 1 | X["a"].is_null()) & (X["name"] == "John")'

    def test_float_nan_values(self):
        """Test with float nan replacement values."""
        mapping = {"a": 0.5, "b": 1.5}
        rule = '(X["a"] < 1) & (X["b"] >= 1)'
        result = add_missing_value_conditions(rule, mapping)
        assert result == '(X["a"] < 1 | X["a"].is_null()) & (X["b"] >= 1 | X["b"].is_null())'


class TestDecodeNumericConditions:
    """Test cases for decode_numeric_encodings function."""

    def test_user_example(self):
        """Test the exact user example."""
        mapping = {"A": {"a": 1, "b": 2, "c": 3}, "B": {"a": -8.1, "b": 1.1, "c": 3}}
        rule = '(X["A"] >= 2) & (X["B"] < 0)'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["A"].is_in(["b", "c"])) & (X["B"] == "a")'

    def test_single_match_equality(self):
        """Test that single matching category uses == operator."""
        mapping = {"A": {"x": 1, "y": 2}}
        rule = '(X["A"] == 1)'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["A"] == "x")'

    def test_multiple_matches_isin(self):
        """Test that multiple matching categories use .is_in()."""
        mapping = {"col": {"low": 1, "med": 5, "high": 10}}
        rule = '(X["col"] > 3)'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["col"].is_in(["med", "high"]))'

    def test_greater_than_operator(self):
        """Test > operator finds correct categories."""
        mapping = {"A": {"a": 1, "b": 5, "c": 10}}
        rule = '(X["A"] > 4)'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["A"].is_in(["b", "c"]))'

    def test_greater_equal_operator(self):
        """Test >= operator finds correct categories."""
        mapping = {"A": {"a": 1, "b": 5, "c": 10}}
        rule = '(X["A"] >= 5)'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["A"].is_in(["b", "c"]))'

    def test_less_than_operator(self):
        """Test < operator finds correct categories."""
        mapping = {"A": {"a": 1, "b": 5, "c": 10}}
        rule = '(X["A"] < 6)'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["A"].is_in(["a", "b"]))'

    def test_less_equal_operator(self):
        """Test <= operator finds correct categories."""
        mapping = {"A": {"a": 1, "b": 5, "c": 10}}
        rule = '(X["A"] <= 5)'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["A"].is_in(["a", "b"]))'

    def test_not_equal_operator(self):
        """Test != operator finds correct categories."""
        mapping = {"col": {"low": 1, "med": 5, "high": 10}}
        rule = '(X["col"] != 5)'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["col"].is_in(["low", "high"]))'

    def test_multiple_conditions(self):
        """Test multiple conditions are all decoded."""
        mapping = {"A": {"a": 1, "b": 2}, "B": {"x": 10, "y": 20}}
        rule = '(X["A"] >= 2) & (X["B"] < 15)'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["A"] == "b") & (X["B"] == "x")'

    def test_empty_mapping(self):
        """Test with empty mapping."""
        rule = '(X["A"] >= 2)'
        result = decode_numeric_encodings(rule, {})
        assert result == '(X["A"] >= 2)'

    def test_column_not_in_mapping(self):
        """Test column not in mapping is unchanged."""
        mapping = {"A": {"a": 1, "b": 2}}
        rule = '(X["A"] >= 2) & (X["B"] < 5)'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["A"] == "b") & (X["B"] < 5)'

    def test_negative_encoded_values(self):
        """Test with negative encoded values."""
        mapping = {"A": {"neg": -5, "zero": 0, "pos": 5}}
        rule = '(X["A"] < 0)'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["A"] == "neg")'

    def test_float_encoded_values(self):
        """Test with float encoded values."""
        mapping = {"A": {"low": 0.5, "mid": 1.5, "high": 2.5}}
        rule = '(X["A"] >= 1.0)'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["A"].is_in(["mid", "high"]))'

    def test_all_categories_match(self):
        """Test when all categories satisfy the condition."""
        mapping = {"A": {"a": 1, "b": 2, "c": 3}}
        rule = '(X["A"] >= 0)'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["A"].is_in(["a", "b", "c"]))'

    def test_single_quotes_input(self):
        """Test with single quotes in input."""
        mapping = {"A": {"a": 1, "b": 2}}
        rule = "(X['A'] >= 2)"
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["A"] == "b")'

    def test_non_numeric_value_unchanged(self):
        """Test non-numeric comparison is unchanged."""
        mapping = {"A": {"a": 1, "b": 2}}
        rule = '(X["A"] >= 2) & (X["name"] == "test")'
        result = decode_numeric_encodings(rule, mapping)
        assert result == '(X["A"] == "b") & (X["name"] == "test")'


class TestConvertToBool:
    """Test cases for format_as_boolean_conditions function."""

    def test_user_example_true_with_equals(self):
        """Test True with == operator."""
        rule = '(X["flag"] == "True") & (X["active"] != "False")'
        result = format_as_boolean_conditions(rule, ["flag", "active"])
        assert result == '(X["flag"] == True) & (X["active"] == True)'

    def test_user_example_numeric(self):
        """Test numeric 1 and 0 representations."""
        rule = '(X["enabled"] != 1) & (X["disabled"] == 0)'
        result = format_as_boolean_conditions(rule, ["enabled", "disabled"])
        assert result == '(X["enabled"] == False) & (X["disabled"] == False)'

    def test_lowercase_true_false(self):
        """Test lowercase true/false strings."""
        rule = '(X["is_valid"] == "true") & (X["is_ready"] != "false")'
        result = format_as_boolean_conditions(rule, ["is_valid", "is_ready"])
        assert result == '(X["is_valid"] == True) & (X["is_ready"] == True)'

    def test_true_equals_becomes_true(self):
        """Test True with == becomes True."""
        rule = '(X["col"] == "True")'
        result = format_as_boolean_conditions(rule, ["col"])
        assert result == '(X["col"] == True)'

    def test_true_not_equals_becomes_false(self):
        """Test True with != becomes False (with ==)."""
        rule = '(X["col"] != "True")'
        result = format_as_boolean_conditions(rule, ["col"])
        assert result == '(X["col"] == False)'

    def test_false_equals_becomes_false(self):
        """Test False with == becomes False."""
        rule = '(X["col"] == "False")'
        result = format_as_boolean_conditions(rule, ["col"])
        assert result == '(X["col"] == False)'

    def test_false_not_equals_becomes_true(self):
        """Test False with != becomes True (with ==)."""
        rule = '(X["col"] != "False")'
        result = format_as_boolean_conditions(rule, ["col"])
        assert result == '(X["col"] == True)'

    def test_numeric_one_equals(self):
        """Test numeric 1 with == becomes True."""
        rule = '(X["flag"] == 1)'
        result = format_as_boolean_conditions(rule, ["flag"])
        assert result == '(X["flag"] == True)'

    def test_numeric_one_not_equals(self):
        """Test numeric 1 with != becomes False."""
        rule = '(X["flag"] != 1)'
        result = format_as_boolean_conditions(rule, ["flag"])
        assert result == '(X["flag"] == False)'

    def test_numeric_zero_equals(self):
        """Test numeric 0 with == becomes False."""
        rule = '(X["flag"] == 0)'
        result = format_as_boolean_conditions(rule, ["flag"])
        assert result == '(X["flag"] == False)'

    def test_numeric_zero_not_equals(self):
        """Test numeric 0 with != becomes True."""
        rule = '(X["flag"] != 0)'
        result = format_as_boolean_conditions(rule, ["flag"])
        assert result == '(X["flag"] == True)'

    def test_empty_bool_columns(self):
        """Test with empty bool_columns list."""
        rule = '(X["col"] == "True")'
        result = format_as_boolean_conditions(rule, [])
        assert result == '(X["col"] == "True")'

    def test_column_not_in_bool_columns(self):
        """Test column not in bool_columns is unchanged."""
        rule = '(X["flag"] == "True") & (X["other"] == "True")'
        result = format_as_boolean_conditions(rule, ["flag"])
        assert result == '(X["flag"] == True) & (X["other"] == "True")'

    def test_non_boolean_value_unchanged(self):
        """Test non-boolean value is unchanged."""
        rule = '(X["flag"] == "Other")'
        result = format_as_boolean_conditions(rule, ["flag"])
        assert result == '(X["flag"] == "Other")'

    def test_mixed_conditions(self):
        """Test mixed boolean and non-boolean conditions."""
        rule = '(X["flag"] == "True") & (X["count"] > 5) & (X["active"] != 0)'
        result = format_as_boolean_conditions(rule, ["flag", "active"])
        assert result == '(X["flag"] == True) & (X["count"] > 5) & (X["active"] == True)'

    def test_single_quotes(self):
        """Test with single quotes around column names."""
        rule = "(X['flag'] == 'True') & (X['active'] != 'False')"
        result = format_as_boolean_conditions(rule, ["flag", "active"])
        assert result == '(X["flag"] == True) & (X["active"] == True)'

    def test_string_one_representation(self):
        """Test string '1' is treated as True."""
        rule = '(X["flag"] == "1")'
        result = format_as_boolean_conditions(rule, ["flag"])
        assert result == '(X["flag"] == True)'

    def test_string_zero_representation(self):
        """Test string '0' is treated as False."""
        rule = '(X["flag"] == "0")'
        result = format_as_boolean_conditions(rule, ["flag"])
        assert result == '(X["flag"] == False)'

    def test_all_true_variants(self):
        """Test all variants of True are handled."""
        rule = '(X["a"] == "True") & (X["b"] == "true") & (X["c"] == 1)'
        result = format_as_boolean_conditions(rule, ["a", "b", "c"])
        assert result == '(X["a"] == True) & (X["b"] == True) & (X["c"] == True)'

    def test_all_false_variants(self):
        """Test all variants of False are handled."""
        rule = '(X["a"] == "False") & (X["b"] == "false") & (X["c"] == 0)'
        result = format_as_boolean_conditions(rule, ["a", "b", "c"])
        assert result == '(X["a"] == False) & (X["b"] == False) & (X["c"] == False)'


class TestConvertFloatToIntEdgeCases:
    """Additional test cases for format_floats_as_integers to achieve 100% coverage."""

    def test_equality_operator_unchanged(self):
        """Test that == operator is not converted (line 196 coverage)."""
        rule = '(X["id"] == 5.0)'
        result = format_floats_as_integers(rule, ["id"])
        # Should remain unchanged as == is not one of the comparison operators
        assert result == '(X["id"] == 5.0)'

    def test_not_equal_operator_unchanged(self):
        """Test that != operator is not converted (line 196 coverage)."""
        rule = '(X["id"] != 3.5)'
        result = format_floats_as_integers(rule, ["id"])
        # Should remain unchanged as != is not one of the comparison operators
        assert result == '(X["id"] != 3.5)'

    def test_non_numeric_value_unchanged(self):
        """Test non-numeric value with a convertible operator is left unchanged."""
        rule = '(X["id"] >= "abc")'
        result = format_floats_as_integers(rule, ["id"])
        assert result == '(X["id"] >= "abc")'



class TestDecodeNumericConditionsEdgeCases:
    """Additional test cases for decode_numeric_encodings to achieve 100% coverage."""

    def test_non_numeric_value_in_mapping(self):
        """Test non-numeric value when column is in mapping (lines 332-334 coverage)."""
        mapping = {"A": {"a": 1, "b": 2, "c": 3}}
        rule = '(X["A"] >= "some_string")'
        result = decode_numeric_encodings(rule, mapping)
        # Should remain unchanged because value is not numeric
        assert result == '(X["A"] >= "some_string")'

    def test_no_categories_match(self):
        """Test when no categories satisfy the condition (line 364 coverage)."""
        mapping = {"A": {"a": 1, "b": 2, "c": 3}}
        # All values are <= 3, so no values satisfy > 10
        rule = '(X["A"] > 10)'
        result = decode_numeric_encodings(rule, mapping)
        # Should remain unchanged when no categories match
        assert result == '(X["A"] > 10)'


class TestDecodeOnehotEncodings:
    """Test cases for decode_onehot_encodings function."""

    def test_equal_category(self):
        """Test >= 0.5 split becomes an equality condition."""
        mapping = {"status__active": ("status", "active")}
        rule = '(X["status__active"] >= 0.5)'
        result = decode_onehot_encodings(rule, mapping)
        assert result == '(X["status"] == "active")'

    def test_not_equal_category(self):
        """Test < 0.5 split becomes a not-equal condition."""
        mapping = {"status__active": ("status", "active")}
        rule = '(X["status__active"] < 0.5)'
        result = decode_onehot_encodings(rule, mapping)
        assert result == '(X["status"] != "active")'

    def test_greater_than_treated_as_equal(self):
        """Test > 0.5 is treated the same as >= 0.5."""
        mapping = {"status__active": ("status", "active")}
        rule = '(X["status__active"] > 0.5)'
        result = decode_onehot_encodings(rule, mapping)
        assert result == '(X["status"] == "active")'

    def test_null_category_greater_equal(self):
        """Test the designated null category renders as is_null()."""
        mapping = {"status__missing": ("status", "MISSING")}
        rule = '(X["status__missing"] >= 0.5)'
        result = decode_onehot_encodings(rule, mapping, null_category="MISSING")
        assert result == 'X["status"].is_null()'

    def test_null_category_less_than(self):
        """Test the designated null category renders as ~is_null()."""
        mapping = {"status__missing": ("status", "MISSING")}
        rule = '(X["status__missing"] < 0.5)'
        result = decode_onehot_encodings(rule, mapping, null_category="MISSING")
        assert result == '(~X["status"].is_null())'

    def test_null_category_equals(self):
        """Test == 1 on the null category renders as is_null()."""
        mapping = {"status__missing": ("status", "MISSING")}
        rule = '(X["status__missing"] == 1)'
        result = decode_onehot_encodings(rule, mapping, null_category="MISSING")
        assert result == 'X["status"].is_null()'

    def test_null_category_not_equals(self):
        """Test != 1 on the null category renders as ~is_null()."""
        mapping = {"status__missing": ("status", "MISSING")}
        rule = '(X["status__missing"] != 1)'
        result = decode_onehot_encodings(rule, mapping, null_category="MISSING")
        assert result == '(~X["status"].is_null())'

    def test_column_not_in_mapping(self):
        """Test column not in mapping is unchanged."""
        mapping = {"status__active": ("status", "active")}
        rule = '(X["other"] >= 0.5)'
        result = decode_onehot_encodings(rule, mapping)
        assert result == '(X["other"] >= 0.5)'

    def test_non_numeric_value_unchanged(self):
        """Test non-numeric value is unchanged."""
        mapping = {"status__active": ("status", "active")}
        rule = '(X["status__active"] >= "abc")'
        result = decode_onehot_encodings(rule, mapping)
        assert result == '(X["status__active"] >= "abc")'

    def test_null_category_unsupported_operator(self):
        """Test an unsupported operator on the null category is unchanged."""
        mapping = {"status__missing": ("status", "MISSING")}
        rule = '(X["status__missing"] <= 2.0)'
        result = decode_onehot_encodings(rule, mapping, null_category="MISSING")
        assert result == '(X["status__missing"] <= 2.0)'


class TestDecodeNullIndicators:
    """Test cases for decode_null_indicators function."""

    def test_greater_equal_becomes_is_null(self):
        """Test >= 0.5 split becomes is_null()."""
        mapping = {"amount__is_null": "amount"}
        rule = '(X["amount__is_null"] >= 0.5)'
        result = decode_null_indicators(rule, mapping)
        assert result == 'X["amount"].is_null()'

    def test_less_than_becomes_not_is_null(self):
        """Test < 0.5 split becomes ~is_null()."""
        mapping = {"amount__is_null": "amount"}
        rule = '(X["amount__is_null"] < 0.5)'
        result = decode_null_indicators(rule, mapping)
        assert result == '(~X["amount"].is_null())'

    def test_column_not_in_mapping(self):
        """Test column not in mapping is unchanged."""
        mapping = {"amount__is_null": "amount"}
        rule = '(X["other"] >= 0.5)'
        result = decode_null_indicators(rule, mapping)
        assert result == '(X["other"] >= 0.5)'

    def test_non_numeric_value_unchanged(self):
        """Test non-numeric value is unchanged."""
        mapping = {"amount__is_null": "amount"}
        rule = '(X["amount__is_null"] >= "abc")'
        result = decode_null_indicators(rule, mapping)
        assert result == '(X["amount__is_null"] >= "abc")'

    def test_unsupported_operator_unchanged(self):
        """Test an unsupported operator is unchanged."""
        mapping = {"amount__is_null": "amount"}
        rule = '(X["amount__is_null"] == 0.5)'
        result = decode_null_indicators(rule, mapping)
        assert result == '(X["amount__is_null"] == 0.5)'


class TestQuoteStringValues:
    """Test cases for quote_string_values function."""

    def test_bare_value_quoted(self):
        """Test that a bare value is wrapped in quotes."""
        rule = '(X["col"] == retail)'
        result = quote_string_values(rule, ["col"])
        assert result == '(X["col"] == "retail")'

    def test_already_quoted_unchanged(self):
        """Test that an already-quoted value is unchanged."""
        rule = '(X["col"] == "retail")'
        result = quote_string_values(rule, ["col"])
        assert result == '(X["col"] == "retail")'

    def test_column_not_in_list_unchanged(self):
        """Test column not in columns is unchanged."""
        rule = '(X["other"] == retail)'
        result = quote_string_values(rule, ["col"])
        assert result == '(X["other"] == retail)'


class TestRoundThresholds:
    """Test cases for round_thresholds function."""

    def test_default_ndigits(self):
        """Test rounding to the default 2 decimal places."""
        rule = '(X["amount"] >= 1234.56789)'
        result = round_thresholds(rule, ["amount"])
        assert result == '(X["amount"] >= 1234.57)'

    def test_custom_ndigits(self):
        """Test rounding to a custom number of decimal places."""
        rule = '(X["amount"] >= 1.23456)'
        result = round_thresholds(rule, ["amount"], ndigits=3)
        assert result == '(X["amount"] >= 1.235)'

    def test_column_not_in_list_unchanged(self):
        """Test column not in columns is unchanged."""
        rule = '(X["other"] >= 1.23456)'
        result = round_thresholds(rule, ["amount"])
        assert result == '(X["other"] >= 1.23456)'

    def test_non_numeric_value_unchanged(self):
        """Test non-numeric value is unchanged."""
        rule = '(X["amount"] == "abc")'
        result = round_thresholds(rule, ["amount"])
        assert result == '(X["amount"] == "abc")'


class TestDropNullClauses:
    """Test cases for drop_null_clauses function."""

    def test_strip_is_null_clause(self):
        """Test that the is_null clause is stripped."""
        rule = '((X["amount"] >= 5.0) | X["amount"].is_null())'
        result = drop_null_clauses(rule, ["amount"])
        assert result == '(X["amount"] >= 5.0)'

    def test_column_not_in_list_unchanged(self):
        """Test column not in columns is unchanged."""
        rule = '((X["amount"] >= 5.0) | X["amount"].is_null())'
        result = drop_null_clauses(rule, ["other"])
        assert result == '((X["amount"] >= 5.0) | X["amount"].is_null())'

    def test_multiple_conditions(self):
        """Test stripping applies within a larger rule."""
        rule = '((X["a"] >= 5.0) | X["a"].is_null()) & (X["b"] < 1.0)'
        result = drop_null_clauses(rule, ["a"])
        assert result == '(X["a"] >= 5.0) & (X["b"] < 1.0)'


class TestDropNotNullConditions:
    """Test cases for drop_not_null_conditions function."""

    def test_drop_trailing_condition(self):
        """Test dropping a not-null condition at the end of a rule."""
        rule = '(X["a"] > 1) & (~X["b"].is_null())'
        result = drop_not_null_conditions(rule, ["b"])
        assert result == '(X["a"] > 1)'

    def test_drop_leading_condition(self):
        """Test dropping a not-null condition at the start of a rule."""
        rule = '(~X["b"].is_null()) & (X["a"] > 1)'
        result = drop_not_null_conditions(rule, ["b"])
        assert result == '(X["a"] > 1)'

    def test_column_not_in_list_unchanged(self):
        """Test column not in columns is unchanged."""
        rule = '(X["a"] > 1) & (~X["b"].is_null())'
        result = drop_not_null_conditions(rule, ["other"])
        assert result == '(X["a"] > 1) & (~X["b"].is_null())'


class TestPrettifyRules:
    """Test cases for prettify_rules function."""

    def test_single_step(self):
        """Test applying a single transformation step."""
        result = prettify_rules(
            ['(X["amount"] >= 1.23456)'],
            steps=[lambda r: round_thresholds(r, ["amount"])],
        )
        assert result == ['(X["amount"] >= 1.23)']

    def test_multiple_steps_applied_in_order(self):
        """Test that steps are applied in the given order."""
        mapping = {"A": {"x": 1, "y": 2}}
        result = prettify_rules(
            ['(X["A"] >= 2) & (X["amount"] > 1.239)'],
            steps=[
                lambda r: decode_numeric_encodings(r, mapping),
                lambda r: round_thresholds(r, ["amount"]),
            ],
        )
        assert result == ['(X["A"] == "y") & (X["amount"] > 1.24)']

    def test_column_name_mapping_applied_last(self):
        """Test that column_name_mapping renames columns after all steps."""
        result = prettify_rules(
            ['(X["amt"] > 5)'],
            steps=[],
            column_name_mapping={"amt": "amount"},
        )
        assert result == ['(X["amount"] > 5)']

    def test_multiple_rules(self):
        """Test that each rule in the list is processed independently."""
        result = prettify_rules(
            ['(X["a"] > 1)', '(X["b"] < 2)'],
            steps=[],
        )
        assert result == ['(X["a"] > 1)', '(X["b"] < 2)']


class TestDecodeStringImputation:
    """Test cases for decode_string_imputation function."""

    def test_equals_placeholder_becomes_is_null(self):
        """Test that an equality on the placeholder becomes is_null()."""
        rule = '(X["status"] == "MISSING")'
        result = decode_string_imputation(rule, {"status": "MISSING"})
        assert result == 'X["status"].is_null()'

    def test_not_equals_placeholder_becomes_not_is_null(self):
        """Test that a not-equal on the placeholder becomes ~is_null()."""
        rule = '(X["status"] != "MISSING")'
        result = decode_string_imputation(rule, {"status": "MISSING"})
        assert result == '(~X["status"].is_null())'

    def test_column_not_in_mapping_unchanged(self):
        """Test column not in mapping is unchanged."""
        rule = '(X["other"] == "MISSING")'
        result = decode_string_imputation(rule, {"status": "MISSING"})
        assert result == '(X["other"] == "MISSING")'

    def test_non_placeholder_value_unchanged(self):
        """Test a value other than the placeholder is unchanged."""
        rule = '(X["status"] == "active")'
        result = decode_string_imputation(rule, {"status": "MISSING"})
        assert result == '(X["status"] == "active")'

    def test_unsupported_operator_unchanged(self):
        """Test an unsupported operator is unchanged."""
        rule = '(X["status"] > "MISSING")'
        result = decode_string_imputation(rule, {"status": "MISSING"})
        assert result == '(X["status"] > "MISSING")'


class TestDecodeDiscretizedBins:
    """Test cases for decode_discretized_bins function."""

    def test_greater_equal_bin(self):
        """Test >= bin index decodes to the bin's lower edge."""
        rule = '(X["amount"] >= 2)'
        result = decode_discretized_bins(rule, {"amount": [0, 10, 50, 200]})
        assert result == '(X["amount"] >= 50)'

    def test_less_than_bin(self):
        """Test < bin index decodes to the bin's lower edge."""
        rule = '(X["amount"] < 2)'
        result = decode_discretized_bins(rule, {"amount": [0, 10, 50, 200]})
        assert result == '(X["amount"] < 50)'

    def test_non_integer_threshold_uses_ceiling(self):
        """Test that a non-integer bin threshold is ceiling-rounded first."""
        rule = '(X["amount"] >= 1.5)'
        result = decode_discretized_bins(rule, {"amount": [0, 10, 50, 200]})
        assert result == '(X["amount"] >= 50)'

    def test_column_not_in_mapping_unchanged(self):
        """Test column not in mapping is unchanged."""
        rule = '(X["other"] >= 2)'
        result = decode_discretized_bins(rule, {"amount": [0, 10, 50, 200]})
        assert result == '(X["other"] >= 2)'

    def test_out_of_range_bin_unchanged(self):
        """Test a bin index outside the edges range is unchanged."""
        rule = '(X["amount"] >= 10)'
        result = decode_discretized_bins(rule, {"amount": [0, 10, 50, 200]})
        assert result == '(X["amount"] >= 10)'

    def test_unsupported_operator_unchanged(self):
        """Test an unsupported operator is unchanged."""
        rule = '(X["amount"] == 2)'
        result = decode_discretized_bins(rule, {"amount": [0, 10, 50, 200]})
        assert result == '(X["amount"] == 2)'

    def test_non_numeric_value_unchanged(self):
        """Test non-numeric value is unchanged."""
        rule = '(X["amount"] >= "abc")'
        result = decode_discretized_bins(rule, {"amount": [0, 10, 50, 200]})
        assert result == '(X["amount"] >= "abc")'


class TestDecodeScaledThresholds:
    """Test cases for decode_scaled_thresholds function."""

    def test_linear_inverse(self):
        """Test a simple linear inverse transform."""
        rule = '(X["amount"] >= 5.0)'
        result = decode_scaled_thresholds(rule, {"amount": lambda x: x * 2})
        assert result == '(X["amount"] >= 10.0)'

    def test_operator_preserved(self):
        """Test that the original operator is kept as-is."""
        rule = '(X["amount"] < 5.0)'
        result = decode_scaled_thresholds(rule, {"amount": lambda x: x * 2})
        assert result == '(X["amount"] < 10.0)'

    def test_column_not_in_mapping_unchanged(self):
        """Test column not in mapping is unchanged."""
        rule = '(X["other"] >= 5.0)'
        result = decode_scaled_thresholds(rule, {"amount": lambda x: x * 2})
        assert result == '(X["other"] >= 5.0)'

    def test_non_numeric_value_unchanged(self):
        """Test non-numeric value is unchanged."""
        rule = '(X["amount"] >= "abc")'
        result = decode_scaled_thresholds(rule, {"amount": lambda x: x * 2})
        assert result == '(X["amount"] >= "abc")'




