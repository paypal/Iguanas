from iguanas.rule_formatting import (
    # add_missing_value_conditions,
    # format_floats_as_integers,
    # format_as_boolean_conditions,
    # decode_numeric_encodings,
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


# class TestConvertFloatToInt:
#     """Test cases for format_floats_as_integers function."""

#     def test_user_example(self):
#         """Test the exact user example."""
#         rule = '(X["a"] >= 0.1) & (X["b"] >= 9.1)'
#         result = format_floats_as_integers(rule, ["a"])
#         assert result == '(X["a"] >= 1) & (X["b"] >= 9.1)'

#     def test_greater_equal_operator(self):
#         """Test >= operator uses ceiling."""
#         rule = '(X["a"] >= 0.1)'
#         result = format_floats_as_integers(rule, ["a"])
#         assert result == '(X["a"] >= 1)'

#     def test_greater_operator(self):
#         """Test > operator uses floor."""
#         rule = '(X["a"] > 0.9)'
#         result = format_floats_as_integers(rule, ["a"])
#         assert result == '(X["a"] > 0)'

#     def test_less_equal_operator(self):
#         """Test <= operator uses floor."""
#         rule = '(X["a"] <= 9.9)'
#         result = format_floats_as_integers(rule, ["a"])
#         assert result == '(X["a"] <= 9)'

#     def test_less_operator(self):
#         """Test < operator uses ceiling."""
#         rule = '(X["a"] < 9.1)'
#         result = format_floats_as_integers(rule, ["a"])
#         assert result == '(X["a"] < 10)'

#     def test_multiple_columns(self):
#         """Test conversion on multiple columns."""
#         rule = '(X["a"] > 0.1) & (X["b"] < 10.9)'
#         result = format_floats_as_integers(rule, ["a", "b"])
#         assert result == '(X["a"] > 0) & (X["b"] < 11)'

#     def test_selective_conversion(self):
#         """Test conversion only happens for specified columns."""
#         rule = '(X["a"] >= 0.5) & (X["b"] >= 0.5) & (X["c"] >= 0.5)'
#         result = format_floats_as_integers(rule, ["a", "c"])
#         assert result == '(X["a"] >= 1) & (X["b"] >= 0.5) & (X["c"] >= 1)'

#     def test_already_integer_values(self):
#         """Test that integer values remain unchanged."""
#         rule = '(X["a"] >= 1) & (X["b"] < 10)'
#         result = format_floats_as_integers(rule, ["a", "b"])
#         assert result == '(X["a"] >= 1) & (X["b"] < 10)'

#     def test_empty_int_columns(self):
#         """Test with empty int_columns list."""
#         rule = '(X["a"] >= 0.5)'
#         result = format_floats_as_integers(rule, [])
#         assert result == '(X["a"] >= 0.5)'

#     def test_non_numeric_values_unchanged(self):
#         """Test that non-numeric values are not converted."""
#         rule = '(X["name"] == "test") & (X["a"] >= 0.5)'
#         result = format_floats_as_integers(rule, ["name", "a"])
#         assert result == '(X["name"] == "test") & (X["a"] >= 1)'

#     def test_negative_values(self):
#         """Test conversion with negative values."""
#         rule = '(X["a"] >= -0.5) & (X["b"] < -2.3)'
#         result = format_floats_as_integers(rule, ["a", "b"])
#         assert result == '(X["a"] >= 0) & (X["b"] < -2)'


# class TestAddNullConditions:
#     """Test cases for add_missing_value_conditions function."""

#     def test_user_example(self):
#         """Test the exact user example."""
#         mapping = {"a": 0, "b": 0.3, "c": 100}
#         rule = '(X["a"] < 1) & (X["b"] >= 3) & (X["c"] > 10)'
#         result = add_missing_value_conditions(rule, mapping)
#         assert (
#             result
#             == '(X["a"] < 1 | X["a"].isnull()) & (X["b"] >= 3) & (X["c"] > 10 | X["c"].isnull())'
#         )

#     def test_single_quotes(self):
#         """Test that single quotes are handled correctly."""
#         mapping = {"a": 0}
#         rule = "(X['a'] < 1)"
#         result = add_missing_value_conditions(rule, mapping)
#         assert result == '(X["a"] < 1 | X["a"].isnull())'

#     def test_greater_equal_condition_satisfied(self):
#         """Test >= condition satisfied by nan value."""
#         mapping = {"a": 5}
#         rule = '(X["a"] >= 3)'
#         result = add_missing_value_conditions(rule, mapping)
#         assert result == '(X["a"] >= 3 | X["a"].isnull())'

#     def test_greater_equal_condition_not_satisfied(self):
#         """Test >= condition not satisfied by nan value."""
#         mapping = {"a": 5}
#         rule = '(X["a"] >= 10)'
#         result = add_missing_value_conditions(rule, mapping)
#         assert result == '(X["a"] >= 10)'

#     def test_greater_condition_satisfied(self):
#         """Test > condition satisfied by nan value."""
#         mapping = {"a": 5}
#         rule = '(X["a"] > 3)'
#         result = add_missing_value_conditions(rule, mapping)
#         assert result == '(X["a"] > 3 | X["a"].isnull())'

#     def test_less_equal_condition_satisfied(self):
#         """Test <= condition satisfied by nan value."""
#         mapping = {"a": 5}
#         rule = '(X["a"] <= 10)'
#         result = add_missing_value_conditions(rule, mapping)
#         assert result == '(X["a"] <= 10 | X["a"].isnull())'

#     def test_less_condition_satisfied(self):
#         """Test < condition satisfied by nan value."""
#         mapping = {"a": 5}
#         rule = '(X["a"] < 10)'
#         result = add_missing_value_conditions(rule, mapping)
#         assert result == '(X["a"] < 10 | X["a"].isnull())'

#     def test_equality_condition_satisfied(self):
#         """Test == condition satisfied by nan value."""
#         mapping = {"a": 0}
#         rule = '(X["a"] == 0)'
#         result = add_missing_value_conditions(rule, mapping)
#         assert result == '(X["a"] == 0 | X["a"].isnull())'

#     def test_not_equal_condition_satisfied(self):
#         """Test != condition satisfied by nan value."""
#         mapping = {"a": 0}
#         rule = '(X["a"] != 5)'
#         result = add_missing_value_conditions(rule, mapping)
#         assert result == '(X["a"] != 5 | X["a"].isnull())'

#     def test_multiple_conditions_mixed(self):
#         """Test multiple conditions with some satisfied and some not."""
#         mapping = {"a": 5, "b": 10}
#         rule = '(X["a"] <= 10) & (X["b"] > 5)'
#         result = add_missing_value_conditions(rule, mapping)
#         assert result == '(X["a"] <= 10 | X["a"].isnull()) & (X["b"] > 5 | X["b"].isnull())'

#     def test_empty_mapping(self):
#         """Test with empty mapping."""
#         rule = '(X["a"] < 1)'
#         result = add_missing_value_conditions(rule, {})
#         assert result == '(X["a"] < 1)'

#     def test_column_not_in_mapping(self):
#         """Test column not in mapping is unchanged."""
#         mapping = {"a": 0}
#         rule = '(X["a"] < 1) & (X["b"] < 5)'
#         result = add_missing_value_conditions(rule, mapping)
#         assert result == '(X["a"] < 1 | X["a"].isnull()) & (X["b"] < 5)'

#     def test_non_numeric_value_unchanged(self):
#         """Test non-numeric comparison is unchanged."""
#         mapping = {"a": 0, "name": "test"}
#         rule = '(X["a"] < 1) & (X["name"] == "John")'
#         result = add_missing_value_conditions(rule, mapping)
#         assert result == '(X["a"] < 1 | X["a"].isnull()) & (X["name"] == "John")'

#     def test_float_nan_values(self):
#         """Test with float nan replacement values."""
#         mapping = {"a": 0.5, "b": 1.5}
#         rule = '(X["a"] < 1) & (X["b"] >= 1)'
#         result = add_missing_value_conditions(rule, mapping)
#         assert result == '(X["a"] < 1 | X["a"].isnull()) & (X["b"] >= 1 | X["b"].isnull())'


# class TestDecodeNumericConditions:
#     """Test cases for decode_numeric_encodings function."""

#     def test_user_example(self):
#         """Test the exact user example."""
#         mapping = {"A": {"a": 1, "b": 2, "c": 3}, "B": {"a": -8.1, "b": 1.1, "c": 3}}
#         rule = '(X["A"] >= 2) & (X["B"] < 0)'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["A"].is_in(["b", "c"])) & (X["B"] == "a")'

#     def test_single_match_equality(self):
#         """Test that single matching category uses == operator."""
#         mapping = {"A": {"x": 1, "y": 2}}
#         rule = '(X["A"] == 1)'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["A"] == "x")'

#     def test_multiple_matches_isin(self):
#         """Test that multiple matching categories use .is_in()."""
#         mapping = {"col": {"low": 1, "med": 5, "high": 10}}
#         rule = '(X["col"] > 3)'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["col"].is_in(["med", "high"]))'

#     def test_greater_than_operator(self):
#         """Test > operator finds correct categories."""
#         mapping = {"A": {"a": 1, "b": 5, "c": 10}}
#         rule = '(X["A"] > 4)'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["A"].is_in(["b", "c"]))'

#     def test_greater_equal_operator(self):
#         """Test >= operator finds correct categories."""
#         mapping = {"A": {"a": 1, "b": 5, "c": 10}}
#         rule = '(X["A"] >= 5)'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["A"].is_in(["b", "c"]))'

#     def test_less_than_operator(self):
#         """Test < operator finds correct categories."""
#         mapping = {"A": {"a": 1, "b": 5, "c": 10}}
#         rule = '(X["A"] < 6)'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["A"].is_in(["a", "b"]))'

#     def test_less_equal_operator(self):
#         """Test <= operator finds correct categories."""
#         mapping = {"A": {"a": 1, "b": 5, "c": 10}}
#         rule = '(X["A"] <= 5)'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["A"].is_in(["a", "b"]))'

#     def test_not_equal_operator(self):
#         """Test != operator finds correct categories."""
#         mapping = {"col": {"low": 1, "med": 5, "high": 10}}
#         rule = '(X["col"] != 5)'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["col"].is_in(["low", "high"]))'

#     def test_multiple_conditions(self):
#         """Test multiple conditions are all decoded."""
#         mapping = {"A": {"a": 1, "b": 2}, "B": {"x": 10, "y": 20}}
#         rule = '(X["A"] >= 2) & (X["B"] < 15)'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["A"] == "b") & (X["B"] == "x")'

#     def test_empty_mapping(self):
#         """Test with empty mapping."""
#         rule = '(X["A"] >= 2)'
#         result = decode_numeric_encodings(rule, {})
#         assert result == '(X["A"] >= 2)'

#     def test_column_not_in_mapping(self):
#         """Test column not in mapping is unchanged."""
#         mapping = {"A": {"a": 1, "b": 2}}
#         rule = '(X["A"] >= 2) & (X["B"] < 5)'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["A"] == "b") & (X["B"] < 5)'

#     def test_negative_encoded_values(self):
#         """Test with negative encoded values."""
#         mapping = {"A": {"neg": -5, "zero": 0, "pos": 5}}
#         rule = '(X["A"] < 0)'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["A"] == "neg")'

#     def test_float_encoded_values(self):
#         """Test with float encoded values."""
#         mapping = {"A": {"low": 0.5, "mid": 1.5, "high": 2.5}}
#         rule = '(X["A"] >= 1.0)'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["A"].is_in(["mid", "high"]))'

#     def test_all_categories_match(self):
#         """Test when all categories satisfy the condition."""
#         mapping = {"A": {"a": 1, "b": 2, "c": 3}}
#         rule = '(X["A"] >= 0)'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["A"].is_in(["a", "b", "c"]))'

#     def test_single_quotes_input(self):
#         """Test with single quotes in input."""
#         mapping = {"A": {"a": 1, "b": 2}}
#         rule = "(X['A'] >= 2)"
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["A"] == "b")'

#     def test_non_numeric_value_unchanged(self):
#         """Test non-numeric comparison is unchanged."""
#         mapping = {"A": {"a": 1, "b": 2}}
#         rule = '(X["A"] >= 2) & (X["name"] == "test")'
#         result = decode_numeric_encodings(rule, mapping)
#         assert result == '(X["A"] == "b") & (X["name"] == "test")'


# class TestConvertToBool:
#     """Test cases for format_as_boolean_conditions function."""

#     def test_user_example_true_with_equals(self):
#         """Test True with == operator."""
#         rule = '(X["flag"] == "True") & (X["active"] != "False")'
#         result = format_as_boolean_conditions(rule, ["flag", "active"])
#         assert result == '(X["flag"] == True) & (X["active"] == True)'

#     def test_user_example_numeric(self):
#         """Test numeric 1 and 0 representations."""
#         rule = '(X["enabled"] != 1) & (X["disabled"] == 0)'
#         result = format_as_boolean_conditions(rule, ["enabled", "disabled"])
#         assert result == '(X["enabled"] == False) & (X["disabled"] == False)'

#     def test_lowercase_true_false(self):
#         """Test lowercase true/false strings."""
#         rule = '(X["is_valid"] == "true") & (X["is_ready"] != "false")'
#         result = format_as_boolean_conditions(rule, ["is_valid", "is_ready"])
#         assert result == '(X["is_valid"] == True) & (X["is_ready"] == True)'

#     def test_true_equals_becomes_true(self):
#         """Test True with == becomes True."""
#         rule = '(X["col"] == "True")'
#         result = format_as_boolean_conditions(rule, ["col"])
#         assert result == '(X["col"] == True)'

#     def test_true_not_equals_becomes_false(self):
#         """Test True with != becomes False (with ==)."""
#         rule = '(X["col"] != "True")'
#         result = format_as_boolean_conditions(rule, ["col"])
#         assert result == '(X["col"] == False)'

#     def test_false_equals_becomes_false(self):
#         """Test False with == becomes False."""
#         rule = '(X["col"] == "False")'
#         result = format_as_boolean_conditions(rule, ["col"])
#         assert result == '(X["col"] == False)'

#     def test_false_not_equals_becomes_true(self):
#         """Test False with != becomes True (with ==)."""
#         rule = '(X["col"] != "False")'
#         result = format_as_boolean_conditions(rule, ["col"])
#         assert result == '(X["col"] == True)'

#     def test_numeric_one_equals(self):
#         """Test numeric 1 with == becomes True."""
#         rule = '(X["flag"] == 1)'
#         result = format_as_boolean_conditions(rule, ["flag"])
#         assert result == '(X["flag"] == True)'

#     def test_numeric_one_not_equals(self):
#         """Test numeric 1 with != becomes False."""
#         rule = '(X["flag"] != 1)'
#         result = format_as_boolean_conditions(rule, ["flag"])
#         assert result == '(X["flag"] == False)'

#     def test_numeric_zero_equals(self):
#         """Test numeric 0 with == becomes False."""
#         rule = '(X["flag"] == 0)'
#         result = format_as_boolean_conditions(rule, ["flag"])
#         assert result == '(X["flag"] == False)'

#     def test_numeric_zero_not_equals(self):
#         """Test numeric 0 with != becomes True."""
#         rule = '(X["flag"] != 0)'
#         result = format_as_boolean_conditions(rule, ["flag"])
#         assert result == '(X["flag"] == True)'

#     def test_empty_bool_columns(self):
#         """Test with empty bool_columns list."""
#         rule = '(X["col"] == "True")'
#         result = format_as_boolean_conditions(rule, [])
#         assert result == '(X["col"] == "True")'

#     def test_column_not_in_bool_columns(self):
#         """Test column not in bool_columns is unchanged."""
#         rule = '(X["flag"] == "True") & (X["other"] == "True")'
#         result = format_as_boolean_conditions(rule, ["flag"])
#         assert result == '(X["flag"] == True) & (X["other"] == "True")'

#     def test_non_boolean_value_unchanged(self):
#         """Test non-boolean value is unchanged."""
#         rule = '(X["flag"] == "Other")'
#         result = format_as_boolean_conditions(rule, ["flag"])
#         assert result == '(X["flag"] == "Other")'

#     def test_mixed_conditions(self):
#         """Test mixed boolean and non-boolean conditions."""
#         rule = '(X["flag"] == "True") & (X["count"] > 5) & (X["active"] != 0)'
#         result = format_as_boolean_conditions(rule, ["flag", "active"])
#         assert result == '(X["flag"] == True) & (X["count"] > 5) & (X["active"] == True)'

#     def test_single_quotes(self):
#         """Test with single quotes around column names."""
#         rule = "(X['flag'] == 'True') & (X['active'] != 'False')"
#         result = format_as_boolean_conditions(rule, ["flag", "active"])
#         assert result == '(X["flag"] == True) & (X["active"] == True)'

#     def test_string_one_representation(self):
#         """Test string '1' is treated as True."""
#         rule = '(X["flag"] == "1")'
#         result = format_as_boolean_conditions(rule, ["flag"])
#         assert result == '(X["flag"] == True)'

#     def test_string_zero_representation(self):
#         """Test string '0' is treated as False."""
#         rule = '(X["flag"] == "0")'
#         result = format_as_boolean_conditions(rule, ["flag"])
#         assert result == '(X["flag"] == False)'

#     def test_all_true_variants(self):
#         """Test all variants of True are handled."""
#         rule = '(X["a"] == "True") & (X["b"] == "true") & (X["c"] == 1)'
#         result = format_as_boolean_conditions(rule, ["a", "b", "c"])
#         assert result == '(X["a"] == True) & (X["b"] == True) & (X["c"] == True)'

#     def test_all_false_variants(self):
#         """Test all variants of False are handled."""
#         rule = '(X["a"] == "False") & (X["b"] == "false") & (X["c"] == 0)'
#         result = format_as_boolean_conditions(rule, ["a", "b", "c"])
#         assert result == '(X["a"] == False) & (X["b"] == False) & (X["c"] == False)'


# class TestConvertFloatToIntEdgeCases:
#     """Additional test cases for format_floats_as_integers to achieve 100% coverage."""

#     def test_equality_operator_unchanged(self):
#         """Test that == operator is not converted (line 196 coverage)."""
#         rule = '(X["id"] == 5.0)'
#         result = format_floats_as_integers(rule, ["id"])
#         # Should remain unchanged as == is not one of the comparison operators
#         assert result == '(X["id"] == 5.0)'

#     def test_not_equal_operator_unchanged(self):
#         """Test that != operator is not converted (line 196 coverage)."""
#         rule = '(X["id"] != 3.5)'
#         result = format_floats_as_integers(rule, ["id"])
#         # Should remain unchanged as != is not one of the comparison operators
#         assert result == '(X["id"] != 3.5)'


# class TestDecodeNumericConditionsEdgeCases:
#     """Additional test cases for decode_numeric_encodings to achieve 100% coverage."""

#     def test_non_numeric_value_in_mapping(self):
#         """Test non-numeric value when column is in mapping (lines 332-334 coverage)."""
#         mapping = {"A": {"a": 1, "b": 2, "c": 3}}
#         rule = '(X["A"] >= "some_string")'
#         result = decode_numeric_encodings(rule, mapping)
#         # Should remain unchanged because value is not numeric
#         assert result == '(X["A"] >= "some_string")'

#     def test_no_categories_match(self):
#         """Test when no categories satisfy the condition (line 364 coverage)."""
#         mapping = {"A": {"a": 1, "b": 2, "c": 3}}
#         # All values are <= 3, so no values satisfy > 10
#         rule = '(X["A"] > 10)'
#         result = decode_numeric_encodings(rule, mapping)
#         # Should remain unchanged when no categories match
#         assert result == '(X["A"] > 10)'
