"""Class for converting between lambda and string representations of rules."""
from typing import Dict, Callable


class _ConvertRuleLambdasToRuleStrings:
    """
    Converts a set of rules (each being represented in the standard Iguanas 
    lambda expression format) into the standard Iguanas string format.   

    Parameters
    ----------
    rule_lambdas (Dict[str, Callable[[Dict], str]]): Set of rules defined 
        using the standard Iguanas lambda expression format (values) and their 
        names (keys).
    lambda_kwargs (Dict[str, Dict[str, float]]): For each rule (keys), a 
        dictionary containing the features used in the rule (keys) and the 
        current values (values). Defaults to None.             
    lambda_args (Dict[str, List[str]]): For each rule (keys), a list 
        containing the current values used in the rule. Defaults to None.        

    Raises
    ------
    Exception
        Either `lambda_kwargs` or `lambda_args` must be provided.

    Attributes
    ----------
    rule_strings : Dict[str, str]
        Set of rules defined using the standard Iguanas string format (values)
        and their names (keys).
    """

    def __init__(self,
                 rule_lambdas: Dict[str, Callable[[Dict], str]],
                 lambda_kwargs=None, lambda_args=None):
        if lambda_kwargs is None and lambda_args is None:
            raise ValueError(
                'Either `lambda_kwargs` or `lambda_args` must be provided'
            )
        self.rule_lambdas = rule_lambdas
        self.lambda_kwargs = {} if lambda_kwargs is None else lambda_kwargs
        self.lambda_args = {} if lambda_args is None else lambda_args
        self.rule_strings = {}

    def convert(self) -> Dict[str, str]:
        """
        Converts a set of rules (each being represented in the standard Iguanas 
        lambda expression format) into the standard Iguanas string format.   

        Returns
        -------
        Dict[str, str]
            Set of rules defined using the standard Iguanas string format.
        """
        for rule_name, rule_lambda in self.rule_lambdas.items():
            if self.lambda_kwargs != {}:
                rule_string = rule_lambda(**self.lambda_kwargs[rule_name])
            else:
                rule_string = rule_lambda(*self.lambda_args[rule_name])
            self.rule_strings[rule_name] = rule_string
        return self.rule_strings
