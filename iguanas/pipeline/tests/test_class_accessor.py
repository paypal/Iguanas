import pytest
from iguanas.pipeline import ClassAccessor


def test_get():
    ca = ClassAccessor('sf', 'rules_to_keep')
    pipeline_params = {
        'sf': {
            'rules_to_keep': ['Rule1']
        }
    }
    assert ca.get(pipeline_params) == ['Rule1']


def test_error():
    ca = ClassAccessor('sf3', 'rules_to_keep')
    pipeline_params = {
        'sf': {
            'rules_to_keep': ['Rule1']
        }
    }
    with pytest.raises(ValueError, match="There are no steps in `pipeline` corresponding to `class_tag`='sf3'"):
        ca.get(pipeline_params)
