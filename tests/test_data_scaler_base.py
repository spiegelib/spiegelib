"""
Tests for DataScalerBase class
"""

import pytest
from spiegelib.features import DataScalerBase

class TestDataScalerBase():

    def test_data_scaler_base_construction(self):
        with pytest.raises(TypeError) as exc_info:
            dataScaler = DataScalerBase()
        assert exc_info.type is TypeError
        assert exc_info.value.args[0] == (
            "Can't instantiate abstract class "
            "DataScalerBase with abstract methods fit, transform"
        )
