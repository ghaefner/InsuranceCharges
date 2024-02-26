import pandas as pd
import pytest
from src.api import read_data
from pandas.testing import assert_frame_equal

@pytest.fixture
def mock_df():
    data = {
        "Name": ["Alice", "Bob"],
        "Age": [25, 30],
        "Smoker": ["yes", "no"]
    }
    return pd.DataFrame(data)

def test_read_data_valid():
    pass
