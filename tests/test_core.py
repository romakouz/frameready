# at the top of any notebook
from frameready.core import update_dtypes, transform_datetime, handle_missing, MissingMethod
import pandas as pd
import numpy as np
import pytest # used for testing

@pytest.fixture # used for testing

# Test code
def sample_df():
    return pd.DataFrame({
        'DOB': ['1950-03-23', '1978-12-21', '1989-04-05', '1999-06-17'],
        'Race': ["Asian", "Caucasian", pd.NA, "African-American"],
        'Sex': ["Male", "Female", "Male", "Male"],
        'Weight (kg)': ["72.2 kg", np.nan, "63.8 kg", "43.1kg"],
    })

def test_update_dtypes_continuous(sample_df):
    schema = {'Weight (kg)': "continuous"}
    result = update_dtypes(sample_df, schema, force_string_to_numeric=True)
    assert result['Weight (kg)'].dtype == "float64"
    assert result['Weight (kg)'].notna().any()

def test_update_dtypes_binary(sample_df):
    schema = {'Sex': ("binary", "Female")}
    result = update_dtypes(sample_df, schema)
    assert set(result['Sex'].dropna().unique()).issubset({0, 1})

def test_handle_missing_drops_rows(sample_df):
    schema: dict[str, MissingMethod] = {'Race': "drop"}
    result = handle_missing(sample_df, schema=schema)
    assert result['Race'].isna().sum() == 0
    assert len(result) < len(sample_df)

### Example run code
# create sample dataframe (replace with actual dataframe)
test = pd.DataFrame(
    {
        'DOB' : ['1950-03-23', '1978-12-21', '1989-04-05', '1999-06-17'],
        'Steatosis' : [0, 3, 2, 1],
        'Race' : ["Asian", "Caucasian", pd.NA, "African-American"],
        'Sex' : ["Male", "Female", "Male", "Male"],
        'Deceased' : ["2000-01-01", "No", "No", None],
        'PATId' : ["PAT1", "PAT2", "PAT3", "PAT4"],
        'Weight (kg)' : ["72.2 kg", np.nan, "63.8 kg", "43.1kg"],
        'Height (cm)' : [180.1,189.7,140.3, 128.9]
    }
)

# define schema
dtype_schema = {
    'DOB' : "datetime",
    'Steatosis' : "ordinal",
    'Race' : "categorical",
    'Sex' : ("binary", "Female"),
    'Deceased' : ("binary", lambda x: x != "No"),
    'PATId' : 'id',
    'Weight (kg)' : "continuous",
    'Height (cm)' : "float"
}
datetime_schema = {
    'DOB' : ("bin", [0,18,35,50,75, 200], ["0-18", "18-35", "35-50","50-75", "75+"], "year", "2010-01-01")
}

missing_schema = {
    'Race' : ("string", "Other"),
    'Weight (kg)' : "median",
    'Deceased' : "drop",
    'Steatosis' : "zero"
}

print("TEST DTYPES:")
print(test.dtypes)

# perform transformations
test_mapped = update_dtypes(test, dtype_schema, force_string_to_numeric = True)
test_mapped = transform_datetime(test_mapped, schema = datetime_schema)
test_mapped = handle_missing(test_mapped, schema = missing_schema)

print("TEST DTYPES POST MAPPING")
print(test_mapped.dtypes)
test_mapped.head()
