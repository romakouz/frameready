# Frameready
This package offers simple wrapper functions for preprocessing pandas DataFrames for machine learning and statistical analysis.

This is of general use for simplifying code extracted from medical information system, webscraping, and publicly available datasets.

## Current support
The package currently supports the following operations:
- Updating object dtypes to align with statistical data types, such as continuous, categorical, ordinal, binary, and various representations for datetime types.
- Handling simple missing value scenarios (record deletion, imputation, reformatting from merged excel files)
- Concatenating various csv files of the same format into a single dataframe

## Installation
```bash
pip3 install frameready 
``` 

## Import
```python
import frameready as fr
```
Alternatively, import specific functions
```python
from frameready import update_dtypes, transform_datetime, handle_missing
```

## Usage
This package currently only offers support for pandas dataframes, and functions primarliy by declaring schema specifying the various transformations to be performed on each column.

Guidance and examples on defining schema are provided in [Schema usage](#schema-usage).
Example code is provided [here](tests/test_core.py)

### Transformation order
After loading a dataset as a pandas dataframe and inspecting the dtypes and dataframe columns, it is recommended that transformations are applied in the following order:

1. Update column data types using `update_dtypes()` and a dtype schema.
2. Update any datetime objects using `transform_datetime()` and a datetime schema.
3. Update missing values using `handle_missing()` and a missing value method schema.
4. Visualize the updated dataframe to confirm transformations were perfomed as expected.

Additional support for inferring/generating schema, encoding dummies, and summarizing features will be provided in future versions (see [Future Support](#future-support))


### Schema usage
#### Dtype schema
For updating dtypes, specify the data type for each column in the dataframe for which you would like to change the data type. The binary and ordinal categorical ("ordinal_cat") data types require additional arguments for how you would like the transformation to be performed.

Default mappings to datetime and raw pandas dtypes are also supported.

```python
schema = {
    'DOB' : "datetime",
    'Steatosis' : ("ordinal_cat", [0, 1, 2, 3]),
    'Race' : "categorical",
    'Sex' : ("binary", "Female"),
    'Deceased' : ("binary", lambda x: x != "No"),
    'PATId' : 'id',
    'Weight (kg)' : "continuous",
    'Height (cm)' : "float"
}
```

#### Datetime schema
For updating datetime data types, specify the method for each column in the dataframe for which you would like to change the data type. The duration, extract, after, and bin methods require additional arguments related to the tranformation, including arguments such as time units, reference date, threshold, and—in the case of bin, which bins the data into categories based on cut-offs—bins and labels.

```python
schema1 = {
    'DOB' : ("ordinal")
}
schema2 = {
    'DOB' : ("duration", "year", "2000-01-01")
}
schema3 = {
    'DOB' : ("extract", ["year", "month", "day", "weekofyear"])
}
schema4 = {
    'DOB' : ("after", "2000-01-01")
}
schema5 = {
    'DOB' : ("bin", [0,18,35,50,75, 200], ["0-18", "18-35", "35-50","50-75", "75+"], "year", "2010-01-01")
}

```

#### Missing schema
For handling missing values, specify the method for each column in the dataframe for which you would like to fill/remove/impute missing values. The string method requires an additional arguments `defaultstring` that will replace any missing values in the column.

- An `ffill = True` argument can be specified to forward-fill <ins>all</ins> empty rows in the dataframe with the value in the same column in the closest preceding non-empty row. This is especially useful for panel/time-series data, or datasets imported from excel files with merged cells.

***Warning:*** Any columns using the missing value method "drop" will cause all rows with a missing value in that column to be dropped from the resulting dataframe. If multiple columns use "drop" the union of rows with missing values in any column will be dropped.

```python
schema = {
    'Race' : ("string", "Other"),
    'Weight' : "median",
    'Sex' : "drop",
    'Steatosis' : "zero",
    'Datetime column' : None
}
```

## Requirements
- Python 3.9+
- pandas
- numpy

## Future support
- encoding options (dummy/one-hot, ordinal, binary)
- heuristic data type inference
- more missing data imputation options (kNN, etc.)
- better dataset view/summary options, pre- and post-preprocessing

I essentially add to this package whenever I encounter a preprocessing task which is sufficiently repetitive and general while performing statistical analysis. 
Therefore, this package will be continuously enriched as I encounter new scenarios for simplifying code (time permitting, of course).