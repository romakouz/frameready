'''
frameready.py
================
Generalizable preprocessing functions for data analysis with pandas DataFrames.

Handles:
- datetime transformations
- updating dtypes (continuous, categorical, ordinal, binary, datetime)
- missing values
- encoding (dummy/one-hot, ordinal, binary)
- view/summarize daraset features

Usage
-----
import frameready as fr
# or, to get functions
from frameready import update_dtypes, transform_datetime, handle_missing

'''

from __future__ import annotations
import os 
import warnings
import pandas as pd
import numpy as np
from typing import Literal, Callable, Mapping # use Literal to force inputs to match prespecified list

####################################
# default dtypes and missing methods
FeatureType = Literal['continuous', 'categorical', 'binary', 'ordinal', 'datetime']
MissingMethod = Literal['zero', 'mean', 'median', 'mode', 'blank', 'string', 'drop', None]

#############################################
# internal language mapping to useful objects

# converting dtypes
_DTYPE_MAP: dict[str, str] = {
    "continuous" : "float64",
    "binary" : "Int64",
    "ordinal" : "Int64",
    "ordinal_cat" : "ordinal_cat", # used for ordered categorical representation needed for ordinal logistic regression
    "categorical" : "category",
    "datetime" : "datetime",
    "id" : "string"
}
_ALIAS_MAP: dict[str, str] = {
    "float"  : "float64",
    "int"    : "Int64",
    "int64"  : "Int64",
}
_RAW_DTYPE_STRINGS = {"float64", "float32", "Int64", "int64", "int32", "float", "int"}

# converting dates to duration
_DURATION_UNIT_MAP: dict[str, str] = {
    "year"  : "years",
    "month" : "months",
    "week"  : "weeks",
    "day"   : "days",
    "hour"  : "hours",
    "minute": "minutes",
    "second": "seconds",
}
_TIMEDELTA_UNITS = {"week", "day", "hour", "minute", "second"}
_CALENDAR_UNITS  = {"year", "month"}

# extracting numerical components from dates
_EXTRACT_UNIT_MAP: dict[str, str] = {
    "year"       : "year",
    "month"      : "month",
    "week"       : "isocalendar().week", 
    "day"        : "day",
    "hour"       : "hour",
    "minute"     : "minute",
    "second"     : "second",
    "dayofweek"  : "dayofweek",
    "dayofyear"  : "dayofyear",
    "weekofyear" : "isocalendar().week",
    "quarter"    : "quarter",
}

# Coarse units — integer makes more sense (no one is 32.47 years old)
_INTEGER_UNITS = {"year", "month", "week", "day"}

# Fine units — float makes more sense
_FLOAT_UNITS = {"hour", "minute", "second"}

# Default missing data methods
_DEFAULT_MISSING_BY_DTYPE: dict[str, MissingMethod] = {
    "float64"  : "median",
    "float32"  : "median",
    "Int64"    : "zero",
    "int64"    : "zero",
    "int32"    : "zero",
    "bool"     : "zero",
    "boolean"  : "zero",
    "category" : "mode",
    "object"   : "mode",
    "string"   : "blank"
}

####################
# csv concatenation
def concat_csvs(
        input_dir : str,
        print_filenames : bool = True,
        low_memory : bool = False
) -> pd.DataFrame:
    ''' 
    Concatenate all CSV files in a directory into a single DataFrame.
 
    Parameters
    ----------
    input_dir : str
    print_filenames : bool
    low_memory : bool
        Passed to pd.read_csv. Set False to avoid mixed-type warnings.
 
    Returns
    -------
    pd.DataFrame   
    '''
    print(f"Concatenating all csv files in directory {input_dir} into a single pandas dataframe")
    frames = []

    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            if print_filenames:
                print(file)
            frames.append(
                pd.read_csv(os.path.join(input_dir,file), low_memory = low_memory)
            )
    
    if not frames:
        raise FileNotFoundError(f"No csv files found in {input_dir}")
    
    return pd.concat(frames, ignore_index = True)

#######################
# dtype transformations
def transform_datetime(
        df: pd.DataFrame,
        schema: Mapping[str, str | tuple],
        preserve_colnames: bool = False,
        reference_date: str | pd.Timestamp | None = None
) -> pd.DataFrame:
    '''
    Transform datetime columns into numeric features suitable for model inputs.
    Call after update_dtypes() has converted columns to datetime dtype.
    This is intentially separate from update_dtypes() to allow for a deliberate
    choice of transformation based on research interpretation.

    Schema mapping options:
    -----------------------
    - "ordinal"
    Maps dates to ordinal representation for time series analysis

    - ("duration", time_unit, reference_date)
    Converts datetimes to numeric representation of time_units since reference_date.
    time_unit can be one of 'year', 'month', 'week', 'day', 'hour', 'second'
    Useful for time-to-event, age, survival analysis

    - ("extract", [time_units])
    Extracts numeric/categorical representation of time_units from datetime object.
    time_units should be a list containing any of 'year', 'month', 'week', 'day', 'hour', 'second', 
    'dayofweek', 'weekofyear'
    Defaults to ['year', 'month', 'day']
    Useful for forest-based methods

    - ("after", threshold)
    Coverts datetime to binary (0/1) column, where 0 = "before", 1 = "after"
    Used for clinically meanignful interventions

    - ("bin", [bins], [labels], time_unit, reference_date)
    Maps datetimes to specific labeled groups specified by 'bins' and 'labels'.
    Datetimes are first converted to "duration" representation using time_units and 
    reference_date
    
    Parameters
    ----------
    df : pd.DataFrame
    schema : dict  {column_name : transform_spec}
    preserve_colnames : bool (default = False)
        Whether to keep the original colnames in the transformed columns
    reference_date : str | pd.Timestamp | None (default = None)
        Global fallback reference date for duration transforms if not
        specified per-column in the schema.
 
    Returns
    -------
    pd.DataFrame
        Original datetime columns are dropped and replaced by derived
        numeric columns. The bin transform keeps the original column.  
    '''
    df = df.copy()
    ref = pd.Timestamp(reference_date) if reference_date else None

    for col, spec in schema.items():
        # safety checks
        if col not in df.columns:
            warnings.warn(f"Columns '{col}' not found - skipping datetime transform")
            continue

        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            warnings.warn(
                f"column '{col}' is not datetime dtype - skipping. "
                f"Run update_dtypes() first to convert to datetime"
            )
            continue

        # get spec 
        if isinstance(spec, str):
            transform = spec 
            args = []
        else:
            transform = spec[0]
            args = list(spec[1:])

        # transformations
        if transform == "ordinal":
            df[f"{col}_ordinal"] = df[col].map(pd.Timestamp.toordinal)
            print(f"  '{col}' → '{col}_ordinal' (integer ordinal)")
            df = df.drop(columns=[col])

        elif transform == "duration":
            time_unit = args[0]
            reference = pd.Timestamp(args[1]) if len(args) > 1 else ref
            if time_unit is None:
                raise ValueError(
                    f"'{col}': duration transform requires a time_unit is specified. "
                )
            elif time_unit not in _DURATION_UNIT_MAP:
                raise ValueError(
                    f"'{col}': invalid time_unit '{time_unit}'. "
                    f"Valid: {list(_DURATION_UNIT_MAP.keys())}."
                )
            if reference is None:
                raise ValueError(
                    f"'{col}': duration transform requires a reference date. "
                    f"Pass it in the column-specific tuple or as the reference_date argument"
                )
            # calculate duration
            precision = 0 if time_unit in _INTEGER_UNITS else 2
            unit_map = _DURATION_UNIT_MAP[time_unit]
            if time_unit in _TIMEDELTA_UNITS:
                duration = (reference - df[col]) / pd.Timedelta(1, unit_map)
            elif time_unit == "year":
                duration = (reference - df[col]).dt.days / 365.25
            elif time_unit == "month":
                duration = (reference - df[col]).dt.days / 30.4375
            else:
                raise ValueError(f"Unhandled time_unit '{time_unit}'.")
            df[f"{col}_duration_{time_unit}"] = duration.round(precision)
            print(f"  '{col}' → '{col}_duration_{time_unit}' (numerical duration)")
            df = df.drop(columns=[col])

        elif transform == "extract":
            time_units = args[0] if args else ["year", "month", "day"]
            for time_unit in time_units:
                if time_unit not in _EXTRACT_UNIT_MAP:
                    raise ValueError(
                        f"'{col}': invalid time_unit '{time_unit}'. "
                        f"Valid: {list(_EXTRACT_UNIT_MAP.keys())}."
                    )
                unit_map = _EXTRACT_UNIT_MAP[time_unit]

                if time_unit in ("weekofyear", "week"):
                    df[f"{col}_{time_unit}"] = df[col].dt.isocalendar().week.astype("Int64")
                else:
                    df[f"{col}_{time_unit}"] = getattr(df[col].dt, unit_map)
                print(f"  '{col}' → '{col}_{time_unit}' (integer unit)")
            df = df.drop(columns=[col])

        elif transform == "after":
            if not args:
                raise ValueError(
                    f"'{col}': after transform requires a threshold date. "
                    f"e.g. ('{col}', 'flag', '2020-03-16')"
                )
            threshold = pd.Timestamp(args[0])
            df[f"{col}_after_{threshold}"] = (df[col] >= threshold).astype("Int64")
            print(f"  '{col}' → '{col}_after_{threshold}' (binary flag)")
            df = df.drop(columns=[col])
        
        elif transform == "bin":
            if (len(args) < 3) or (len(args) < 4 and ref is None):
                raise ValueError(
                    f"'{col}': bin transform requires (bins, labels, time_unit, reference_date). "
                    f"No reference_date found in args or the reference_date argument."

                    f"e.g. ('{col}', 'bin', [0,18,35,50,65], "
                    f"['<18','18-35','35-50','50-65','65+'], 'year', '2020-03-13')"
                )
            bins, labels, time_unit = args[0], args[1], args[2]
            if time_unit not in _DURATION_UNIT_MAP:
                raise ValueError(
                    f"'{col}': invalid time_unit '{time_unit}'. "
                    f"Valid: {list(_DURATION_UNIT_MAP.keys())}."
                )
            reference = pd.Timestamp(args[3]) if len(args) > 3 else ref
            # get bins
            precision = 0 if time_unit in _INTEGER_UNITS else 2
            unit_map = _DURATION_UNIT_MAP[time_unit]
            if time_unit in _TIMEDELTA_UNITS:
                duration = (reference - df[col]) / pd.Timedelta(1, unit_map)
            elif time_unit == "year":
                duration = (reference - df[col]).dt.days / 365.25
            elif time_unit == "month":
                duration = (reference - df[col]).dt.days / 30.4375
            else:
                raise ValueError(f"Unhandled time_unit '{time_unit}'.")
            duration = (duration).round(precision)
            df[f"{col}_group"] = pd.cut(duration, bins=bins, labels=labels, include_lowest=True)
            print(f"  '{col}' → '{col}_group' (ordered categorical, original kept)")
    return df


def update_dtypes(
    df: pd.DataFrame,
    schema: Mapping[str, str | tuple[str, str | Callable | list]],
    force_string_to_numeric: bool = False
) -> pd.DataFrame:
    """ 
    Update column dtypes based on a feature type schema

    Accepted feature types:
    -----------------------
    "continuous"                    -> float64
    ("binary", rule)                -> Int64 (if rule is string then rows matching flag (case-insensitive) -> 1,
                                    if callable then function applied to rows to determine 1/0)
    "discrete"                      -> Int64 (useful for counts data)
    "ordinal"                       -> Int64
    ("ordinal_cat", ordered_list)   -> ordinal_cat
    "categorical"                   -> pandas Categorical
    "datetime"                      -> datetime64
    "id"                            -> string (kept as-is, not used in modelling)

    Raw pandas dtypes ('float64', 'int64', etc.) are also accepted for convenience.

    Parameters: 
    df : pd.DataFrame,
    schema : dict {column_name : dtype_spec}
        type_spec is either a plain string (e.g. "continuous") or a tuple of
        (feature_type, positive_pattern) for binary columns.

        e.g. {
            "age"      : "continuous",
            "race"     : "categorical",
            "dob"      : "datetime",
            "pat_id"   : "id",
            "sex"      : ("binary", "female"),
            "medicaid" : ("binary", "yes"),
            "deceased" : ("binary", "true"),
        }

        If the positive pattern is a string, it is matched case-insensitively and matches and stripped of
        leading/trailing whitespace, returning 1 for matching rows, and 0 for non-null non-matching rows. 

        If the positive pattern is a callable, it is applied to the rows as a boolean/integer mask.

    Returns:
    pd.DataFrame
    """
    df = df.copy() 

    for col, spec in schema.items(): 
        # get spec
        if isinstance(spec, tuple):
            ftype, arg = spec[0], spec[1]
        else:
            ftype, arg = spec, None

        # safety checks 
        if col not in df.columns:
            warnings.warn(f"Columns '{col}' not found - skipping dtype update")
            continue
        
        if ftype not in _DTYPE_MAP and ftype not in _RAW_DTYPE_STRINGS:
            raise ValueError(
                f"'{col}': unknown feature type '{ftype}'. "
                f"Valid semantic types: {list(_DTYPE_MAP.keys())}. "
                f"Valid raw dtype strings: {_RAW_DTYPE_STRINGS}."
            )
        if ftype == "binary" and arg is None:
            raise ValueError(
                f"'{col}': 'binary' requires a rule as the second element of the tuple.\n"
                f"  String:   ('{col}', 'binary', 'female')\n"
                f"  Callable: ('{col}', 'binary', lambda x: x > 30.0)"
            )
        if ftype == "ordinal_cat" and arg is None:
            raise ValueError(
                f"'{col}': 'ordinal_cat' requires an ordered label list as the second element.\n"
                f"  Example: ('{col}', 'ordinal_cat', ['Low', 'Medium', 'High'])"
            )
        if ftype == "ordinal_cat" and not isinstance(arg, list):
            raise ValueError(
                f"'{col}': 'ordinal_cat' label list must be a list, got {type(arg)}."
            )

        # raw int/float types
        if ftype in _RAW_DTYPE_STRINGS:
            ftype = _ALIAS_MAP.get(ftype, ftype)
            if force_string_to_numeric and _is_string_dtype(df[col]):
                df[col] = df[col].astype(str).str.extract(r"([-\d.]+)")[0]
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(ftype)  # type: ignore[arg-type]
        # continuous
        elif ftype == "continuous":
            if force_string_to_numeric and _is_string_dtype(df[col]):
                df[col] = df[col].astype(str).str.extract(r"([-\d.]+)")[0]
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")  # type: ignore[arg-type]
        # discrete, ordinal
        elif ftype in ("discrete", "ordinal"):
            if force_string_to_numeric and _is_string_dtype(df[col]):
                df[col] = df[col].astype(str).str.extract(r"([-\d.]+)")[0]
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(pd.Int64Dtype()) # type: ignore[arg-type]
        # binary
        elif ftype == "binary":
            assert arg is not None
            assert isinstance(arg, (str, type(lambda: None))) or callable(arg)
            df[col] = _coerce_binary_explicit(df[col], arg)  # type: ignore[arg-type]
        # ordinal_cat 
        elif ftype == "ordinal_cat":
            assert isinstance(arg, list)
            unique_vals = set(df[col].dropna().unique())
            missing = set(arg) - unique_vals
            if missing:
                warnings.warn(
                    f"'{col}': the following labels were not found in the column: {missing}. "
                    f"These will produce NaN in the output."
                )
            df[col] = pd.Categorical(df[col], categories=arg, ordered=True)
            print(f"  '{col}': ordered categorical ({arg[0]} → {arg[-1]})")

        # categorical and id
        elif ftype == "categorical":
            df[col] = df[col].astype("category")

        elif ftype == "id":
            df[col] = df[col].astype(str)
        
        # datetime
        elif ftype == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
            
    return df

def _is_string_dtype(series: pd.Series) -> bool:
    """
    Check for string-like dtype in both pandas 1.x and 2.x.
    """
    return series.dtype == object or pd.api.types.is_string_dtype(series)

def _coerce_binary_explicit(
    series: pd.Series,
    rule: str | Callable
) -> pd.Series:
    ''' 
    Encode a series as 0/1 using either a string pattern or a callable rule.

    Parameters
    ----------
    series : pd.Series
    rule : str | callable
        If str: rows matching the string (case-insensitive) → 1, others → 0.
        If callable: function applied to the series returning a boolean mask.
            e.g. lambda x: x > 30.0

    Returns
    -------
    pd.Series  (dtype Int64 — nullable integer)
    '''
    nulls = series.isna() 

    if callable(rule):
        try:
            result = rule(series)
        except Exception as e:
            raise ValueError(
            f"binary rule callable raised an error when applied: {e}"
        )
        if not pd.api.types.is_bool_dtype(result) and not pd.api.types.is_integer_dtype(result):
            raise ValueError(
                f"binary rule callable must return a boolean or integer mask, "
                f"got dtype '{result.dtype}'."
            )
        encoded = result.astype("Int64")

    elif isinstance(rule, str):
        encoded = (
            series.astype(str).str.strip().str.lower().eq(rule.strip().lower()).astype("Int64")
        )
        
    else:
        raise ValueError(
            f"binary rule must be a string pattern or callable, got {type(rule)}."
        )
    # restore nulls
    encoded[nulls] = pd.NA
    return encoded

#########################
# handling missing values
def handle_missing(
    df: pd.DataFrame,
    schema: Mapping[str, MissingMethod | tuple[MissingMethod, str]] | None = None,
    columns: list[str] | None = None,
    ffill: bool = False,
    default_numeric: MissingMethod = "median",
    default_categorical: MissingMethod = "mode"
) -> pd.DataFrame:
    """
    Fill or drop missing values.
 
    Methods
    -------
    "zero"                     -> fill with 0
    "mean"                     -> fill with column mean  (numeric only)
    "median"                   -> fill with column median (numeric only)
    "mode"                     -> fill with most frequent value
    "blank"                    -> fill with ""
    ("string", defaultstring)  -> fill with defaultstring
    "drop"                     -> drop rows where this column is null
    None                       -> leave as-is
 
    Parameters
    ----------
    df : pd.DataFrame
    schema : dict  {column: method}  — per-column override
    columns : list  — restrict to these columns (default: all with nulls)
    ffill : bool  — forward-fill entire df first (useful for panel/time-series data)
    default_numeric : MissingMethod — fallback for numeric columns not in schema
    default_categorical : MissingMethod — fallback for non-numeric columns not in schema
 
    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    if ffill:
        print("ffill enabled — forward-filling missing values.")
        df = df.ffill()

    # default to all columns with missing values
    columns = columns or list(df.columns[df.isna().any()])
    # default to empty dict
    schema = schema or {} 
    # keep track of any rows to drop
    rows_to_drop = pd.Series(False, index=df.index) 

    for col in columns: 
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found - skipping missing value imputation")
            continue
        dtype_name = str(df[col].dtype).lower()
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        # get method from schema if exists (else get default method for dtype if found)
        method = schema.get(
            col,
            _DEFAULT_MISSING_BY_DTYPE.get(
                dtype_name,
                default_numeric if is_numeric else default_categorical,
            ),
        )
        if isinstance(method, tuple):
            method_name, fill_value = method[0], method[1]
        else:
            method_name, fill_value = method, None
        # get null count
        null_count = df[col].isna().sum()
        if null_count == 0:
            continue
        print(f"  '{col}' ({dtype_name}): {null_count} nulls → method='{method_name}'")
        # missing values
        
        if method_name is None:
            continue
        elif method_name == "zero":
            df[col] = df[col].fillna(0)
        elif method_name == "mean":
            if not is_numeric:
                raise ValueError(
                    f"'{col}': 'mean' method is only valid for numeric columns, "
                    f"got dtype '{dtype_name}'."
                )
            df[col] = df[col].fillna(df[col].mean())
        elif method_name == "median":
            if not is_numeric:
                raise ValueError(
                    f"'{col}': 'median' method is only valid for numeric columns, "
                    f"got dtype '{dtype_name}'."
                )
            df[col] = df[col].fillna(df[col].median())
        elif method_name == "mode":
            mode_val = df[col].mode()
            if len(mode_val) == 0:
                warnings.warn(
                    f"'{col}': could not compute mode — all values are null. "
                    f"Skipping imputation."
                )
            else:
                df[col] = df[col].fillna(mode_val[0])
        elif method_name == "blank":
            if hasattr(df[col], 'cat'):
                if fill_value not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(fill_value)
            df[col] = df[col].fillna("")
        elif method_name == "string":
            if fill_value is None:
                raise ValueError(
                    f"'{col}': 'string' method requires a fill value as the second "
                    f"element of the tuple.\n"
                    f"  Example: {'{col}' : ('string', 'Unknown')}"
                )
            # if categorical, add fill_value to categories first
            if hasattr(df[col], 'cat'):
                if fill_value not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(fill_value)
            df[col] = df[col].fillna(fill_value)
        elif method_name == "drop":
            print(f"Dropping rows with missing value for column '{col}'")
            rows_to_drop |= df[col].isna()
        else:
            raise ValueError(f"Unknown missing value method: '{method_name}'")

    if rows_to_drop.any():
        n_dropped = rows_to_drop.sum()
        print(f"  Dropping {n_dropped} rows due to 'drop' method columns.")
        df = pd.DataFrame(df.loc[~rows_to_drop])
    
    return df
    
#######################
# Remaining work:
# encode dummies? --> Does this need to be any different from existing supported methods

# consolidate categories? --> needing to group several sparse categories into a larger group\

# detect_feature_types? --> some sort of logic for inferring the feature types of a raw dataframe

# summarize features? --> a more informative df.info() method for seeing 
                        #  feature details, dtype, null proportion, unique records, and example values