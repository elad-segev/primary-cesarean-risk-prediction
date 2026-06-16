import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import config as c
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Dealing with missing values
# ---------------------------------------------------------------------------

def apply_complete_case_analysis(df: pd.DataFrame, required_cols: list) -> pd.DataFrame:

    original_n = len(df)
    
    df_complete = df.dropna(subset=required_cols).copy()
    
    effective_n = len(df_complete)
    dropped_n = original_n - effective_n
    
    return df_complete , (dropped_n, round((dropped_n / original_n) * 100, 2))