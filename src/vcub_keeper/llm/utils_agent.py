from threading import local

import pandas as pd

_thread_local = local()


def set_current_dataframe(df: pd.DataFrame):
    """Set the current DataFrame for tools to access."""
    _thread_local.dataframe = df


def get_current_dataframe() -> pd.DataFrame:
    """Get the current DataFrame."""
    return getattr(_thread_local, "dataframe", None)
