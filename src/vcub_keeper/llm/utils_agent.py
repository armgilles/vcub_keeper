from threading import local

_thread_local = local()


def set_current_dataframes(dataframes: dict):
    """
    To load dataframes in thread-local storage for use in tools.
    set_current_dataframes({
        "last_info_station_pd": last_info_station_pd,
        "df_historical_station": df_historical_station  # Keep as LazyFrame
    })
    """
    global current_dataframes
    current_dataframes = dataframes


def get_current_dataframe(name: str):
    """
    To get dataframes from thread-local storage for use in tools.
    last_info_station_pd = get_current_dataframe("last_info_station_pd") OR
    df_historical_station = get_current_dataframe("df_historical_station")
    """
    return current_dataframes.get(name)
