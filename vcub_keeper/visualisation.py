import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, offline


def plot_station_activity(data, station_ids=[],
                          feature_to_plot='available_stand', 
                          date_col="date",
                          start_date='',
                          end_date=''):
    """
    Plot Time Series
    Parameters
    ----------
    data : pd.DataFrame
        Tableau temporelle de l'activité des stations Vcub
    features_to_plot : str
        Nom de la colonne à afficher sur le graphique
    date_col : str
        Nom de la colonne à utiliser pour la temporalité
    station_ids : list
        Liste des stations à afficher sur le graphique
    start_date : str
        Date de début du graphique yyyy-mm-dd
    end_date : str
        Date de fin du graphique yyyy-mm-dd
    
    Returns
    -------
        
    Examples
    --------
    
    plot_station_activity(activite, station_ids=[25], start_date='2017-08-28',
                          end_date='2017-09-02')
    """

    if not isinstance(station_ids, list):
        raise TypeError('station_ids should be a list')

    init_notebook_mode(connected=True)

    all_station_id = data['station_id'].unique()

    if len(station_ids) == 0:
        raise ValueError('station_ids should not be empty')
    else:
        for station_id in station_ids:
            if station_id not in all_station_id:
                print(station_id + ' is not in the list station_ids.')
    
    if start_date != '':
        data = data[data[date_col] >= start_date]
        
    if end_date != '':
        data = data[data[date_col] <= end_date]

    # Init list of trace
    data_graph = []
    for station_id in station_ids:
        temp = data[data['station_id'] == station_id].copy()
        trace = go.Scatter(x=temp[date_col],
                           y=temp[feature_to_plot],
                           mode='lines',
                           name='station N° ' + str(station_id))
        data_graph.append(trace)

    # Design graph
    layout = dict(
        title='Activité des stations',
        showlegend=True,
        xaxis=dict(
                rangeslider=dict(
                    visible=True
                ),
                type='date',
                tickformat='%a %Y-%m-%d %H:%M',
        ),
        yaxis=dict(
            title=feature_to_plot
        )
    )

    fig = dict(data=data_graph, layout=layout)

    iplot(fig)