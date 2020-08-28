import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, offline


import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, offline

def plot_station_activity(data, station_id,
                          features_to_plot=['available_stands'], 
                          date_col="date",
                          start_date='',
                          end_date='',
                          return_data=False):
    """
    Plot Time Series
    Parameters
    ----------
    data : pd.DataFrame
        Tableau temporelle de l'activité des stations Vcub
    station_id : Int
        Numéro de la station de Vcub
    features_to_plot : List
        Noms des la colonne à afficher sur le graphique
    date_col : str
        Nom de la colonne à utiliser pour la temporalité
    station_ids : list
        Liste des stations à afficher sur le graphique
    start_date : str
        Date de début du graphique yyyy-mm-dd
    end_date : str
        Date de fin du graphique yyyy-mm-dd
    return_data : bool
        Retour le DataFrame lié à la station demandé et au contraintes de date si remplie.
    Returns
    -------
    data : pd.DataFrame
        Could return it if return_data is True.
        
    Examples
    --------
    
    plot_station_activity(activite, station_id=25, start_date='2017-08-28',
                          end_date='2017-09-02')
    """

    if not isinstance(features_to_plot, list):
        raise TypeError('features_to_plot should be a list')

    init_notebook_mode(connected=True)

    all_station_id = data['station_id'].unique()

    if station_id not in all_station_id:
        raise ValueError(str(station_id) + ' is not in a correct station_id.')
        #print(station_id + ' is not in a correct station_id.')
    
    if start_date != '':
        data = data[data[date_col] >= start_date]
        
    if end_date != '':
        data = data[data[date_col] <= end_date]
        
    temp = data[data['station_id'] == station_id].copy()

    # Init list of trace
    data_graph = []
    for feature in features_to_plot:
        trace = go.Scatter(x=temp[date_col],
                           y=temp[feature],
                           mode='lines',
                           name=feature)
        data_graph.append(trace)

    # Design graph
    layout = dict(
        title='Activité de la stations N° '+ str(station_id),
        showlegend=True,
        xaxis=dict(
                rangeslider=dict(
                    visible=True
                ),
                type='date',
                tickformat='%a %Y-%m-%d %H:%M',
        ),
        yaxis=dict(
            title='Valeurs'
        )
    )

    fig = dict(data=data_graph, layout=layout)

    iplot(fig)
    if return_data is True:
        return temp