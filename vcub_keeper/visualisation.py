import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, offline

from vcub_keeper.reader.reader_utils import filter_periode
from vcub_keeper.ml.cluster import predict_anomalies_station
from vcub_keeper.transform.features_factory import (get_transactions_in,
                                                    get_transactions_out,
                                                    get_transactions_all,
                                                    get_consecutive_no_transactions_out)
from vcub_keeper.config import NON_USE_STATION_ID


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
        # print(station_id + ' is not in a correct station_id.')

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
        title='Activité de la stations N° ' + str(station_id),
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


def plot_profile_station(data, station_id, feature_to_plot, aggfunc='mean',
                         filter_data=True, vmin=None):
    """
    Affiche un graphique permettant d'obversé l'activité de la semaine lié à la varible
    `feature_to_plot` suivant le jour et l'heure.
    On prend uniquement les données lorsque la station est ouverte (status = 1)

    ----------
    data : pd.DataFrame
        Tableau temporelle de l'activité des stations Vcub
    station_id : Int
        Numéro de la station de Vcub
    feature_to_plot : str
        Noms ds la colonne à afficher sur le graphique
    aggfunc : str
        Type d'aggrégation à faire sur feature_to_plot
    filter_data : bool
        Est ce que l'on filtre les donner selon la fonction reader/reader.py filter_periode
    vmin : int
        Valeur minimal pour la colormap
    Returns
    -------
    None

    Examples
    --------

    plot_profile_station(ts_activity, station_id=108, feature_to_plot='transactions_all',
                          aggfunc='mean', filter_data=False)
    """

    station = data[data['station_id'] == station_id].copy()

    if filter_data is True:
        station = filter_periode(station, NON_USE_STATION_ID)

    # station status == 1 (ok)
    station = station[station['status'] == 1]

    # Resample hours
    station = station.set_index('date')
    station_resample = \
        station.resample('H', label='right').agg({feature_to_plot: 'sum'}).reset_index()

    station_resample['month'] = station_resample['date'].dt.month
    station_resample['weekday'] = station_resample['date'].dt.weekday
    station_resample['hours'] = station_resample['date'].dt.hour

    pivot_station = station_resample.pivot_table(index=["weekday"],
                                                 columns=["hours"],
                                                 values=feature_to_plot,
                                                 aggfunc=aggfunc)

    plt.subplots(figsize=(20, 5))
    sns.heatmap(pivot_station, linewidths=.5, cmap="coolwarm", vmin=vmin)
    plt.title("Profile d'activité de la station N°" + str(station_id) + ' / ' + feature_to_plot + ' (aggrégation : ' + aggfunc + ')');


def plot_station_anomalies(data, clf, station_id,
                           start_date='',
                           end_date='',
                           return_data=False,
                           offline_plot=False,
                           display_title=True,
                           return_plot=False):
    """
    Plot Time Series
    Parameters
    ----------
    data : pd.DataFrame
        Tableau temporelle de l'activité des stations Vcub
    clf : Pipeline Scikit Learn
        Estimator already fit
    station_id : Int
        ID station
    start_date : str [opt]
        Date de début du graphique yyyy-mm-dd
    end_date : str [opt]
        Date de fin du graphique yyyy-mm-dd
    return_data : bool [opt]
        Retour le DataFrame lié à la station demandé et au contraintes de date si remplie.
    offline_plot : bool [opt]
        Pour exporter le graphique
    display_title : bool [opt]
        Afin d'afficher le titre du graphique
    offline_plot : bool [opt]
        Pour retourner le graphique et l'utiliser dans une application

    Returns
    -------
    data : pd.DataFrame
        Could return it if return_data is True.

    Examples
    --------

    plot_station_anomalies(data=ts_activity, clf=clf, station_id=22)
    """

    # Filter on station_id
    data_station = data[data['station_id'] == station_id].copy()

    if 'consecutive_no_transactions_out' not in data.columns:
        # Some features
        data_station = get_transactions_in(data_station)
        data_station = get_transactions_out(data_station)
        data_station = get_transactions_all(data_station)
        data_station = get_consecutive_no_transactions_out(data_station)

    data_pred = predict_anomalies_station(data=data_station,
                                          clf=clf,
                                          station_id=station_id)

    if start_date != '':
        data_pred = data_pred[data_pred['date'] >= start_date]

    if end_date != '':
        data_pred = data_pred[data_pred['date'] <= end_date]

    # Init list of trace
    data_graph = []

    # Axe 1
    trace = go.Scatter(x=data_pred['date'],
                       y=data_pred['available_bikes'],
                       mode='lines',
                       line={'width': 2},
                       name="Vélo disponible")
    data_graph.append(trace)

    # Axe 2
    trace_ano = go.Scatter(x=data_pred['date'],
                           y=data_pred['consecutive_no_transactions_out'],
                           mode='lines',
                           line={'width': 1,
                                 'dash': 'dot',
                                 'color': 'rgba(189,189,189,1)'},
                           yaxis='y2',

                           name='Absence consécutive de prise de vélo')
    data_graph.append(trace_ano)

    # For shape hoverdata anomaly
    data_pred['ano_hover_text'] = np.NaN
    data_pred.loc[data_pred['anomaly'] == -1,
                  'ano_hover_text'] = data_pred['available_bikes']
    trace_ano2 = go.Scatter(x=data_pred['date'],
                            y=data_pred['ano_hover_text'],
                            mode='lines',
                            text='x',
                            connectgaps=False,
                            line={'width': 2,
                                  'color': 'red'},
                            name='anomaly')
    data_graph.append(trace_ano2)

    # Shapes anomaly
    shapes = []
    # https://github.com/armgilles/vcub_keeper/issues/38
    data_pred['no_anomalie'] = (data_pred['anomaly'] == 1)
    data_pred['anomaly_grp'] = data_pred['no_anomalie'].cumsum()

    grp = \
        data_pred[data_pred['anomaly'] == -1].groupby('anomaly_grp',
                                                      as_index=False)['date'].agg({'min': 'min',
                                                                                   'max': 'max'})

    max_value = data_pred['available_bikes'].max()
    for idx, row in grp.iterrows():
        shapes.append(dict(type="rect",
                           xref="x",
                           yref="y",
                           x0=row['min'],
                           y0=0,
                           x1=row['max'],
                           y1=max_value,
                           fillcolor="red",
                           opacity=0.7,
                           layer="below",
                           line_width=0
                           ))

    data_pred = data_pred.drop(['no_anomalie', 'anomaly_grp'], axis=1)
    
    if display_title:
        title = "Détection d'anomalies sur la stations N° " + str(station_id)
    else:
        title = None

    # Design graph
    layout = dict(
        title=title,
        showlegend=True,
        legend=dict(orientation="h",
                    yanchor="top",
                    xanchor="center",
                    y=1.2,
                    x=0.5
                    ),
        xaxis=dict(
                rangeslider=dict(
                    visible=True
                ),
                type='date',
                tickformat='%a %Y-%m-%d %H:%M',
        ),
        yaxis=dict(
            title='Valeurs'
        ),
        yaxis2={'overlaying': 'y',
                'side': 'right',
                'visible': False},
        template='plotly_white',
        hovermode='x',
        shapes=shapes
    )

    fig = dict(data=data_graph, layout=layout)
    if return_plot is True:
        return fig
    if offline_plot is False:
        iplot(fig)
    else:
        offline.plot(fig)

    if return_data is True:
        return data_pred
