import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, offline
from plotly.subplots import make_subplots
from keplergl import KeplerGl

from vcub_keeper.reader.reader_utils import filter_periode
from vcub_keeper.ml.cluster import predict_anomalies_station, logistic_predict_proba_from_model
from vcub_keeper.transform.features_factory import (get_transactions_in,
                                                    get_transactions_out,
                                                    get_transactions_all,
                                                    get_consecutive_no_transactions_out)
from vcub_keeper.config import (NON_USE_STATION_ID, MAPBOX_TOKEN,
                                THRESHOLD_PROFILE_STATION, FEATURES_TO_USE_CLUSTER)


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
    sns.heatmap(pivot_station, linewidths=.5, cmap="coolwarm", vmin=vmin, annot=True)
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


def plot_map_station_with_plotly(station_control,
                                 station_id=None,
                                 width=800,
                                 height=600,
                                 offline_plot=False,
                                 return_plot=False,):
    """
    Affiche une cartographie de l'agglomération de Bordeaux avec toutes les stations Vcub et leurs états
    provenant des algorithmes (normal, inactive et anomaly).

    Si "station_id" est indiqué, alors la cartographie est focus sur la lat / lon de station (Numéro)

    Parameters
    ----------
    data : pd.DataFrame
        En provenance de station_control.csv (vcub_watcher)
    station_id : Int [opt]
        Numéro de station que l'on souhaite voir (en focus) sur la cartographie.
    width : int [opt]
        Largeur du graphique (en px).
    height : int [opt]
        Longeur du graphique (en px).
    offline_plot : bool [opt]
        Pour retourner le graphique et l'utilisé dans une application
    return_plot : bool [opt]
        Retourne le graphique pour être utilisé par le front.

    Returns
    -------
    None

    Examples
    --------

    plot_map_station_with_plotly(station_control=station_control, offline_plot=False)
    """
    # Param plot with a given station_id
    if station_id is not None:
        # On centre le graphique sur la lat / lon de la station
        center_lat = \
            station_control[station_control['station_id'] == station_id]['lat'].values[0]
        center_lon = \
            station_control[station_control['station_id'] == station_id]['lon'].values[0]
        zoom_plot = 15
    else:
        center_lat = 44.837794
        center_lon = -0.581662
        zoom_plot = 11

    # Preprocess avant graphique
    station_control['etat'] = 'normal'

    # Non monitoré
    station_control.loc[station_control['mean_activity'] < THRESHOLD_PROFILE_STATION,
                        'etat'] = 'non surveillée'

    # En anoamlie (HS prediction)
    station_control.loc[station_control['is_anomaly'] == 1, 'etat'] = 'anomaly'

    # Inactive
    station_control.loc[station_control['is_inactive'] == 1, 'etat'] = 'inactive'

    # Transform date to string
    try:
        station_control['anomaly_since_str'] = \
            station_control['anomaly_since'].dt.strftime(date_format='%Y-%m-%d %H:%M')
    except KeyError:
        # Pour vcub_watcher intégration
        # https://github.com/armgilles/vcub_keeper/issues/49#issuecomment-822504771
        station_control['anomaly_since_str'] = \
            station_control['En anomalie depuis'].dt.strftime(date_format='%Y-%m-%d %H:%M')

    station_control['anomaly_since_str'] = station_control['anomaly_since_str'].fillna('-')

    # Color for etat
    color_etat = {'anomaly': '#EB4D50',
                  'inactive': '#5E9BE6',
                  'normal': '#6DDE75',
                  'non surveillée': '#696A6A'}

    # To know when use add_trace after init fig
    wtf_compteur = 0

    for etat in station_control['etat'].unique():
        # Filter
        temp = station_control[station_control['etat'] == etat]

        # Building text
        texts = []
        for idx, station in temp.iterrows():
            text = str(station['NOM']) \
                + " <br />" + "station N° : " + str(station['station_id']) \
                + " <br />" + "Nombre de vélo dispo : " + str(station['available_bikes']) \
                + " <br />" + "Activité suspecte depuis : " + str(station['anomaly_since_str'])
            texts.append(text)

        if wtf_compteur == 0:
            fig = go.Figure(go.Scattermapbox(lat=temp['lat'],
                                             lon=temp['lon'],
                                             mode='markers',
                                             hoverinfo='text',
                                             hovertext=texts,
                                             marker_size=9,
                                             marker_color=color_etat[etat],
                                             name=etat))
        else:
            fig.add_trace(go.Scattermapbox(lat=temp['lat'],
                                           lon=temp['lon'],
                                           mode='markers',
                                           hoverinfo='text',
                                           hovertext=texts,
                                           marker_size=9,
                                           marker_color=color_etat[etat],
                                           name=etat))

        wtf_compteur = 1

    fig.update_layout(mapbox=dict(center=dict(lat=center_lat, lon=center_lon),
                                  accesstoken=MAPBOX_TOKEN,
                                  zoom=zoom_plot,
                                  style="light"),
                      showlegend=True,
                      width=width,
                      height=height,
                      legend=dict(orientation="h",
                                  yanchor="top",
                                  xanchor="center",
                                  y=1.1,
                                  x=0.5
                                  ))
    # To get map on Front
    if return_plot is True:
        return fig

    if offline_plot is False:
        iplot(fig)
    else:
        offline.plot(fig)


def plot_map_station_with_kepler(station_control, station_id=None):
    """
    Affiche une cartographie de l'agglomération de Bordeaux avec toutes les stations Vcub et leurs états
     provenant des algorithmes (normal, inactive et anomaly).

    Si "station_id" est indiqué, alors la cartographie est focus sur la lat / lon de station (Numéro)

    Parameters
    ----------
    data : pd.DataFrame
        En provenance de station_control.csv (vcub_watcher)
    station_id : Int [opt]
        Numéro de station que l'on souhaite voir (en focus) sur la cartographie.
    Returns
    -------
    map_kepler : Graphique

    Examples
    --------

    map_kepler = plot_map_station_with_kepler(data=station_control, station_id=6)
    """

    # Global config plot
    config_global = {
      "version": "v1",
      "config": {
        "visState": {
          "filters": [],
          "layers": [
            {
              "id": "fmdzqhw",
              "type": "point",
              "config": {
                "dataId": "data_1",
                "label": "Station Vcub",
                "color": [
                  18,
                  147,
                  154
                ],
                "columns": {
                  "lat": "lat",
                  "lng": "lon",
                  "altitude": None
                },
                "isVisible": True,
                "visConfig": {
                  "radius": 12,
                  "fixedRadius": False,
                  "opacity": 0.8,
                  "outline": False,
                  "thickness": 2,
                  "strokeColor": None,
                  "colorRange": {
                    "name": "Custom Palette",
                    "type": "custom",
                    "category": "Custom",
                    "colors": [
                      "#696A6A",
                      "#6DDE75",
                      "#EB4D50",
                      "#5E9BE6"
                    ]
                  },
                  "strokeColorRange": {
                    "name": "Global Warming",
                    "type": "sequential",
                    "category": "Uber",
                    "colors": [
                      "#5A1846",
                      "#900C3F",
                      "#C70039",
                      "#E3611C",
                      "#F1920E",
                      "#FFC300"
                    ]
                  },
                  "radiusRange": [
                    0,
                    50
                  ],
                  "filled": True
                },
                "hidden": False,
                "textLabel": [
                  {
                    "field": None,
                    "color": [
                      255,
                      255,
                      255
                    ],
                    "size": 18,
                    "offset": [
                      0,
                      0
                    ],
                    "anchor": "start",
                    "alignment": "center"
                  }
                ]
              },
              "visualChannels": {
                "colorField": {
                  "name": "etat_id_sort",
                  "type": "integer"
                },
                "colorScale": "ordinal",
                "strokeColorField": None,
                "strokeColorScale": "quantile",
                "sizeField": None,
                "sizeScale": "linear"
              }
            }
          ],
          "interactionConfig": {
            "tooltip": {
              "fieldsToShow": {
                "data_1": [
                  {
                    "name": "station_id",
                    "format": None
                  },
                  {
                    "name": "NOM",
                    "format": None
                  },
                  {
                    "name": "etat",
                    "format": None
                  },
                  {
                    "name": "available_bikes",
                    "format": None
                  },
                  {
                    "name": "anomaly_since_str",
                    "format": None
                  }
                ]
              },
              "compareMode": True,
              "compareType": "absolute",
              "enabled": True
            },
            "brush": {
              "size": 0.5,
              "enabled": False
            },
            "geocoder": {
              "enabled": False
            },
            "coordinate": {
              "enabled": False
            }
          },
          "layerBlending": "normal",
          "splitMaps": [],
          "animationConfig": {
            "currentTime": None,
            "speed": 1
          }
        },
        "mapState": {
          "bearing": 0,
          "dragRotate": False,
          "latitude": 44.85169239146265,
          "longitude": -0.5868239240658858,
          "pitch": 0,
          "zoom": 11.452871077625481,
          "isSplit": False
        },
        "mapStyle": {
          "styleType": "muted",
          "topLayerGroups": {
            "water": False
          },
          "visibleLayerGroups": {
            "label": True,
            "road": True,
            "border": False,
            "building": True,
            "water": True,
            "land": True,
            "3d building": False
          },
          "threeDBuildingColor": [
            137,
            137,
            137
          ],
          "mapStyles": {}
        }
      }
    }

    # Preprocess avant graphique
    station_control['etat'] = 'normal'

    # Non monitoré
    station_control.loc[station_control['mean_activity'] < THRESHOLD_PROFILE_STATION,
                        'etat'] = 'non surveillée'

    # En anoamlie (HS prediction)
    station_control.loc[station_control['is_anomaly'] == 1, 'etat'] = 'anomaly'

    # Inactive
    station_control.loc[station_control['is_inactive'] == 1, 'etat'] = 'inactive'

    # Transform date to string
    try:
        station_control['anomaly_since_str'] = \
            station_control['anomaly_since'].dt.strftime(date_format='%Y-%m-%d %H:%M')
    except KeyError:
        # Pour vcub_watcher intégration
        # https://github.com/armgilles/vcub_keeper/issues/49#issuecomment-822504771
        station_control['anomaly_since_str'] = \
            station_control['En anomalie depuis'].dt.strftime(date_format='%Y-%m-%d %H:%M')

    # Drop date for Kepler
    station_control = station_control.drop(['last_date_anomaly', 'anomaly_since'], axis=1)

    # Add fake stations to have every type of etat (anomaly / normal / inactive
    # & non surveillée) to match with fill color in plot

    fake_station_every_etat = \
        [[990, 0.4, 1, 0, 35, 60.9616624, -39.1527227, 'Station fake anomaly',
            'anomaly', '2021-04-07 09:00'],
         [991, 0.42, 1, 0, 21, 60.9616624, -39.1527227, 'Station fake normal',
          'normal', '-'],
         [992, 0.36, 0, 1, 22, 60.9616624, -39.1527227, 'Station fake inactive',
            'inactive', '-'],
         [993, 0.0, 0, 0, 22, 60.9616624, -39.1527227, 'Station non surveillée',
            'non surveillée', '-']]

    fake_station_every_etat_df = pd.DataFrame(fake_station_every_etat, columns=station_control.columns)
    station_control = pd.concat([station_control, fake_station_every_etat_df])

    # Sorting DataFrame to fill color in correct order
    etat_id_sort = {'non surveillée': 0,
                    'normal': 1,
                    'anomaly': 2,
                    'inactive': 3}

    station_control['etat_id_sort'] = station_control['etat'].map(etat_id_sort)
    station_control = station_control.sort_values('etat_id_sort')

    # Param plot with a given station_id
    if station_id is not None:
        # On centre le graphique sur la lat / lon de la station
        center_lat = \
            station_control[station_control['station_id'] == station_id]['lat'].values[0]
        center_lon = \
            station_control[station_control['station_id'] == station_id]['lon'].values[0]

        config_global['config']['mapState']['latitude'] = center_lat
        config_global['config']['mapState']['longitude'] = center_lon
        config_global['config']['mapState']['zoom'] = 15.3
        config_global['config']['mapState']['bearing'] = 24
        config_global['config']['mapState']['dragRotate'] = True
        config_global['config']['mapState']['pitch'] = 54
        # Building 3D
        config_global['config']['mapStyle']['visibleLayerGroups']['3d building'] = True
        # file_name = 'keplergl_map_station.html'
    else:
        # file_name = 'keplergl_map_global.html'
        pass

    # Load kepler.gl with map data and config
    map_kepler = KeplerGl(height=400, data={"data_1": station_control}, config=config_global)

    # Export
    # map_kepler.save_to_html(file_name=file_name) # No more export
    return map_kepler


def plot_station_anomalies_with_score(data, clf, station_id,
                           start_date='',
                           end_date='',
                           return_data=False,
                           offline_plot=False,
                           display_title=True,
                           return_plot=False):
    """
    Plot Time Series activty and anomaly score
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

    plot_station_anomalies_with_score(data=ts_activity, clf=clf, station_id=22)
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
    
    data_pred['anomaly_score'] = \
        logistic_predict_proba_from_model(clf.decision_function(data_pred[FEATURES_TO_USE_CLUSTER])) * 100

    if start_date != '':
        data_pred = data_pred[data_pred['date'] >= start_date]

    if end_date != '':
        data_pred = data_pred[data_pred['date'] <= end_date]
        
    # Figure
    
    if display_title:
        title = "Détection d'anomalies sur la stations N° " + str(station_id)
    else:
        title = None
    
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        specs=[[{"secondary_y": True}],
                               [{"secondary_y": True}]],
                        row_heights=[0.82, 0.18],
                        vertical_spacing=0.01,
                        x_title=title,
                       )
    # Row 1

    # Axe 1
    fig.add_trace(go.Scatter(x=data_pred['date'],
                             y=data_pred['available_bikes'],
                             mode='lines',
                             line={'width': 2},
                             name="Vélo disponible"),
                  row=1, col=1)
                

    # Axe 2
    fig.add_trace(go.Scatter(x=data_pred['date'],
                             y=data_pred['consecutive_no_transactions_out'],
                             mode='lines',
                             line={'width': 1,
                                   'dash': 'dot',
                                   'color': 'rgba(189,189,189,1)'},
                             #yaxis='y2',
                             name='Absence consécutive de prise de vélo'),
                  row=1, col=1, secondary_y=True)

    # For shape hoverdata anomaly
    data_pred['ano_hover_text'] = np.NaN
    data_pred.loc[data_pred['anomaly'] == -1,
                  'ano_hover_text'] = data_pred['available_bikes']
    fig.add_trace(go.Scatter(x=data_pred['date'],
                             y=data_pred['ano_hover_text'],
                             mode='lines',
                             text='x',
                             connectgaps=False,
                             line={'width': 2,
                                   'color': 'red'},
                             name='anomaly'),
                 row=1, col=1) 
    
    # Row 2
    fig.add_trace(go.Scatter(x=data_pred['date'],
                             y=data_pred['anomaly_score'],
                             line={'width': 1, 'color': 'black'},
                             fill="tozeroy",
                             mode='lines', #'lines' #'none'
                             name="Score d'anomalie"),
                  row=2, col=1)

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


    # Design graph
    layout = dict(
        showlegend=True,
        legend=dict(orientation="h",
                    yanchor="top",
                    xanchor="center",
                    y=1.2,
                    x=0.5
                    ),
        xaxis=dict(
                rangeslider=dict(
                    visible=False
                ),
                type='date',
                tickformat='%a %Y-%m-%d %H:%M',
        ),
        yaxis={'title': 'Nombre de vélo disponible',
               'title_font': {'color': 'rgba(100, 111, 251, 1)'},
               'tickfont': {'color': 'rgba(100, 111, 251, 1)'}
                },
        yaxis2={'title': 'Absence consécutive de prise de vélo',
                'title_font': {'color': 'rgba(122, 122, 122, 1)'},
                #'overlaying': 'y',
                'side': 'right',
                'showgrid': False,
                'visible': True,
                'tickfont': {'color': 'rgba(122, 122, 122, 1)'}
               },
        yaxis3={'title': 'Score',
                'title_font': {'color': 'rgba(0, 0, 0, 0.8)'},
                'range': [0, 100],
                #'gridwidth': 25
                'tickmode': 'linear',
                'tick0': 0.0,
                'dtick': 25
               },
        template='plotly_white',
        hovermode='x',
        shapes=shapes
    )

    fig.update_layout(layout)
    
    # Horizontal line for anomaly score
    fig.add_shape(go.layout.Shape(type="line",
                                  name='test',
                                  x0=data_pred['date'].min(),
                                  y0=50,
                                  x1=data_pred['date'].max(),
                                  y1=50,
                                  line=dict(color='Red', width=1,
                                            dash='dot'),
                                  #xref='x',
                                  #yref='y'
                                 ),
                    row=2, col=1)
    
    if return_plot is True:
        return fig
    if offline_plot is False:
        iplot(fig)
    else:
        offline.plot(fig)

    if return_data is True:
        return data_pred
