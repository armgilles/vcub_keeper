## Informations sur les données du projet :

Les fichiers de données sont organisés par répertoire : 
 - `raw` : Fichiers sources bruts (non retravaillées).
 - `ref` : Fichiers de référence (d'attributs).
 - `clean` : Fichiers retravaillés provenant du répertoire `raw`.

### Raw - Fichiers sources

1. `bordeaux-YYYY.csv` (YYYY étant l'année) : Fichiers d'activités des stations (un fichier égal à une année)

|          |   id | status   |   available_stands |   available_bikes | timestamp           |
|---------:|-----:|:---------|-------------------:|------------------:|:--------------------|
|  9799112 |    1 | open     |                  8 |                25 | 2020-01-01 03:49:13 |
|   462704 |    1 | open     |                  8 |                25 | 2020-01-01 03:54:13 |
| 11305359 |    1 | open     |                  8 |                25 | 2020-01-01 03:57:13 |
|  4283349 |    1 | open     |                  8 |                25 | 2020-01-01 04:02:13 |
|  3945493 |    1 | open     |                  9 |                24 | 2020-01-01 04:08:13 |

Ces fichiers sont concaténés puis retravaillés notamment afin d'avoir les informations sur le nombre d'ajouts ou de prises de vélo par station sur un temps de 5 minutes (fichiers sources) avant d'être resampler sur 10 minutes. Ces transformations sont assurées par la fonction `vcub_keeper/create/creator.py create_activity_time_series()` créant le fichier dans le répertoire `clean` nommé `time_serie_activity.h5`.


### Ref - Fichiers de références

1. `station_attribute.csv` Fichier de référence sur les stations Vcub de Bordeaux provenant du portail open-data de [bordeaux-metropole](https://opendata.bordeaux-metropole.fr/explore/dataset/ci_vcub_p/table/). Celui-ci peut etre créé à partir de `/create/creator.py create_station_attribute()`
   - Le fichier est légèrement modifié (changement de nom de colonnes, filtre sur les colonnes).
   - Fonction de lecture : `/reader/reader.py read_stations_attributes()`
  
|    | Geo Point           | Geo Shape                                                | COMMUNE   |   total_stand | NOM              | TYPEA   |   station_id |     lat |       lon |
|---:|:--------------------|:---------------------------------------------------------|:----------|--------------:|:-----------------|:--------|-------------:|--------:|----------:|
|  0 | 44.83803,-0.58437   | {"type": "Point", "coordinates": [-0.58437, 44.83803]}   | Bordeaux  |            33 | Meriadeck        | VLS     |            1 | 44.838  | -0.58437  |
|  1 | 44.83784,-0.59028   | {"type": "Point", "coordinates": [-0.59028, 44.83784]}   | Bordeaux  |            20 | St Bruno         | VLS     |            2 | 44.8378 | -0.59028  |
|  2 | 44.840813,-0.593233 | {"type": "Point", "coordinates": [-0.593233, 44.840813]} | Bordeaux  |            28 | Piscine Judaique | VLS     |            3 | 44.8408 | -0.593233 |
|  3 | 44.84221,-0.58482   | {"type": "Point", "coordinates": [-0.58482, 44.84221]}   | Bordeaux  |            20 | St Seurin        | VLS     |            4 | 44.8422 | -0.58482  |
|  4 | 44.840712,-0.581124 | {"type": "Point", "coordinates": [-0.581124, 44.840712]} | Bordeaux  |            40 | Place Gambetta   | VLS     |            5 | 44.8407 | -0.581124 |


1. `meteo.csv` Fichier météo qui indique différents indicateurs météo à l'heure créer à partir de `vcub_keeper/create/creator.py create_meteo()`. Ce fichier n'est pas utilisé dans la partie Machine Learning.

|       | date                |   temperature |   pressure |   humidity |   precipitation |   wind_speed |
|------:|:--------------------|--------------:|-----------:|-----------:|----------------:|-------------:|
| 15787 | 2020-09-17 19:00:00 |          26.4 |     1006.2 |         39 |               0 |          1.5 |
| 15788 | 2020-09-17 20:00:00 |          24.2 |     1006.3 |         49 |               0 |          0.5 |
| 15789 | 2020-09-17 21:00:00 |          23.9 |     1006.5 |         46 |               0 |          2.6 |
| 15790 | 2020-09-17 22:00:00 |          24.4 |     1006.3 |         45 |               0 |          3.1 |
| 15791 | 2020-09-17 23:00:00 |          24.4 |     1005.9 |         47 |               0 |          2.6 |

   - Fonction de lecture : `/reader/reader.py read_meteo()`


### Clean - Fichiers retravaillés

1. `time_serie_activity.h5` fichier retraillé à partir des données `raw` sur l'activité des stations. La lecture est assurée par la fonction  `vcub_keeper/reader/reader.py read_time_serie_activity()`. 


|          |   station_id | date                |   available_stands |   available_bikes |   status |   transactions_in |   transactions_out |   transactions_all |
|---------:|-------------:|:--------------------|-------------------:|------------------:|---------:|------------------:|-------------------:|-------------------:|
| 16564507 |          251 | 2020-08-28 11:10:00 |                 28 |                12 |        1 |                 0 |                  0 |                  0 |
| 16564508 |          251 | 2020-08-28 11:20:00 |                 28 |                12 |        1 |                 0 |                  0 |                  0 |
| 16564509 |          251 | 2020-08-28 11:30:00 |                 26 |                14 |        1 |                 2 |                  0 |                  2 |
| 16564510 |          251 | 2020-08-28 11:40:00 |                 26 |                14 |        1 |                 0 |                  0 |                  0 |
| 16564511 |          251 | 2020-08-28 11:50:00 |                 26 |                14 |        1 |                 0 |                  0 |                  0 |

- `transactions_in` : Nombre d'ajouts de vélo qu'il y a eu pour une même station entre 2 points de données
- `transactions_out` : Nombre de prise de vélo qu'il y a eu pour une même station entre 2 points de données 
- `transactions_in` : Nombre de transactions de vélo (ajout et prise) qu'il y a eu pour une même
    station entre 2 points de données

## API de données : 

Les données live sont obtenu par le projet [Jitenshea](https://github.com/garaud/jitenshea) ou à partir de l'open data de [Bordeaux](https://opendata.bordeaux-metropole.fr/explore/dataset/ci_vcub_p/information/) 
