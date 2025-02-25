## Informations sur les données du projet :

Les fichiers de données sont organisés par répertoire : 
 - `raw` : Fichiers sources bruts (non retravaillées).
 - `ref` : Fichiers de référence (d'attributs).
 - `clean` : Fichiers retravaillés provenant du répertoire `raw`.

### Raw - Fichiers sources

Ce répertoire n'est plus utilisé depuis la version 1.4.0 du projet. On n'utilise plus de fichiers sources bruts, mais directement l'API de Bordeaux.

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


2. `meteo.csv` Fichier météo qui indique différents indicateurs météo à l'heure créer à partir de `vcub_keeper/create/creator.py create_meteo()`. Ce fichier n'est pas utilisé dans la partie Machine Learning. Old

|       | date                |   temperature |   pressure |   humidity |   precipitation |   wind_speed |
|------:|:--------------------|--------------:|-----------:|-----------:|----------------:|-------------:|
| 15787 | 2020-09-17 19:00:00 |          26.4 |     1006.2 |         39 |               0 |          1.5 |
| 15788 | 2020-09-17 20:00:00 |          24.2 |     1006.3 |         49 |               0 |          0.5 |
| 15789 | 2020-09-17 21:00:00 |          23.9 |     1006.5 |         46 |               0 |          2.6 |
| 15790 | 2020-09-17 22:00:00 |          24.4 |     1006.3 |         45 |               0 |          3.1 |
| 15791 | 2020-09-17 23:00:00 |          24.4 |     1005.9 |         47 |               0 |          2.6 |

   - Fonction de lecture : `/reader/reader.py read_meteo()`

3. `station_profile.csv` Fichier de référence sur les stations Vcub de Bordeaux provenant du portail open-data de [bordeaux-metropole](https://opendata.bordeaux-metropole.fr/explore/dataset/ci_vcub_p/table/). Celui-ci peut etre créé à partir de `/create/creator.py create_station_profile()`
   - Le fichier est légèrement modifié (changement de nom de colonnes, filtre sur les colonnes).
   - Fonction de lecture : `/reader/reader.py read_stations_profile()`

┌────────────┬─────────────┬──────────┬────────┬───┬─────┬─────┬─────┬──────────────────────────┐
│ station_id ┆ total_point ┆ mean     ┆ median ┆ … ┆ 98% ┆ 99% ┆ max ┆ profile_station_activity │
│ ---        ┆ ---         ┆ ---      ┆ ---    ┆   ┆ --- ┆ --- ┆ --- ┆ ---                      │
│ u16        ┆ i64         ┆ f64      ┆ f64    ┆   ┆ f64 ┆ f64 ┆ i64 ┆ str                      │
╞════════════╪═════════════╪══════════╪════════╪═══╪═════╪═════╪═════╪══════════════════════════╡
│ 72         ┆ 135460      ┆ 0.010586 ┆ 0.0    ┆ … ┆ 0.0 ┆ 1.0 ┆ 1   ┆ low                      │
│ 206        ┆ 2005        ┆ 0.011471 ┆ 0.0    ┆ … ┆ 0.0 ┆ 1.0 ┆ 1   ┆ low                      │
│ 157        ┆ 101547      ┆ 0.011768 ┆ 0.0    ┆ … ┆ 0.0 ┆ 1.0 ┆ 1   ┆ low                      │
│ 80         ┆ 106987      ┆ 0.012123 ┆ 0.0    ┆ … ┆ 0.0 ┆ 1.0 ┆ 1   ┆ low                      │
│ 156        ┆ 105750      ┆ 0.012444 ┆ 0.0    ┆ … ┆ 0.0 ┆ 1.0 ┆ 1   ┆ low                      │
└────────────┴─────────────┴──────────┴────────┴───┴─────┴─────┴─────┴──────────────────────────┘


### Clean - Fichiers retravaillés

1. `learning_dataset.parquet` fichier retraillé à partir des données de la fonction  `create_learning_dataset()` sur l'activité des stations. La lecture est assurée par la fonction  `vcub_keeper/reader/reader.py read_learning_dataset()`. 


 ────────────┬────────────┬────────────┬────────────┬────────┬────────────┬────────────┬───────────┐
│ station_id ┆ date       ┆ available_ ┆ available_ ┆ status ┆ transactio ┆ transactio ┆ transacti │
│ ---        ┆ ---        ┆ stands     ┆ bikes      ┆ ---    ┆ ns_in      ┆ ns_out     ┆ ons_all   │
│ i32        ┆ datetime[μ ┆ ---        ┆ ---        ┆ u8     ┆ ---        ┆ ---        ┆ ---       │
│            ┆ s]         ┆ i64        ┆ i64        ┆        ┆ i64        ┆ i64        ┆ i64       │
╞════════════╪════════════╪════════════╪════════════╪════════╪════════════╪════════════╪═══════════╡
│ 1          ┆ 2022-01-01 ┆ 11         ┆ 22         ┆ 1      ┆ 0          ┆ 0          ┆ 0         │
│            ┆ 01:10:00   ┆            ┆            ┆        ┆            ┆            ┆           │
│ 1          ┆ 2022-01-01 ┆ 11         ┆ 22         ┆ 1      ┆ 0          ┆ 0          ┆ 0         │
│            ┆ 01:20:00   ┆            ┆            ┆        ┆            ┆            ┆           │
│ 1          ┆ 2022-01-01 ┆ 11         ┆ 22         ┆ 1      ┆ 0          ┆ 0          ┆ 0         │
│            ┆ 01:30:00   ┆            ┆            ┆        ┆            ┆            ┆           │
│ 1          ┆ 2022-01-01 ┆ 11         ┆ 22         ┆ 1      ┆ 0          ┆ 0          ┆ 0         │
│            ┆ 01:40:00   ┆            ┆            ┆        ┆            ┆            ┆           │
│ 1          ┆ 2022-01-01 ┆ 11         ┆ 22         ┆ 1      ┆ 0          ┆ 0          ┆ 0         │
│            ┆ 01:50:00   ┆            ┆            ┆        ┆            ┆            ┆           │
└────────────┴────────────┴────────────┴────────────┴────────┴────────────┴────────────┴───────────┘

- `transactions_in` : Nombre d'ajouts de vélo qu'il y a eu pour une même station entre 2 points de données
- `transactions_out` : Nombre de prise de vélo qu'il y a eu pour une même station entre 2 points de données 
- `transactions_all` : Nombre de transactions de vélo (ajout et prise) qu'il y a eu pour une même station entre 2 points de données

## API de données : 

Les données live sont obtenu par le projet [Jitenshea](https://github.com/garaud/jitenshea) ou à partir de l'open data de [Bordeaux](https://opendata.bordeaux-metropole.fr/explore/dataset/ci_vcub_p/information/). On priorise l'API de Bordeaux maintenant.
