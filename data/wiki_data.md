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

1. `tb_stvel_p.csv` Fichier de référence sur les stations Vcub de Bordeaux provenant du portail open-data de [bordeaux-metropole](https://opendata.bordeaux-metropole.fr/explore/dataset/tb_stvel_p/information/?sort=-gid&q=croix+blanche).
   - Le fichier est légèrement modifié (changement de nom de colonnes, filtre sur les colonnes).
   - Fonction de lecture : `/reader/reader.py read_stations_attributes()`
  
|    | Geo Point             | Geo Shape                                                  | COMMUNE                |   total_stand | NOM                                                                                       | TYPEA   | ADRESSE                           | TARIF    |   station_id |     lat |       lon |
|---:|:----------------------|:-----------------------------------------------------------|:-----------------------|--------------:|:------------------------------------------------------------------------------------------|:--------|:----------------------------------|:---------|-------------:|--------:|----------:|
|  0 | 44.8351755,-0.5720497 | {"type": "Point", "coordinates": [-0.5720497, 44.8351755]} | BORDEAUX               |            17 | Grosse Cloche (fermée depuis le 26/09/2016. Réouverture prévue septembre 2017)            | VCUB    | 12 place de la Ferme de Richemont | VLS      |          104 | 44.8352 | -0.57205  |
|  1 | 44.8723721,-0.5906954 | {"type": "Point", "coordinates": [-0.5906954, 44.8723721]} | BRUGES                 |            20 | Bruges La Vache                                                                           | VCUB    | Rue Léopold Laplante              | VLS PLUS |          169 | 44.8724 | -0.590695 |
|  2 | 44.8500962,-0.5855819 | {"type": "Point", "coordinates": [-0.5855819, 44.8500962]} | BORDEAUX               |            16 | Place Marie Brizard (supprimée le 11 mars 2016 en raison des travaux tram D)              | VCUB    | 209 rue Fondaudège                | VLS      |           35 | 44.8501 | -0.585582 |
|  3 | 44.8492876,-0.4966899 | {"type": "Point", "coordinates": [-0.4966899, 44.8492876]} | ARTIGUES-PRES-BORDEAUX |            19 | Artigues Feydeau                                                                          | VCUB    | 22 Boulevard Feydeau              | VLS PLUS |          150 | 44.8493 | -0.49669  |
|  4 | 44.7821503,-0.5661566 | {"type": "Point", "coordinates": [-0.5661566, 44.7821503]} | VILLENAVE-D'ORNON      |            21 | Pont de la Maye (retirée le 19 novembre 2015 en raison des travaux d'extension du tram C) | VCUB    | face au 564 route de Toulouse     | VLS PLUS |           76 | 44.7822 | -0.566157 |


1. `meteo.csv` Fichier météo qui indique différents indicateurs météo à l'heure.

|     | date                |   min_temp |   mean_teamp |   max_temp |   pressure_mean |   humidity_mean |   precipitation |
|----:|:--------------------|-----------:|-------------:|-----------:|----------------:|----------------:|----------------:|
| 652 | 2020-09-12 00:00:00 |       14.4 |         21.1 |       30   |          1012.3 |            65   |             0   |
| 653 | 2020-09-13 00:00:00 |       14.6 |         23.5 |       33.6 |          1014   |            48.3 |             0   |
| 654 | 2020-09-14 00:00:00 |       19.2 |         26.6 |       37.2 |          1011   |            40   |             0   |
| 655 | 2020-09-15 00:00:00 |       17.9 |         23.9 |       31.9 |          1011.7 |            66.5 |             0.5 |
| 656 | 2020-09-16 00:00:00 |       17.6 |         23.7 |       31.9 |          1012   |            67.1 |             0   |

   - Fonction de lecture : `/reader/reader.py read_meteo()`


## Clean - Fichiers retravaillés

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
