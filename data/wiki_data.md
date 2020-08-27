## Informations sur les données du projet :

Les fichiers de données sont organisés par répertoire : 
 - `Raw` : Fichiers sources bruts (non retravaillées).
 - `Ref` : Fichiers de référence (d'attributs).
 - `Clean` : Fichiers retravailler provenant du répertoire `Raw`.

### Raw - Fichiers sources

1. TO DO

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

## Clean - Fichier re-travaillé

TO DO
