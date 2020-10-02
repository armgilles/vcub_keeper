# vcub_keeper
Analyse de l'activité des stations Vcub dans la métropole de Bordeaux afin de détecter en amont les stations hors service

## Problème : 

Je suis un grand utilisateur du service Vcub (vélos en libre-service) à Bordeaux. Le problème c'est que trop souvent à mon gout, j'arrive à une station pour prendre un vélo, car celle-ci est hors service.

Il peut y avoir plusieurs problèmes (écran HS, application HS...). Pourtant lorsque l'on a un peu d'expérience, on sait si celle-ci doit être presque vide ou pleine à une certaine heure. Par exemple si je regarde **la station des châtrons** le matin vers 9H, celle-ci doit être presque vide (en général 3 à 5 vélos disponibles) en semaine. Lorsque celle-ci est pleine le matin, je sais qu'il y a un problème sur cette station.

## Objectif : 

Analyser les données des stations Vcub de Bordeaux afin de déterminer si la stations est hors service.

## Data :

[Wiki](https://github.com/armgilles/vcub_keeper/blob/master/data/wiki_data.md) des données utilisées.

## WIP : 

![image](https://user-images.githubusercontent.com/8374843/94968006-6d7d5000-0500-11eb-853b-7b944a11bb26.png)

Dimension reduction and activity of a station (we can see the 7 days of the week).

![image](https://user-images.githubusercontent.com/8374843/94968827-e630dc00-0501-11eb-9130-128679683423.png)

As previous image but in 3D for better exploration.

![image](https://user-images.githubusercontent.com/8374843/94968688-a8cc4e80-0501-11eb-8ad5-3c667ad730e5.png)

Anomaly detection on station `Place de la Victoire`, green line is our Vcub keeper which launches the alert for abnormal activity.
