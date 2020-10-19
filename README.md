# vcub_keeper
Analyse de l'activité des stations Vcub dans la métropole de Bordeaux afin de détecter en amont les stations hors service

![image](https://user-images.githubusercontent.com/8374843/96422013-ca168580-11f7-11eb-8939-d773b1a22953.png)
_Impossible de valider la prise d'un vélo (absence de bouton "validation") sur une station quasi pleine._

## Problème : 

Je suis un grand utilisateur du service Vcub (vélos en libre-service) à Bordeaux. Le problème c'est que trop souvent à mon gout, j'arrive à une station pour prendre un vélo et celle-ci est hors service ce qui provoque une certaine frustration...

Il n'est pas question ici de prédire le nombre de vélos disponibles à la station afin de ne pas tomber sur une station vide, mais bien d'arriver à prédire le fait que la station soit HS avant que celle-ci soit déconnectée par TBM.

Il peut y avoir plusieurs problèmes (écran HS, application HS, problème dans l'API des cadenas...). Pourtant lorsque l'on a un peu d'expérience, on sait si celle-ci doit être presque vide ou pleine à une certaine heure. Par exemple si je regarde **la station des châtrons** le matin vers 9H, celle-ci doit être presque vide (en général 3 à 5 vélos disponibles) en semaine. Lorsque celle-ci est pleine le matin, je sais qu'il y a un problème sur cette station.

## Objectif : 

Analyser les données des stations Vcub de Bordeaux afin de déterminer si la station est hors service (impossibilité de prendre un vélo alors que des vélos sont disponibles) alors que cette station est active.

Prévenir via twitter @TBM_V3 dès qu'une station est détecté comme HS (fonctionnalité dans un autre repo privé).

## Data :

[Wiki](https://github.com/armgilles/vcub_keeper/blob/master/data/wiki_data.md) des données utilisées.

## Études : 

### Analyse exploratives : 

![image](https://user-images.githubusercontent.com/8374843/94968006-6d7d5000-0500-11eb-853b-7b944a11bb26.png)

Réduction de dimension (via `PCA`) sur l'activité d'une station avec plus d'un an d'historique. On peut facilement distinguer les 7 jours de la semaine qui forment des silos. Plus les points sont hauts, plus il y a une absence d'activité sur la station et potentiellement un problème sur la station.

![image](https://user-images.githubusercontent.com/8374843/94968827-e630dc00-0501-11eb-9130-128679683423.png)

Identique à l'image précédente, mais en 3D afin de mieux observer certains phénomènes.

### Prédictions : 

![image](https://user-images.githubusercontent.com/8374843/96337330-a2ec7680-1086-11eb-84ec-c42c4cd5f7f6.png)

Détection d'anomalies sur la station `Rue de la Croix Blanche` à partir des données en temps réel de la station (l'API est privé et n'est pas présente dans le repo sorry ;) )
