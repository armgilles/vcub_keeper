### Qu'est-ce qu'une station suspecte ?
Une station suspecte est une station où l'on ne peut pas prendre de vélo bien que des vélos soient disponibles. Cela peut provenir de plusieurs problèmes techniques.

### Comment ça marche ?
Un algorithme analyse l'activité des stations, si une station a une activité plus faible que la normale un certain temps, alors la station est classifiée en suspecte.

### Une station est suspecte alors qu'elle marche correctement
Il est possible que l'algorithme ait fait une erreur, une faible activité peut être liée à un contexte particulier (jour férié, mauvais temps ...)

### Qu'est-ce qu'une alerte faible (Sur Twitter)
C'est lorsque l'activité de la station est faible (plus que la normale). Il y a des chances que celle-ci rencontre un problème, mais cela correspond à une 1re alerte.

### Qu'est-ce qu'une alerte grave (Sur Twitter)
C'est lorsque l'activité de la station est très faible et que cela dure depuis un certain temps. Il y a de fortes chances que celle-ci rencontre un problème et que cela perturbe les utilisateurs. Cette alerte est la plus grave et représente une absence d'activité sur cette station rarement observée.

### C'est quoi une station inactive
Une station inactive est une station qui a été déconnectée du système des Vcub. TBM peut ainsi déconnecter une station pour diverses raisons (travaux, maintenance ...)

### Pourquoi il y a des stations non surveillées
L'algorithme est basé sur l'activité des stations. Si une station a trop peu d'activité en général, il n'est pas possible de détecter convenablement de possibles anomalies dans son fonctionnement.

### J'ai vu une station qui ne marchait pas or ce n'est pas le cas ici
Pour qu'une station soit suspectée de mauvais fonctionnement, il faut que des vélos y soient disponibles. Nous avons déterminé qu'il fallait au moins 2 vélos disponibles pour que la station soit analysable par l'algorithme (vélos cassés ...)

### Sur les graphiques des stations, il y a des absences de données
Nous nous servons de l'Open Data pour réaliser l'application, nous ne sommes pas directement connectés au service TBM Vcub. Cela peut arriver qu'il y ait des problèmes de connexions.

### Informations sur le projet
Ce projet a été réalisé par Armand GILLES, il a pour but d'être un projet bac à sable tout en étant utile pour la communauté avec des contraintes de production comme dans la vraie vie. Il est très utile pour utlisé les dernières technologies avec des données réelless. Une partie du projet est open-source, il est possible de le retrouver sur GitHub. Le projet est réalisé en Python.

### Pourquoi faire un chatbot
L'idée de faire un chatbot est de rendre l'application plus accessible et de permettre à l'utilisateur d'interagir avec elle de manière plus naturelle. Dans un second temps, c'est aussi pour le côté fun et utilisation d'IA de type LLM. Le chatbot est basé sur un modèle de MistralIA, il est de type "Agentic IA", c'est à dire qu'il peut agir de manière indépendante pour atteindre des objectifs définis. Il ne se contente pas de réagir passivement aux entrées, mais peut initier des actions basées sur son environnement et avec les outils à sa disposition (données internes, calcul géographique, prédictions, RAG etc...).