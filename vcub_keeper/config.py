import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Change config env in production
IS_PROD = False

# Paths
try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
except:
    # Case of heroku env var
    print("Try to find environnement variable")
    ROOT_DIR = os.environ.get("ROOT_DIR")
    IS_PROD = True
# If package is install with pip & in dev, ROOT_DIR is bad
if "site-packages" in ROOT_DIR:  # isntall via pip
    ROOT_DIR = os.environ.getenv("ROOT_DIR") # with .env file
    IS_PROD = True

ROOT_DATA_RAW = ROOT_DIR + '/data/raw/'
ROOT_DATA_CLEAN = ROOT_DIR + '/data/clean/'
ROOT_DATA_REF = ROOT_DIR + '/data/ref/'
ROOT_MODEL = ROOT_DIR + '/model/'


# Only in dev
if IS_PROD is False:
    # ROOT_DATA_RAW
    if not os.path.exists(ROOT_DATA_RAW):
        os.mkdir(ROOT_DATA_RAW)
        print('Create '+ROOT_DATA_RAW)

    # ROOT_DATA_CLEAN
    if not os.path.exists(ROOT_DATA_CLEAN):
        os.mkdir(ROOT_DATA_CLEAN)
        print('Create '+ROOT_DATA_CLEAN)

    # ROOT_DATA_REF
    if not os.path.exists(ROOT_DATA_REF):
        os.mkdir(ROOT_DATA_REF)
        print('Create '+ROOT_DATA_REF)

    # ROOT_MODEL
    if not os.path.exists(ROOT_MODEL):
        os.mkdir(ROOT_MODEL)
        print('Create '+ROOT_MODEL)

SEED = 2020

# Key api meteo
API_METEO = os.getenv("API_METEO")

# Station Vcub ID non user
# https://github.com/armgilles/vcub_keeper/issues/29#issuecomment-703246491
NON_USE_STATION_ID = [244, 249, 250]


# Features to use during clustering
FEATURES_TO_USE_CLUSTER = ['consecutive_no_transactions_out', 
                           'Sin_weekday', 'Cos_weekday',
                           'Sin_hours', 'Cos_hours',
                          ]

# Station profile rules to determine contamination anomalies
# based on ROOT_DATA_REF/station_profile.csv
PROFILE_STATION_RULE = {'very high' : 36, # 6 heures
                        'hight' : 54,     # 9 heures
                        'medium' : 66,    # 11 heures
                        'low' : 144       # 24 heures
                       }

# Utiliser dans ml/train_cluster.py. Permet d'apprendre uniquement les stations avec 
# un certain niveau d'activité
THRESHOLD_PROFILE_STATION = 0.3