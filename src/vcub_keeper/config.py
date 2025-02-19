import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Change config env in production
IS_PROD = False

# Paths
try:
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    print("ROOT_DIR: ", ROOT_DIR)
except:  # noqa: E722
    # Case of heroku env var
    print("Try to find environnement variable")
    ROOT_DIR = os.environ.get("ROOT_DIR")
    IS_PROD = True

# If package is install with pip
# if "site-packages" in os.path.dirname(os.path.dirname(os.path.realpath(__file__))):  # install via pip
if "site-packages" in str(Path(__file__).resolve().parent):  # install via pip
    from dotenv import load_dotenv

    print("In site-packages")
    load_dotenv()
    ROOT_DIR = Path(os.environ.get("ROOT_DIR", ""))
    print("ROOT_DIR from environment variable: ", ROOT_DIR)
    # On verifie si le repertoire existe bien
    if ROOT_DIR.exists():
        print("ROOT_DIR from environment variable exists: ", ROOT_DIR)
    else:
        # On essaye de trouver le repertoire (utile pour le dev hors linux)
        print("ROOT_DIR from environment variable does not exist")
        ROOT_DIR = str(Path.cwd()).split("vcub_keeper")[0] + "vcub_keeper"
        print("ROOT_DIR from cwd: ", ROOT_DIR)
    IS_PROD = True

# In case where ROOT_DIR is None (pre-prod) but we don't need these variables
try:
    ROOT_DATA_RAW = str(ROOT_DIR) + "/data/raw/"
    ROOT_DATA_CLEAN = str(ROOT_DIR) + "/data/clean/"
    ROOT_DATA_REF = str(ROOT_DIR) + "/data/ref/"
    ROOT_MODEL = str(ROOT_DIR) + "/model/"
    ROOT_TESTS_DATA = str(ROOT_DIR) + "/tests/data_for_tests/"
except:  # noqa: E722
    print("Can't have repository variables")
    ROOT_DATA_REF = ""  # https://github.com/armgilles/vcub_keeper/issues/56#issuecomment-1007593715

# Only in dev
if IS_PROD is False:
    # ROOT_DATA_RAW
    if not os.path.exists(ROOT_DATA_RAW):
        os.mkdir(ROOT_DATA_RAW)
        print("Create " + ROOT_DATA_RAW)

    # ROOT_DATA_CLEAN
    if not os.path.exists(ROOT_DATA_CLEAN):
        os.mkdir(ROOT_DATA_CLEAN)
        print("Create " + ROOT_DATA_CLEAN)

    # ROOT_DATA_REF
    if not os.path.exists(ROOT_DATA_REF):
        os.mkdir(ROOT_DATA_REF)
        print("Create " + ROOT_DATA_REF)

    # ROOT_MODEL
    if not os.path.exists(ROOT_MODEL):
        os.mkdir(ROOT_MODEL)
        print("Create " + ROOT_MODEL)

    # ROOT_TESTS_DATA
    if not os.path.exists(ROOT_TESTS_DATA):
        os.mkdir(ROOT_TESTS_DATA)
        print("Create " + ROOT_TESTS_DATA)

SEED = 2020

# Key api meteo
API_METEO = os.getenv("API_METEO")
# Key api mapbox
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
# Key Open Data Bordeaux
# https://opendata.bordeaux-metropole.fr/explore/dataset/ci_vcub_p/information/
KEY_API_BDX = os.getenv("KEY_API_BDX")

# Station Vcub ID non user
# https://github.com/armgilles/vcub_keeper/issues/29#issuecomment-703246491
NON_USE_STATION_ID = [244, 249, 250, 138]


# Features to use during clustering
FEATURES_TO_USE_CLUSTER = [
    "consecutive_no_transactions_out",
    "Sin_quarter",
    "Cos_quarter",
    "Sin_weekday",
    "Cos_weekday",
    "Sin_hours",
    "Cos_hours",
]

# Station profile rules to determine contamination anomalies
# based on ROOT_DATA_REF/station_profile.csv
PROFILE_STATION_RULE = {
    "very high": 36,  # 6 heures
    "hight": 54,  # 9 heures
    "medium": 72,  # 12 heures
    "low": 144,  # 24 heures
}

# Utiliser dans ml/train_cluster.py. Permet d'apprendre uniquement les stations
# avec un certain niveau d'activité
THRESHOLD_PROFILE_STATION = 0.12
