import os
import numpy as np

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# If package is install with pip, ROOT_DIR is bad
if "site-packages" in ROOT_DIR:  # isntall via pip
    ROOT_DIR = os.path.realpath("")

ROOT_DATA_RAW = ROOT_DIR + '/data/raw/'
ROOT_DATA_CLEAN = ROOT_DIR + '/data/clean/'
ROOT_DATA_REF = ROOT_DIR + '/data/ref/'

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

SEED = 2020

# Key api meteo
API_METEO = os.getenv("API_METEO")

# Station Vcub ID non user
# https://github.com/armgilles/vcub_keeper/issues/29#issuecomment-703246491
NON_USE_STATION_ID = [244, 249, 250]