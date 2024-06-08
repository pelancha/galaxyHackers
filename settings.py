
import os
import urllib.parse

WORKDIR = "./"
STORAGE_PATH = os.path.join(WORKDIR, "storage/")

MAP_ACT_ROUTE =  "act_planck_dr5.01_s08s18_AA_f220_daynight_fullivar.fits"
DR5_CLUSTERS_ROUTE = "DR5_cluster-catalog_v1.1.fits"
ARCHIVE_DR5_URL = "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/"

MAP_ACT_FILENAME = "map_act.fits"
DR5_CLUSTERS_FILENAME = "dr5.fits"

MAP_ACT_PATH = os.path.join(STORAGE_PATH, MAP_ACT_FILENAME)
DR5_CLUSTERS_PATH = os.path.join(STORAGE_PATH, DR5_CLUSTERS_FILENAME)

MAP_ACT_CONFIG = {
    "RENAME_DICT" : {
        "SOURCE": os.path.join(STORAGE_PATH, MAP_ACT_ROUTE), 
        "TARGET": MAP_ACT_PATH
        },
    "URL": "http://oshi.at/JTuj",
    "ZIPPED_OUTPUT_PATH": os.path.join(STORAGE_PATH, "ACT.zip"),
    "FALLBACK_URL" : urllib.parse.urljoin(ARCHIVE_DR5_URL, 'maps/', MAP_ACT_ROUTE),
    "OUTPUT_PATH": MAP_ACT_PATH
}

DR5_CONFIG = {
     "RENAME_DICT" : {
         "SOURCE": os.path.join(STORAGE_PATH, DR5_CLUSTERS_ROUTE), 
         "TARGET":  DR5_CLUSTERS_PATH 
         },
    "URL": "http://oshi.at/ysSg",
    "ZIPPED_OUTPUT_PATH": os.path.join(STORAGE_PATH, "dr5.zip"),
    "FALLBACK_URL" : urllib.parse.urljoin(ARCHIVE_DR5_URL, DR5_CLUSTERS_ROUTE),
    "OUTPUT_PATH": DR5_CLUSTERS_PATH 
}


for path in [STORAGE_PATH]:
    os.makedirs(path, exist_ok=True)


mapACT = "act_planck_dr5.01_s08s18_AA_f220_daynight_fullivar.fits"
dr5_clusters = "DR5_cluster-catalog_v1.1.fits"
archieve_dr5 = "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/"

mapACT_url =  urllib.parse.urljoin(archieve_dr5, 'maps/', mapACT)
dr5_clusters_url = urllib.parse.urljoin(archieve_dr5, dr5_clusters)

'''Links to download collected data are available until August, 10'''
act_wget = "http://oshi.at/JTuj"
dr5_wget = "http://oshi.at/ysSg"
data_wget = "http://oshi.at/GtZB"
example_wget = "http://oshi.at/bJwn"
bigSegMap_wget = "http://oshi.at/pfXj"

zipped_act_out = os.path.join(STORAGE_PATH, "ACT.zip")
zipped_dr5_out = os.path.join(STORAGE_PATH, "dr5.zip")

mapACT_out = os.path.join(STORAGE_PATH, mapACT)
dr5_clusters_out = os.path.join(STORAGE_PATH, dr5_clusters)

data_out = os.path.join(WORKDIR, "data/")

zipped_data_out = data_out + "DATA.zip"
zipped_example_out = data_out + "example.zip"

segmentation_maps_pics = data_out + "example/"
samples_location = STORAGE_PATH
dr5_sample_location = samples_location + "test_dr5/"
madcows_sample_location = samples_location + "test_madcows/"
gaia_sample_location = samples_location + "test_gaia/"

clusters_out = segmentation_maps_pics + "cl5.csv"
randoms_out = segmentation_maps_pics + "r5.csv"
stars_out = segmentation_maps_pics + "gaia5.csv"

seg_maps = "segmentation_maps/"
bigSegMapLocation = data_out + "example_big/"
bigSegMap_pics = bigSegMapLocation + "Big/"
cl_bigSegMap_out = bigSegMapLocation + "cluster_bigSegMap.csv"
zipped_bigSegMap_out = data_out + "example_big.zip"
