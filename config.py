from dynaconf import Dynaconf
import os
import urllib 

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml', '.secrets.toml'],
)

settings.STORAGE_PATH = os.path.join(settings.WORKDIR, "storage/")
settings.METRICS_PATH = os.path.join(settings.STORAGE_PATH, "metrics/")
settings.DATA_PATH = os.path.join(settings.STORAGE_PATH, "data/")
settings.BEST_MODELS_PATH = os.path.join(settings.STORAGE_PATH, "best_models/")

settings.MAP_ACT_PATH = os.path.join(settings.DATA_PATH, settings.MAP_ACT_FILENAME)
settings.DR5_CLUSTERS_PATH = os.path.join(settings.DATA_PATH, settings.DR5_CLUSTERS_FILENAME)

settings.MAP_ACT_CONFIG = {
    "RENAME_DICT" : {
        "SOURCE": os.path.join(settings.DATA_PATH, settings.MAP_ACT_ROUTE), 
        "TARGET": settings.MAP_ACT_PATH
        },
    "URL": "http://oshi.at/JTuj",
    "ZIPPED_OUTPUT_PATH": os.path.join(settings.DATA_PATH, "ACT.zip"),
    "FALLBACK_URL" : urllib.parse.urljoin(settings.ARCHIVE_DR5_URL, 'maps/', settings.MAP_ACT_ROUTE),
    "OUTPUT_PATH": settings.MAP_ACT_PATH
}

settings.DR5_CONFIG = {
     "RENAME_DICT" : {
         "SOURCE": os.path.join(settings.DATA_PATH, settings.DR5_CLUSTERS_ROUTE), 
         "TARGET":  settings.DR5_CLUSTERS_PATH 
         },
    "URL": "http://oshi.at/ysSg",
    "ZIPPED_OUTPUT_PATH": os.path.join(settings.DATA_PATH, "dr5.zip"),
    "FALLBACK_URL" : urllib.parse.urljoin(settings.ARCHIVE_DR5_URL, settings.DR5_CLUSTERS_ROUTE),
    "OUTPUT_PATH": settings.DR5_CLUSTERS_PATH 
}

required_paths = [
    settings.STORAGE_PATH, 
    settings.METRICS_PATH, 
    settings.DATA_PATH, 
    settings.BEST_MODELS_PATH
]

for path in required_paths:
    os.makedirs(path, exist_ok=True)


# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
