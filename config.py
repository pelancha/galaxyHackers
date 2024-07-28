from dynaconf import Dynaconf
import os
import urllib 
from pathlib import Path

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml', '.secrets.toml'],
)

settings.STORAGE_PATH = Path(settings.WORKDIR, "storage/")
settings.METRICS_PATH = Path(settings.STORAGE_PATH, "metrics/")
settings.DATA_PATH = Path(settings.STORAGE_PATH, "data/")
settings.DESCRIPTION_PATH = Path(settings.DATA_PATH, "description/")
settings.BEST_MODELS_PATH = Path(settings.STORAGE_PATH, "best_models/")
settings.PREDICTIONS_PATH = Path(settings.STORAGE_PATH, "predictions/")

settings.MAP_ACT_PATH = Path(settings.DATA_PATH, settings.MAP_ACT_FILENAME)
settings.DR5_CLUSTERS_PATH = Path(settings.DATA_PATH, settings.DR5_CLUSTERS_FILENAME)

settings.SEGMENTATION_PATH = Path(settings.STORAGE_PATH, "segmentation/")
settings.SEGMENTATION_SAMPLES_PATH = Path(settings.SEGMENTATION_PATH, "samples/")
settings.SEGMENTATION_SAMPLES_DESCRIPTION_PATH = Path(settings.SEGMENTATION_SAMPLES_PATH, "description/")


settings.MAP_ACT_CONFIG = {
    "RENAME_DICT" : {
        "SOURCE": Path(settings.DATA_PATH, settings.MAP_ACT_ROUTE), 
        "TARGET": settings.MAP_ACT_PATH
        },
    "URL": "http://oshi.at/JTuj",
    "ZIPPED_OUTPUT_PATH": Path(settings.DATA_PATH, "ACT.zip"),
    "FALLBACK_URL" : urllib.parse.urljoin(settings.ARCHIVE_DR5_URL, 'maps/', settings.MAP_ACT_ROUTE),
    "OUTPUT_PATH": settings.MAP_ACT_PATH
}

settings.DR5_CONFIG = {
     "RENAME_DICT" : {
         "SOURCE": Path(settings.DATA_PATH, settings.DR5_CLUSTERS_ROUTE), 
         "TARGET":  settings.DR5_CLUSTERS_PATH 
         },
    "URL": "http://oshi.at/ysSg",
    "ZIPPED_OUTPUT_PATH": Path(settings.DATA_PATH, "dr5.zip"),
    "FALLBACK_URL" : urllib.parse.urljoin(settings.ARCHIVE_DR5_URL, settings.DR5_CLUSTERS_ROUTE),
    "OUTPUT_PATH": settings.DR5_CLUSTERS_PATH 
}


required_paths = [
    settings.STORAGE_PATH, 
    settings.METRICS_PATH, 
    settings.DATA_PATH, 
    settings.DESCRIPTION_PATH,
    settings.BEST_MODELS_PATH,
    settings.SEGMENTATION_PATH,
    settings.SEGMENTATION_SAMPLES_PATH,
    settings.PREDICTIONS_PATH
]


for path in required_paths:
    os.makedirs(path, exist_ok=True)


# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
