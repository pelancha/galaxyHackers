"""Script to get dataloaders"""

"""To create dataloaders you need to adress only function create_dataloaders()"""

import os
import sys
from enum import Enum
from pathlib import Path
from zipfile import ZipFile

import astropy.coordinates as coord
import astropy.table as atpy
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wget
from astropy.coordinates import Angle
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from PIL import Image
from pixell import enmap
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

import legacy_for_img
from config import settings


np.random.seed(settings.SEED)

TORCHVISION_MEAN = torch.Tensor([0.485, 0.456, 0.406])
TORCHVISION_STD = torch.Tensor([0.229, 0.224, 0.225])


class DataPart(str, Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    TEST_DR5 = "test_dr5"
    TEST_MC = "test_mc"
    GAIA = "gaia"



class ClusterDataset(Dataset):
    def __init__(self, images_dir_path: str, description_csv_path: str, transform=None):
        super().__init__()

        self.images_dir_path = images_dir_path
        self.description_df = pd.read_csv(
            description_csv_path, dtype={"idx": str, "target": int}
        )

        self.transform = transform

    def __len__(self):
        return len(self.description_df)

    def __getitem__(self, index):
        img_name, label = self.description_df.iloc[index][["idx", "target"]]

        img_path = Path(self.images_dir_path, f"{img_name}.jpg")
        img = self._read_img(img_path)

        if self.transform:
            img = self.transform(img)

        sample = {"image": img, "label": label}

        return sample

    @staticmethod
    def _read_img(img_path: Path):
        img = Image.open(str(img_path.resolve()))
        return img



def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


"""Obtain GAIA stars catalogue"""


def read_gaia():
    job = Gaia.launch_job_async(
        "select DESIGNATION, ra, dec from gaiadr3.gaia_source "
        "where random_index between 0 and 1000000 and phot_g_mean_mag < 12 and parallax is not null"
    )
    gaiaResponse = job.get_results().to_pandas()
    data_gaia = (
        gaiaResponse.sample(frac=1, random_state=settings.SEED)
        .reset_index(drop=True)
        .rename(columns={"DESIGNATION": "name", "ra": "ra_deg", "dec": "dec_deg"})
    )
    return data_gaia


"""Obtain ACT_DR5, clusters identified there and in MaDCoWS"""


def download_data():

    for config in [settings.MAP_ACT_CONFIG, settings.DR5_CONFIG]:

        if not os.path.exists(config.OUTPUT_PATH):
            try:
                wget.download(
                    url=config.URL, out=settings.DATA_PATH, bar=bar_progress
                )
                with ZipFile(config.ZIPPED_OUTPUT_PATH, "r") as zObject:
                    zObject.extractall(path=settings.DATA_PATH)
                rename_dict = config.RENAME_DICT

                os.rename(rename_dict.SOURCE, rename_dict.TARGET)
                os.remove(config.ZIPPED_OUTPUT_PATH)
            except Exception:
                # Getting 403, what credentials needed?
                wget.download(
                    url=config.FALLBACK_URL,
                    out=config.OUTPUT_PATH,
                    bar=bar_progress,
                )


def read_dr5():

    dr5: atpy.Table = atpy.Table().read(settings.DR5_CLUSTERS_PATH)
    dr5_frame = dr5.to_pandas().reset_index(drop=True)

    dr5_frame["name"] = dr5_frame["name"].astype(str)

    dr5_frame = dr5_frame.rename(columns={"RADeg": "ra_deg", "decDeg": "dec_deg"})

    return dr5_frame


def to_hms_format(time_str):
    parts = time_str.split()
    return f"{parts[0]}h{parts[1]}m{parts[2]}s"


def to_dms_format(time_str):
    parts = time_str.split()
    return f"{parts[0]}d{parts[1]}m{parts[2]}s"


def read_mc():
    # the catalogue of MaDCoWS in VizieR
    CATALOGUE = "J/ApJS/240/33/"

    catalog_list = Vizier.find_catalogs(CATALOGUE)

    Vizier.ROW_LIMIT = -1
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    interesting_table: atpy.Table = catalogs[os.path.join(CATALOGUE, "table3")]
    madcows_table = interesting_table.to_pandas().reset_index(drop=True)

    madcows_table["ra_deg"] = madcows_table["RAJ2000"].apply(
        lambda x: Angle(to_hms_format(x)).degree
    )
    madcows_table["dec_deg"] = madcows_table["DEJ2000"].apply(
        lambda x: Angle(to_dms_format(x)).degree
    )

    return madcows_table.rename(columns={"Name": "name"})


def get_all_clusters():
    """Concat clusters from act_dr5 and madcows to create negative classes in samples"""

    mc = read_mc()
    dr5 = read_dr5()

    needed_cols = ["name", "ra_deg", "dec_deg"]
    clusters_dr5_mc = pd.concat([dr5[needed_cols], mc[needed_cols]], ignore_index=True)

    return clusters_dr5_mc


def get_cluster_catalog() -> coord.SkyCoord:

    clusters = get_all_clusters()

    # The catalog of known found galaxies
    catalog = coord.SkyCoord(
        ra=clusters["ra_deg"] * u.degree, dec=clusters["dec_deg"] * u.degree, unit="deg"
    )

    return catalog


def filter_candiates(candidates: coord.SkyCoord, max_len: int) -> coord.SkyCoord:

    catalog = get_cluster_catalog()

    _, d2d, _ = candidates.match_to_catalog_sky(catalog)

    MIN_ANGLE = 10
    MAX_ANGLE = 20

    candidates_filter = (d2d.arcmin > MIN_ANGLE) & (
        candidates.galactic.b.degree > MAX_ANGLE
    )

    filtered_candidates = candidates[candidates_filter][:max_len]

    return filtered_candidates


def candidates_to_df(candidates: coord.SkyCoord) -> pd.DataFrame:

    b_values = candidates.galactic.b.degree
    l_values = candidates.galactic.l.degree

    names = [f"Rand {l:.3f}{b:+.3f}" for l, b in zip(l_values, b_values)]

    data = pd.DataFrame(
        np.array([names, candidates.ra.deg, candidates.dec.deg]).T,
        columns=["name", "ra_deg", "dec_deg"],
    )

    return data


def generate_candidates_dr5() -> coord.SkyCoord:

    # Needed only for reading metadata and map generation?
    imap_98 = enmap.read_fits(settings.MAP_ACT_PATH)[0]

    # Generating positions of every pixel of telescope's sky zone
    positions = np.array(np.rad2deg(imap_98.posmap()))
    ras, decs = positions[1].ravel(), positions[0].ravel()

    # Shuffling candidates, imitating samples
    np.random.seed(settings.SEED)
    np.random.shuffle(ras)
    np.random.shuffle(decs)

    # Just points from our sky map
    candidates = coord.SkyCoord(ra=ras * u.degree, dec=decs * u.degree, unit="deg")

    return candidates


def generate_candidates_mc():
    """Create sample from MadCows catalogue"""

    n_sim = 10_000

    np.random.seed(settings.SEED)

    ras = np.random.uniform(0, 360, n_sim)
    decs = np.random.uniform(-90, 90, n_sim)

    # Just points from our sky map
    candidates = coord.SkyCoord(ra=ras * u.degree, dec=decs * u.degree, unit="deg")

    return candidates


def create_negative_class_dr5():
    """Create sample from dr5 clsuter catalogue"""

    dr5 = read_dr5()

    candidates = generate_candidates_dr5()

    filtered_candidates = filter_candiates(candidates, max_len=len(dr5))

    frame = candidates_to_df(filtered_candidates)

    return frame


def create_negative_class_mc():

    mc = read_mc()

    candidates = generate_candidates_mc()

    filtered_candidates = filter_candiates(candidates, max_len=len(mc))

    frame = candidates_to_df(filtered_candidates)

    return frame


def create_data_dr5():
    clusters = read_dr5()
    clusters = clusters[["name", "ra_deg", "dec_deg"]]
    clusters["target"] = 1
    random = create_negative_class_dr5()
    random["target"] = 0
    data_dr5 = pd.concat([clusters, random]).reset_index(drop=True)
    data_dr5[["ra_deg", "dec_deg"]] = data_dr5[["ra_deg", "dec_deg"]].astype(float)

    return data_dr5


def create_data_mc():
    clusters = read_mc()
    clusters = clusters[["name", "ra_deg", "dec_deg"]]
    clusters["target"] = 1
    random = create_negative_class_mc()
    random["target"] = 0
    data_mc = pd.concat([clusters, random]).reset_index(drop=True)

    data_mc[["ra_deg", "dec_deg"]] = data_mc[["ra_deg", "dec_deg"]].astype(float)
    return data_mc


"""Split samples into train, validation and tests and get pictures from legacy survey"""


def train_val_test_split():
    dr5 = create_data_dr5()
    test_mc = create_data_mc()



    for part in list(DataPart):
        path = os.path.join(settings.DATA_PATH, part.value)
        os.makedirs(path, exist_ok=True)


    train, validate, test_dr5 = np.split(
        dr5.sample(frac=1, random_state=1), [int(0.6 * len(dr5)), int(0.8 * len(dr5))]
    )

    gaia = read_gaia()


    pairs = [
        (DataPart.TRAIN, train),
        (DataPart.VALIDATE, validate),
        (DataPart.TEST_DR5, test_dr5),
        (DataPart.TEST_MC, test_mc),
        (DataPart.GAIA, gaia)
    ]

    return pairs


def ddos():

    description_path = os.path.join(settings.DATA_PATH, "description")
    os.makedirs(description_path, exist_ok=True)

    pairs = train_val_test_split()
    for part, description in pairs:

        description_file_path = os.path.join(description_path, f"{part.value}.csv")
        description.to_csv(description_file_path, index=False)

        path = os.path.join(settings.DATA_PATH, part.value)
        legacy_for_img.grab_cutouts(
            target_file=description,
            name_col="name",
            ra_col="ra_deg",
            dec_col="dec_deg",
            output_dir=path,
            survey="unwise-neo7",
            imgsize_pix=224*8 if part == DataPart.GAIA else 224,
            file_format="jpg",
        )


"""Create dataloaders"""


def show_original(img):
    denormalized_img = img.clone()
    for channel, m, s in zip(denormalized_img, TORCHVISION_MEAN, TORCHVISION_STD):
        channel.mul_(s).add_(m)

    denormalized_img = denormalized_img.numpy()
    plt.imshow(np.transpose(denormalized_img, (1, 2, 0)))


main_transforms = [
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=TORCHVISION_MEAN, std=TORCHVISION_STD),
]

def check_catalogs():

    is_map = os.path.exists(settings.MAP_ACT_PATH)
    is_dr5 = os.path.exists(settings.DR5_CLUSTERS_PATH)

    if not is_map or not is_dr5:
        download_data()

def create_dataloaders():

    check_catalogs()

    ddos()

    data_transforms = {
        DataPart.TRAIN: transforms.Compose(
            [
                *main_transforms,
                transforms.RandomRotation(
                    15,
                ),
                transforms.RandomHorizontalFlip(),
            ]
        ),
        DataPart.VALIDATE: transforms.Compose(main_transforms),
        DataPart.TEST_DR5: transforms.Compose(main_transforms),
        DataPart.TEST_MC: transforms.Compose(main_transforms),
    }

    custom_datasets = {}
    dataloaders = {}
    for part in list(DataPart):

        dataset = ClusterDataset(
            os.path.join(settings.DATA_PATH, part.value),
            os.path.join(settings.DATA_PATH, "description", f"{part.value}.csv"),
            transform=data_transforms[part],
        )

        custom_datasets[part] = dataset
        dataloaders[part] = DataLoader(dataset, batch_size=64)

    # Get a batch of training data
    batch = next(iter(dataloaders[DataPart.TRAIN]))

    # Make a grid from batch
    out = utils.make_grid(batch["image"])
    show_original(out)

    return custom_datasets, dataloaders
