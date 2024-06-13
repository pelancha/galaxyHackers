'''Script to get dataloaders'''
'''To create dataloaders you need to adress only function create_dataloaders()'''

import os
from os.path import exists
import wget
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
import astropy.table as atpy
from astropy.coordinates import Angle
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from pixell import enmap
import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import skimage
import data.legacy_for_img as legacy_for_img
from zipfile import ZipFile 

import settings
import sys
np.random.seed(settings.SEED)

def bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()

'''Obtain GAIA stars catalogue'''

def read_gaia():
    job = Gaia.launch_job_async("select DESIGNATION, ra, dec from gaiadr3.gaia_source "
                                "where random_index between 0 and 1000000 and phot_g_mean_mag < 12 and parallax is not null")
    gaiaResponse = job.get_results().to_pandas()
    data_gaia = gaiaResponse.sample(n=1000).reset_index(drop=True).rename(columns={"DESIGNATION": "Component_name", "ra": "RA", "dec": "DEC"})
    return data_gaia

'''Obtain ACT_DR5, clusters identified there and in MaDCoWS'''

def download_data():

    for config in [settings.MAP_ACT_CONFIG, settings.DR5_CONFIG]:

        if not os.path.exists(config["OUTPUT_PATH"]):
            try:
                wget.download(url=config["URL"], out=settings.STORAGE_PATH, bar=bar_progress)
                with ZipFile(config["ZIPPED_OUTPUT_PATH"], 'r') as zObject: 
                    zObject.extractall(path=settings.STORAGE_PATH)
                rename_dict = config["RENAME_DICT"]

                os.rename(rename_dict["SOURCE"], rename_dict["TARGET"])
                os.remove(config["ZIPPED_OUTPUT_PATH"])
            except Exception:
                # Getting 403, what credentials needed?
                wget.download(url=config['FALLBACK_URL'], out=config["OUTPUT_PATH"], bar=bar_progress)

def read_dr5():
   
    dr5: atpy.Table = atpy.Table().read(settings.DR5_CLUSTERS_PATH)
    dr5_frame = dr5.to_pandas().reset_index(drop=True)
    
    dr5_frame["name"] = dr5_frame["name"].astype(str)

    dr5_frame = dr5_frame.rename(
    columns={
     "RADeg": "ra_deg",
     "decDeg": "dec_deg"}
)

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

    interesting_table: atpy.Table = catalogs[os.path.join(CATALOGUE,  "table3")]
    madcows_table = interesting_table.to_pandas().reset_index(drop=True)

    madcows_table["ra_deg"] = madcows_table["RAJ2000"].apply(lambda x: Angle(to_hms_format(x)).degree)
    madcows_table["dec_deg"] = madcows_table["DEJ2000"].apply(lambda x: Angle(to_dms_format(x)).degree)


    return madcows_table.rename(columns={"Name": "name"})


def get_all_clusters():

    '''Concat clusters from act_dr5 and madcows to create negative classes in samples'''

    mc = read_mc()
    dr5 = read_dr5()

    needed_cols = ["name", "ra_deg", "dec_deg"]
    clusters_dr5_mc = pd.concat([dr5[needed_cols], mc[needed_cols]], ignore_index=True)

    return clusters_dr5_mc


def get_cluster_catalog() -> coord.SkyCoord:

    clusters = get_all_clusters()
   
    # The catalog of known found galaxies
    catalog = coord.SkyCoord(
        ra = clusters['ra_deg']*u.degree, 
        dec = clusters['dec_deg']*u.degree, 
        unit = 'deg'
        )
    
    return catalog
    

def filter_candiates(candidates: coord.SkyCoord, max_len: int) -> coord.SkyCoord:
    
    catalog = get_cluster_catalog()

    _, d2d, _ = candidates.match_to_catalog_sky(catalog)

    MIN_ANGLE = 10
    MAX_ANGLE = 20

    candidates_filter = (d2d.arcmin>MIN_ANGLE) & (candidates.galactic.b.degree > MAX_ANGLE)

    filtered_candidates = candidates[candidates_filter][: max_len]

    return filtered_candidates


def candidates_to_df(candidates: coord.SkyCoord) -> pd.DataFrame:

    b_values = candidates.galactic.b.degree
    l_values = candidates.galactic.l.degree

    names = [f'Rand {l:.3f}{b:+.3f}' for l, b in zip(l_values, b_values)]

    data = pd.DataFrame(
        np.array([
            names,
            candidates.ra.deg, 
            candidates.dec.deg
        ]).T,
        columns = ['name', 'ra_deg', 'dec_deg']
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
    candidates = coord.SkyCoord(
        ra = ras*u.degree, 
        dec = decs*u.degree, 
        unit = 'deg'
    )

    return candidates


def generate_candidates_mc():
    """Create sample from MadCows catalogue"""

    n_sim = 10_000

    np.random.seed(settings.SEED)

    ras = np.random.uniform(0, 360, n_sim)
    decs = np.random.uniform(-90, 90, n_sim)


    # Just points from our sky map
    candidates = coord.SkyCoord(
        ra = ras*u.degree, 
        dec = decs*u.degree, 
        unit = 'deg'
    )

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
    clusters = clusters[['name', 'ra_deg', 'dec_deg']]
    clusters['target'] = 1
    random = create_negative_class_dr5()
    random['target'] = 0
    data_dr5 = pd.concat([clusters, random]).reset_index(drop=True)

    data_dr5[["ra_deg", "dec_deg"]] =  data_dr5[["ra_deg", "dec_deg"]].astype(float)

    return data_dr5

def create_data_mc():
    clusters = read_mc()
    clusters = clusters[['name', 'ra_deg', 'dec_deg']]
    clusters['target'] = 1
    random = create_negative_class_mc()
    random['target'] = 0
    data_mc = pd.concat([clusters, random]).reset_index(drop=True)

    data_mc[["ra_deg", "dec_deg"]] =  data_mc[["ra_deg", "dec_deg"]].astype(float)

    return data_mc

# class MNISTDataset(Dataset):
#     def __init__(self, images_dir_path: str,
#                  description_csv_path: str):
#         super().__init__()
        
#         self.images_dir_path = images_dir_path
#         self.description_df = pd.read_csv(description_csv_path,
#                                            dtype={'image_name': str, 'label': int})

#     def __len__(self):
#         return len(self.description_df)
    
#     def __getitem__(self, index):
#         img_name, label = self.description_df.iloc[index, :]
        
#         img_path = Path(self.images_dir_path, f'{img_name}.png')
#         img = self._read_img(img_path)
        
#         return dict(sample=img, label=label)
    
#     @staticmethod
#     def _read_img(img_path: Path):
#         img = cv2.imread(str(img_path.resolve()))
#         img = img.astype(np.float32)
#         img = np.transpose(img, (2, 0, 1))
        
#         return img

"""Split samples into train, validation and tests and get pictures from legacy survey"""

def train_val_test_split():
    dr5 = create_data_dr5()
    test_mc = create_data_mc()

    folders = ['train', 'validate', 'test_dr5', 'test_mc']

    for folder in folders:
        path = os.path.join(settings.STORAGE_PATH, folder)
        os.makedirs(path, exist_ok=True)

    train, validate, test_dr5 = \
              np.split(dr5.sample(frac=1, random_state=1), 
                       [int(.6*len(dr5)), int(.8*len(dr5))])
    

    pairs = [
        ("train", train),
        ("validate", validate),
        ("test_dr5", test_dr5),
        ("test_mc", test_mc)
    ]

    return pairs


def ddos():
    
    pairs = train_val_test_split()
    for folder, description in pairs:
        
        path = os.path.join(settings.STORAGE_PATH, folder)
        legacy_for_img.grab_cutouts(
            target_file=description, 
            name_col="name",
            ra_col="ra_deg",
            dec_col="dec_deg",
            output_dir=path, 
            survey='unwise-neo7', 
            imgsize_pix=224, 
            file_format='jpg'
            )
"""Create dataloaders"""

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.507, 0.487, 0.441])
    std = np.array([0.267, 0.256, 0.276])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.pause(0.001)  # pause a bit so that plots are updated


def create_dataloaders():
    if (not os.path.exists(location) or 
      len(os.listdir(location)) == 0):
        os.makedirs(data_out, exist_ok=True)
        try:
            wget.download(url=data_wget, out=data_out)
            with ZipFile(f"{zipped_data_out}", 'r') as zObject: 
                zObject.extractall(path=f"{data_out}")
            os.remove(f"{zipped_data_out}")
        except:
            ddos()
        else:
            if (not os.path.exists(location) or 
                  len(os.listdir(location)) == 0):
                ddos()

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomRotation(15,),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),
         'test_dr5': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),
          'test_madcows': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),
    }

    folderlocation = f'{working_path}{location}'

    image_datasets = {x: datasets.ImageFolder(os.path.join(folderlocation, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test_dr5', 'test_madcows']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                 shuffle=True, num_workers=3)
                  for x in ['train', 'val', 'test_dr5', 'test_madcows']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test_dr5', 'test_madcows']}

    class_names = image_datasets['train'].classes

    print(f'There are {len(class_names)} classes\nSize of train is {dataset_sizes["train"]}\n\tvalidation is {dataset_sizes["val"]}\n\ttest_dr5 is {dataset_sizes["test_dr5"]}\n\ttest_madcows is {dataset_sizes["test_madcows"]}')


    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = utils.make_grid(inputs)
    imshow(out)

    return dataloaders
