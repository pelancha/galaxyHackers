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


def createNegativeClassDR5():

    """Create sample from dr5 clsuter catalogue"""

    clusters = get_all_clusters()
    
   
    # The catalog of known found galaxies
    catalog = coord.SkyCoord(
        ra = clusters['ra_deg']*u.degree, 
        dec = clusters['dec_deg']*u.degree, 
        unit = 'deg'
        )
    

    dr5 = read_dr5()

    # Needed only for reading metadata and map generation?
    imap_98 = enmap.read_fits(settings.MAP_ACT_PATH)[0]

    positions = np.array(np.rad2deg(imap_98.posmap()))
    ras, decs = positions[1].ravel(), positions[0].ravel()


    # Just points from our sky map
    candidates = coord.SkyCoord(
        ra = ras*u.degree, 
        dec = decs*u.degree, 
        unit = 'deg'
    )

    _, d2d, _ = candidates.match_to_catalog_sky(catalog)

    MIN_ANGLE = 10
    MAX_ANGLE = 20

    candidates_filter = (d2d.arcmin>MIN_ANGLE) & (candidates.galactic.b.degree > MAX_ANGLE)

    filtered_candidates = candidates[candidates_filter]

    b_values = filtered_candidates.galactic.b.degree
    l_values = filtered_candidates.galactic.l.degree

    names = [f'Rand {l:.3f}{b:+.3f}' for l, b in zip(l_values, b_values)]
    filtered_candidates.ra.deg
    filtered_candidates.dec.deg

    data = pd.DataFrame(
        np.array([
            names,
            filtered_candidates.ra.deg, 
            filtered_candidates.dec.deg
        ]).T,
        columns = ['name', 'ra_deg', 'dec_deg']
    )

    # Shuffling points to imitate sampling
    data.sample(frac=1, replace=False, random_state=settings.SEED)

    # Truncating if too much samples
    # TODO Discuss the aim of sampling technique and improve it
    data = data.iloc[:len(dr5)]

    return data


def create_data_dr5():
    clusters = read_dr5()

    clusters = clusters[['name', 'RADeg', 'decDeg']].reset_index(drop=True)
    clusters.rename(columns = {'name': 'Component_name', 'RADeg': 'RA', 'decDeg': 'DEC'}, inplace = True )
    clusters['target'] = 1
    random = createNegativeClassDR5()
    random['target'] = 0
    data_dr5 = pd.concat([clusters, random]).reset_index(drop=True)

    return data_dr5

'''Randomiser for sample from MaDCoWS'''

def createNegativeClassRac(x):
    randChoice = np.random.normal(-15, 15)
    while (x + randChoice) > 360 or (x + randChoice) < 0:
      randChoice = np.random.normal(-15, 15)
    return x + randChoice


def createNegativeClassDec(x):
    randChoice = np.random.normal(-15, 15)
    while (x + randChoice) > 90 or (x + randChoice) < -90:
        randChoice = np.random.normal(-15, 15)
    return x + randChoice

"""Create sample from MadCows catalogue"""

def createNegativeClassMC(radegMC, decdegMC):

    clusters = get_all_clusters()


    
    clustersDr5_MC = concat_tables()

    rac = radegMC[np.random.choice(len(radegMC), size=10000)].apply(lambda x: createNegativeClassRac(x))
    dec = decdegMC[np.random.choice(len(decdegMC), size=10000)].apply(lambda x: createNegativeClassDec(x))
    rac, dec = rac.ravel(), dec.ravel()

    ra, de, name = [], [], []
    c = coord.SkyCoord(ra = clustersDr5_MC['RADeg']*u.degree, dec = clustersDr5_MC['decDeg']*u.degree, unit = 'deg')

    for rac_val, dec_val in zip(rac, dec):
        coords = coord.SkyCoord(ra=rac_val*u.degree, dec=dec_val*u.degree, frame='icrs')

        idx, d2d, d3d = coords.match_to_catalog_sky(c)
        if d2d.arcmin < 10:
            continue

        b, l = coords.galactic.b.degree, coords.galactic.l.degree

        if b > 20:
            ra.append(coords.ra.degree)
            de.append(coords.dec.degree)
            name.append(f'Rand {l:.3f}{b:+.3f}')
            if len(ra) == len(radegMC): # number of madcows clusters
                break

    n = len(ra)

    dfNegativeFromMadcows = pd.DataFrame({'Component_name': name, 'RA': ra, 'DEC': de})

    return dfNegativeFromMadcows

def create_data_madcows():
    madcows_table = readMC()
    radegMC = madcows_table.iloc[:, 1].apply(lambda x: Angle(toHmsFormat(x)).degree)
    decdegMC = madcows_table.iloc[:, 2].apply(lambda x: Angle(toDmsFormat(x)).degree)
    clusters = pd.DataFrame({'Component_name': madcows_table['Name'], 'RA': radegMC, 'DEC': decdegMC})
    clusters['target'] = 1
    random = createNegativeClassMC(radegMC, decdegMC)
    random['target'] = 0
    data_madcows = pd.concat([clusters, random]).reset_index(drop=True)
    return data_madcows

"""Split samples into train, validation and tests and get pictures from legacy survey"""

def train_val_test_split():
    data_dr5 = create_data_dr5()
    data_madcows = create_data_madcows()

    folderlocation = f'{working_path}{location}'
    folders = ['train', 'val', 'test_dr5', 'test_madcows']

    for folder in folders:
        path = os.path.join(folderlocation, folder)
        os.makedirs(path, exist_ok=True)

        for iter1 in range(2):
            subpath = os.path.join(path, str(iter1))
            os.makedirs(subpath, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(data_dr5.index, data_dr5['target'], test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    train = data_dr5.iloc[X_train].reset_index(drop=True)
    val = data_dr5.iloc[X_val].reset_index(drop=True)
    test_dr5 = data_dr5.iloc[X_test].reset_index(drop=True)

    train_0 = train[train.target == 0].reset_index(drop=True)
    train_1 = train[train.target == 1].reset_index(drop=True)

    val_0 = val[val.target == 0].reset_index(drop=True)
    val_1 = val[val.target == 1].reset_index(drop=True)

    test_dr5_0 = test_dr5[test_dr5.target == 0].reset_index(drop=True)
    test_dr5_1 = test_dr5[test_dr5.target == 1].reset_index(drop=True)

    test_madcows_0 = data_madcows[data_madcows.target == 0].reset_index(drop=True)
    test_madcows_1 = data_madcows[data_madcows.target == 1].reset_index(drop=True)

    list_train, list_val = [train_0, train_1], [val_0, val_1]
    list_test_dr5, list_test_MC = [test_dr5_0, test_dr5_1], [test_madcows_0, test_madcows_1]
    return list_train, list_val, list_test_dr5, list_test_MC


def ddos():
    list_train, list_val, list_test_dr5, list_test_MC = train_val_test_split()
    train_0, train_1 = list_train
    val_0, val_1 = list_val
    test_dr5_0, test_dr5_1 = list_test_dr5
    test_madcows_0, test_madcows_1 = list_test_MC

    folders = [(train_0, 'train/0'), (train_1, 'train/1'), (val_0, 'val/0'), (val_1, 'val/1'),
               (test_dr5_0, 'test_dr5/0'), (test_dr5_1, 'test_dr5/1'), (test_madcows_0, 'test_madcows/0'),
               (test_madcows_1, 'test_madcows/1')]
    for subfolder, subfolder_name in folders:
        output_dir = f'{working_path}{location}{subfolder_name}'
        legacy_for_img.grab_cutouts(target_file=subfolder, output_dir=output_dir, survey='unwise-neo7', imgsize_pix=224, file_format='jpg')

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
