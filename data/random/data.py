import legacy_for_img

import os
import sys

import wget

import pandas as pd
import numpy as np
import astropy.table as atpy

import torch, numpy
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from torchvision import datasets, models, transforms, utils
import skimage, astropy.coordinates as coord, astropy.units as u
from torch.utils.data import Dataset, DataLoader

from astroquery.vizier import Vizier

from pixell import enmap
from astropy import units as u
from astropy.coordinates import Angle
import astropy.coordinates as coord

from sklearn.model_selection import train_test_split
from typing import List


def toHmsFormat(s):
    l = s.split()
    s = l[0]+'h'+l[1]+'m'+l[2]+'s'
    return s


def toDmsFormat(s):
    l = s.split()
    s = l[0]+'d'+l[1]+'m'+l[2]+'s'
    return s


# Create negative class from Macdows

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


def read_dr5():
    # Карта покрытия ACT
    wget.download(
        url='https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/act_planck_dr5.01_s08s18_AA_f220_daynight_fullivar.fits',
        out='./act_planck_dr5.01_s08s18_AA_f220_daynight_fullivar.fits')

    # Скопления
    wget.download(url='https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/DR5_cluster-catalog_v1.1.fits',
                  out='./DR5_cluster-catalog_v1.1.fits')


    dr5 = atpy.Table().read('/content/DR5_cluster-catalog_v1.1.fits').to_pandas().reset_index(drop=True)
    dr5['name'] = [str(dr5.loc[i, 'name'], encoding='utf-8') for i in range(len(dr5))]

    return dr5

def readMC():
    # the catalogue of MaCDoWs in VizieR
    CATALOGUE = "J/ApJS/240/33"

    catalog_list = Vizier.find_catalogs(CATALOGUE)

    Vizier.ROW_LIMIT = -1
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    interesting_table = catalogs['J/ApJS/240/33/table3']
    madCows_table = interesting_table.to_pandas().reset_index(drop=True)
    madCows_table = madCows_table.iloc[:, [1, 2, 3]]

    return madCows_table

def concat_tables(args: List):
    dr5 = read_dr5()
    madCows_table = readMC()

    radegDr5 = dr5.loc[:, "RADeg"]
    decdegDr5 = dr5.loc[:, "decDeg"]

    radegMC = madCows_table.iloc[:, 1].apply(lambda x: Angle(toHmsFormat(x)).degree)
    decdegMC = madCows_table.iloc[:, 2].apply(lambda x: Angle(toDmsFormat(x)).degree)

    clustersDr5_MC = pd.DataFrame(
        {
            'name': pd.concat([dr5['name'], madCows_table['Name']], ignore_index=True),
            'RADeg': pd.concat([radegDr5, radegMC], ignore_index=True),
            'decDeg': pd.concat([decdegDr5, decdegMC], ignore_index=True)
        }
    )
    # output = [eval(arg) for arg in args if arg in ['clustersDr5_MC', 'radegMC', 'decdegMC', 'radegDr5', 'decdegDr5']]

    return clustersDr5_MC

# Create negative class from dr5
def createNegativeClassDR5():
    # clustersDr5_MC = concat_tables(['clustersDr5_MC'])
    clustersDr5_MC = concat_tables()

    path = '/content/'

    imap_98 = enmap.read_fits(path + 'act_planck_dr5.01_s08s18_AA_f220_daynight_fullivar.fits')[0]

    positions = np.array(np.rad2deg(imap_98.posmap()))
    ras, decs = positions[1], positions[0]
    ras, decs = ras.ravel(), decs.ravel()
    rac = ras[np.random.choice(len(ras), size=100000, replace=False)] + np.random.normal(-0.1,0.1, 100000)
    dec = decs[np.random.choice(len(decs), size=100000, replace=False)] + np.random.normal(-0.1,0.1, 100000)

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
            if len(ra) == len(clustersDr5_MC): # instead of dr5?
                break

    n = len(ra)

    dfNegativeFromDr5 = pd.DataFrame({'Component_name': name, 'RA': ra, 'DEC': de})
    return dfNegativeFromDr5


def create_data_dr5():
    # Final data_dr5 with positive and negative classes
    clusters = read_dr5()
    # clusters = dr5
    clusters = clusters[['name', 'RADeg', 'decDeg']].reset_index(drop=True)
    clusters.rename(columns = {'name': 'Component_name', 'RADeg': 'RA', 'decDeg': 'DEC'}, inplace = True )
    clusters['target'] = 1
    random = createNegativeClassDR5()
    random['target'] = 0
    data_dr5 = pd.concat([clusters, random]).reset_index(drop=True)

    return data_dr5


def createNegativeClassMC(radegMC, decdegMC):
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
            if len(ra) == len(clustersDr5_MC): # instead of dr5.
                break

    n = len(ra)

    dfNegativeFromMacdows = pd.DataFrame({'Component_name': name, 'RA': ra, 'DEC': de})

    return dfNegativeFromMacdows

# Final data_macdows with positive and negative classes

def create_data_macdows():
    madCows_table = readMC()
    radegMC = madCows_table.iloc[:, 1].apply(lambda x: Angle(toHmsFormat(x)).degree)
    decdegMC = madCows_table.iloc[:, 2].apply(lambda x: Angle(toDmsFormat(x)).degree)
    clusters = pd.DataFrame({'Component_name': madCows_table['Name'], 'RA': radegMC, 'DEC': decdegMC})
    clusters['target'] = 1
    random = createNegativeClassMC(radegMC, decdegMC)
    random['target'] = 0
    data_macdows = pd.concat([clusters, random]).reset_index(drop=True)
    return data_macdows

# Test val and test data

def train_val_test_split():
    data_dr5 = create_data_dr5()
    data_macdows = create_data_macdows()

    work_path = '/content/'

    folderlocation = f'{work_path}data/Data224/'
    folders = ['train', 'val', 'test_dr5', 'test_macdows']


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

    test_macdows_0 = data_macdows[data_macdows.target == 0].reset_index(drop=True)
    test_macdows_1 = data_macdows[data_macdows.target == 1].reset_index(drop=True)

    list_train, list_val = [train_0, train_1], [val_0, val_1]
    list_test_dr5, list_test_MC = [test_dr5_0, test_dr5_1], [test_macdows_0, test_macdows_1]
    return list_train, list_val, list_test_dr5, list_test_MC


def ddos(dr5, madCows_table):
    list_train, list_val, list_test_dr5, list_test_MC = train_val_test_split()
    train_0, train_1 = list_train
    val_0, val_1 = list_val
    test_dr5_0, test_dr5_1 = list_test_dr5
    test_macdows_0, test_macdows_1 =list_test_MC

    work_path = '/content/'
    folders = [(train_0, 'train/0'), (train_1, 'train/1'), (val_0, 'val/0'), (val_1, 'val/1'),
               (test_dr5_0, 'test_dr5/0'), (test_dr5_1, 'test_dr5/1'), (test_macdows_0, 'test_macdows/0'),
               (test_macdows_1, 'test_macdows/1')]
    for subfolder, subfolder_name in folders:
        output_dir = f'{work_path}/data/Data224/{subfolder_name}'
        legacy_for_img.grab_cutouts(target_file=subfolder, output_dir=output_dir, survey='unwise-neo7', imgsize_pix=224,
                                    file_format='jpg')

    data_macdows = create_data_macdows(dr5, madCows_table)
    for type_ in [0, 1]:
        test_macdows_subset = data_macdows[data_macdows.target == type_].reset_index(drop=True)
        output_dir = f'{work_path}/data/Data224/test_macdows/{type_}'
        legacy_for_img.grab_cutouts(target_file=test_macdows_subset, output_dir=output_dir, survey='unwise-neo7',
                                    imgsize_pix=224, file_format='jpg')


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = numpy.array([0.507, 0.487, 0.441])
    std = numpy.array([0.267, 0.256, 0.276])
    inp = std * inp + mean
    inp = numpy.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.pause(0.001)  # pause a bit so that plots are updated

# Dataloader

def create_dataloader():
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
          'test_macdows': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),
    }

    folderlocation = '/content/data/Data224/'
    data_dir = folderlocation
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test_dr5', 'test_macdows']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                 shuffle=True, num_workers=3)
                  for x in ['train', 'val', 'test_dr5', 'test_macdows']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test_dr5', 'test_macdows']}
    #
    # class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = utils.make_grid(inputs)

    imshow(out)

    return device, dataloaders


