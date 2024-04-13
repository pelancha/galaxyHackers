# NOT YET TESTED
# NOT YET TESTED
# NOT YET TESTED
# NOT YET TESTED
# NOT YET TESTED
# NOT YET TESTED

import legacy_for_img

import os
import sys
from urllib.request import urlretrieve

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

download_path = "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/"
act_planck_map = "act_planck_dr5.01_s08s18_AA_f090_daynight_map.fits"
dr5_cluster_catalog = "DR5_cluster-catalog_v1.1.fits"

urlretrieve(download_path + "maps/" + act_planck_map, act_planck_map)
urlretrieve(download_path + dr5_cluster_catalog, dr5_cluster_catalog)

work_path = "/content/"
dr5 = atpy.Table().read(f'{work_path}DR5_cluster-catalog_v1.1.fits').to_pandas().reset_index(drop=True)
dr5['name'] = [str(dr5.loc[i, 'name'], encoding='utf-8') for i in range(len(dr5))]

radegDr5 = dr5.loc[:, "RADeg"]
decdegDr5 = dr5.loc[:, "decDeg"]

def toHmsFormat(s):
    hours, minutes, seconds = s.split()
    return f"{hours}h{minutes}m{seconds}s"

def toDmsFormat(s):
    degrees, minutes, seconds = s.split()
    return f"{degrees}d{minutes}m{seconds}s"

radegMC = madCows_table.iloc[:, 1].apply(lambda x: Angle(toHmsFormat(x)).degree)
decdegMC = madCows_table.iloc[:, 2].apply(lambda x: Angle(toDmsFormat(x)).degree)

clustersDr5_MC = pd.DataFrame(
    {
        'name': pd.concat([dr5['name'], madCows_table['Name']], ignore_index=True),
        'RADeg': pd.concat([radegDr5, radegMC], ignore_index=True),
        'decDeg': pd.concat([decdegDr5, decdegMC], ignore_index=True)
    }
)

#positions = np.array(np.rad2deg(imap_98.posmap()))
ras = radegDr5
decs = decdegDr5
#ras, decs = positions[1], positions[0]
ras, decs = ras.ravel(), decs.ravel()
# rac = ras[np.random.choice(len(ras), size=100000, replace=False)] + np.random.normal(-0.1,0.1, 100000)
# dec = decs[np.random.choice(len(decs), size=100000, replace=False)] + np.random.normal(-0.1,0.1, 100000)

rac = ras[np.random.choice(len(ras), size=100000)] + np.random.normal(-0.1,0.1, 100000)
dec = decs[np.random.choice(len(decs), size=100000)] + np.random.normal(-0.1,0.1, 100000)


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


# Final data_dr5 with positive and negative classes

clusters = dr5[['name', 'RADeg', 'decDeg']].reset_index(drop=True)
clusters.rename(columns = {'name': 'Component_name', 'RADeg': 'RA', 'decDeg': 'DEC'}, inplace = True )
clusters['target'] = 1
random = dfNegativeFromDr5
random['target'] = 0
data_dr5 = pd.concat([clusters, random]).reset_index(drop=True)

# Create negative class from Macdows

def createNegativeClassRac(x):
    return np.clip(x + np.random.normal(-15, 15), 0, 360)

def createNegativeClassDec(x):
    return np.clip(x + np.random.normal(-15, 15), -90, 90)

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

clusters = pd.DataFrame({'Component_name': madCows_table['Name'], 'RA': radegMC, 'DEC': decdegMC})
clusters['target'] = 1
random = dfNegativeFromMacdows
random['target'] = 0
data_macdows = pd.concat([clusters, random]).reset_index(drop=True)

# Test val and test data

folderlocation = f'{work_path}data/Data224/'
folders = ['train', 'val', 'test', 'test_macdows']

for folder in folders:
    path = os.path.join(folderlocation, folder)
    os.makedirs(path, exist_ok=True)

    for iter1 in range(2):
        subpath = os.path.join(path, str(iter1))
        os.makedirs(subpath, exist_ok=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_dr5.index, data_dr5['target'], test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

train = data_dr5.iloc[X_train].reset_index(drop=True)
val = data_dr5.iloc[X_val].reset_index(drop=True)
test = data_dr5.iloc[X_test].reset_index(drop=True)

train_0 = train[train.target == 0].reset_index(drop=True)
train_1 = train[train.target == 1].reset_index(drop=True)

val_0 = val[val.target == 0].reset_index(drop=True)
val_1 = val[val.target == 1].reset_index(drop=True)

test_0 = test[test.target == 0].reset_index(drop=True)
test_1 = test[test.target == 1].reset_index(drop=True)

folders = [(train_0, 'train/0'), (train_1, 'train/1'),  (val_0, 'val/0'), (val_1, 'val/1'), (test_0, 'test/0'), (test_1, 'test/1')]
for subfolder, subfolder_name in folders:
    output_dir = f'{work_path}/data/Data224/{subfolder_name}'
    legacy_for_img.grab_cutouts(target_file=subfolder, output_dir=output_dir, survey='unwise-neo7', imgsize_pix=224, file_format='jpg')

for type_ in [0, 1]:
    test_macdows_subset = data_macdows[data_macdows.target == type_].reset_index(drop=True)
    output_dir = f'{work_path}/data/Data224/test_macdows/{type_}'
    legacy_for_img.grab_cutouts(target_file=test_macdows_subset, output_dir=output_dir, survey='unwise-neo7', imgsize_pix=224, file_format='jpg')


# Dataloader

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
     'test': transforms.Compose([
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

image_datasets = {x: datasets.ImageFolder(os.path.join(folderlocation, x), data_transforms[x]) for x in folders}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=3)
              for x in ['train', 'val', 'test', 'test_macdows']}
dataset_sizes = {x: len(image_datasets[x]) for x in folders}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = utils.make_grid(inputs)


# Pixel Prediction
from PIL import Image
from pixell import enmap

class ImageSet(Dataset):
    def __init__(self, dir, transform=None):
        self.data_dir = dir
        self.images = os.listdir(dir)
        self.images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.images[index])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image

def predict_folder(folder, device='cuda:0'):
    model = model_ft.to(device)
    model.load_state_dict(torch.load('/content/ResNet_epoch_5.pth', map_location=device), )
    model.eval()

    trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    data = ImageSet(folder, transform=trans)
    data_load = DataLoader(data, batch_size = 16, shuffle=False)
    probs = np.array([])
    for i, data in enumerate(data_load):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        img = data.to(device)

        outputs = model(img)
        probs = np.append(probs, outputs.data.cpu().detach().numpy())
    return np.array(probs)


def create_directories(folderlocation, num_classes):
    os.makedirs(folderlocation, exist_ok=True)

    for index in ['Cl', 'R']:
        os.makedirs(os.path.join(folderlocation, index), exist_ok=True)
        for i in range(num_classes):
            os.makedirs(os.path.join(folderlocation, index, str(i)), exist_ok=True)

def save_sample_indices(df, target, num_samples, filename):
    sample_df = df[df.target == target].sample(num_samples, random_state=5).reset_index(drop=True)
    sample_df.to_csv(filename, index=False)

def generate_random_coordinates(num_samples):
    ra_min, ra_max, dec_min, dec_max = 0, 360, -90, 90
    return np.random.uniform(ra_min, ra_max, num_samples), np.random.uniform(dec_min, dec_max, num_samples)

def export_csv(num_samples, file):
    ra_coords, dec_coords = generate_random_coordinates(num_samples)
    test_data = pd.DataFrame({'RA': ra_coords, 'DEC': dec_coords})
    test_data.to_csv(file, index=False)

def get_img(ra, de, name, dire):
    width = 1
    box = np.deg2rad([[de-width/2.,ra-width/2.],[de+width/2.,ra+width/2.]])
    pixbox = np.array([[0, 0,], [20, 20]])
    imap_98 = enmap.read_map('./act_planck_dr5.01_s08s18_AA_f090_daynight_map.fits', box=box, pixbox=pixbox)[0]
    decs, ras = np.array(np.rad2deg(imap_98.posmap()))
    decs, ras = decs.ravel(), ras.ravel()
    ras=coord.SkyCoord(ra=ras*u.degree, dec=decs*u.degree, frame='icrs').ra.degree

    data = pd.DataFrame({'Component_name': name, 'RA': ras, 'DEC': decs})
    legacy_for_img.grab_cutouts(target_file=data, output_dir=dire,
                                          survey='unwise-neo7', imgsize_pix = 224*8, file_format='jpg' )


num_classes, num_samples = 5, 5
num_coordinates = 100

filename = "test.csv"
export_csv(num_coordinates, filename)
random_coordinates = pd.read_csv(filename)

folderlocation = './data/example/'

create_directories(folderlocation, num_classes)

clust = df[df.target==1].reset_index(drop=True)
clust['prob'] = predict_folder('./data/Data224/test/1')
rand = df[df.target==0].reset_index(drop=True)
rand['prob'] = predict_folder('./data/Data224/test/0')

r5 = rand[rand.target==0].sample(5, random_state=5).reset_index(drop=True)
r5.to_csv('./data/example/r5.csv',index=False)

cl5 = clust[clust.target==1].sample(5, random_state=5).reset_index(drop=True)
cl5.to_csv('./data/example/cl5.csv',index=False)

cl5 = pd.read_csv('./data/example/cl5.csv')
r5 =  pd.read_csv('./data/example/r5.csv')

for i in range(num_samples):
    get_img(cl5.loc[i, 'RA'], cl5.loc[i, 'DEC'], cl5.loc[i, 'Component_name'], dire='../../data/example/Cl/' + str(i))
    get_img(r5.loc[i, 'RA'], r5.loc[i, 'DEC'], r5.loc[i, 'Component_name'], dire='../../data/example/R/' + str(i))

prob_clust, prob_rand = [], []

for i in range(num_samples):
    prob_clust.append(predict_folder('../../data/example/Cl/' + str(i)))
    prob_rand.append(predict_folder('../../data/example/R/' + str(i)))

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))

for i in range(num_samples):
    ax[0, i].imshow(prob_clust[i].reshape(20, 20), cmap=cm.Blues)
    ax[1, i].imshow(prob_rand[i].reshape(20, 20), cmap=cm.Blues)


prob_big = predict_folder('./data/example/Big')
plt.figure(figsize=(6, 6))
plt.imshow(prob_big.reshape(128, 128), cmap=cm.Blues)
plt.axis('off')

plt.show()
