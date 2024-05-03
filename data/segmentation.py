import legacy_for_img
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import timm
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import astropy.coordinates as coord
from astropy import units as u
import os
import matplotlib as mpl
from matplotlib import cm
import data


class ImageSet(Dataset):
    def __init__(self, dir, transform=None):
        self.data_dir = dir
        self.images = os.listdir(dir)
        self.images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        self.transform = transform

    # Defining the length of the dataset
    def __len__(self):
        return len(self.images)

    # Defining the method to get an item from the dataset
    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.images[index])
        image = Image.open(image_path)

        # Applying the transform
        if self.transform:
            image = self.transform(image)

        return image


def predict_folder(folder, device='cuda:0'): #+ input - model (str) for
    model_ft = timm.create_model('resnet18', pretrained=True, num_classes=1) #make it possible to change model
    model = model_ft.to(device)
    loaded_model = torch.load('./ResNet_epoch_20.pth', map_location=device) #change name according to interesting epoch
    model.load_state_dict(loaded_model["model_state_dict"])
    model.eval()

    trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    data = ImageSet(folder, transform=trans)
    data_load = DataLoader(data, batch_size = 5, shuffle=False)
    probs = np.array([])
    for i, data in enumerate(data_load):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        img = data.to(device)

        #model prediction
        outputs = model(img)
        probs = np.append(probs, outputs.data.cpu().detach().numpy())
    # return np.array(probs)
    return np.array(probs)

'''As pictures of stars from GAIA are not used in training, they should be obtained here'''

def prepare_gaia():
    data_gaia = data.read_gaia()
    path = samples_location + 'test_gaia'
    os.makedirs(path, exist_ok=True)

    legacy_for_img.grab_cutouts(target_file=data_gaia, output_dir=path,
                                          survey='unwise-neo7', imgsize_pix = 224*8, file_format='jpg' )
    return data_gaia


def create_samples(): #+ input - model (str) for predict_folder
    os.makedirs(segmentation_maps_pics, exist_ok=True)

    path = segmentation_maps_pics + 'Cl/'
    os.makedirs(path, exist_ok=True)
    path = segmentation_maps_pics + 'R/'
    os.makedirs(path, exist_ok=True)
    path = segmentation_maps_pics + 'GAIA/'
    os.makedirs(path, exist_ok=True)

    for iter1 in range(5):    # 100 = number of classes
        path = segmentation_maps_pics + 'Cl/'+str(iter1)
        os.makedirs(path, exist_ok=True)
        path = segmentation_maps_pics + 'R/'+str(iter1)
        os.makedirs(path, exist_ok=True)
        path = segmentation_maps_pics + 'GAIA/'+str(iter1)
        os.makedirs(path, exist_ok=True)

    test_dr_0, test_dr_1 = data.train_val_test_split()[2]
    gaia = prepare_gaia()
    samples = [test_dr_1, test_dr_0, gaia]
    id = 0
    for test in samples:
        sample5 = test.sample(5, random_state=5).reset_index(drop=True)
        max_ra = sample5['RA'].max()
        max_de = sample5['DEC'].max()
        required_space = 15 / 60 #15 minutes including shift
        while (max_ra + required_space) > 360 or (max_de - required_space) < -90:
            sample5 = test.sample(5).reset_index(drop=True)
            max_ra = sample5['RA'].max()
            max_de = sample5['DEC'].max()
        match id:
            case 0:
                sample5.to_csv(clusters_out, index=False)
            case 1:
                sample5.to_csv(randoms_out, index=False)
            case 2:
                sample5.to_csv(stars_out, index=False)
        id += 1


def createSegMap(id, ra0, dec0, name, dire): #id: 0 for small segmentation maps, 1 - for a big one
    shift = 5 / 60 #отступ на 5 минут в градусах
    name = []
    ras, decs = [], []
    ra1, dec_current = ra0 + shift, dec0 - shift #ra шагаем вправо, dec шагаем вниз
    match id:
        case 0:
            step = 0.5 / 60 #шаг в 0.5 минуту
            #10 минут - максимальное расстояние подряд в одну сторону, 0.5 минута - один шаг, всё *10
            distance = 100
            cycle_step = 5
        case 1:
            step = 1 / 60 #шаг в 1 минуту
            #30 минут - максимальное расстояние подряд в одну сторону, 1 минута - один шаг
            distance = 30
            cycle_step = 1

    for i in range(0, distance, cycle_step):
        ra_current = ra1
        for j in range(0, distance, cycle_step):
            coords = coord.SkyCoord(ra=ra_current*u.degree, dec=dec_current*u.degree, frame='icrs')
            ras.append(coords.ra.degree)
            decs.append(coords.dec.degree)
            b = coords.galactic.b.degree
            l = coords.galactic.l.degree
            name.append(f'Map {l:.3f}{b:+.3f}')
            ra_current += step
            if (0 > ra_current  or ra_current > 360):
              break
        dec_current -= step
        if (-90 > dec_current or dec_current > 90):
          break

    data = pd.DataFrame({'Component_name': name, 'RA': ras, 'DEC': decs})
    legacy_for_img.grab_cutouts(target_file=data, output_dir=dire,
                                  survey='unwise-neo7', imgsize_pix = 224*8, file_format='jpg' )
    # print(data)
    # return data.shape


def formSegmentationMaps():
    create_samples() #returns csvs
    cl5 = pd.read_csv(clusters_out)
    r5 =  pd.read_csv(randoms_out)
    gaia5 = pd.read_csv(stars_out)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(5):
        createSegMap(0, cl5.loc[i, 'RA'], cl5.loc[i, 'DEC'], cl5.loc[i, 'Component_name'], dire = f'{segmentation_maps_pics}Cl/{i}')
        createSegMap(0, r5.loc[i, 'RA'], r5.loc[i, 'DEC'], r5.loc[i, 'Component_name'], dire = f'{segmentation_maps_pics}R/{i}')
        createSegMap(0, gaia5.loc[i, 'RA'], gaia5.loc[i, 'DEC'], gaia5.loc[i, 'Component_name'], dire = f'{segmentation_maps_pics}GAIA/{i}')

    prob_clusters, prob_randoms, prob_gaia = [], [], []

    for i in range(5):
        prob_clusters.append(predict_folder(f'{segmentation_maps_pics}Cl/{i}', device=device))
        prob_randoms.append(predict_folder(f'{segmentation_maps_pics}R/{i}', device=device))
        prob_gaia.append(predict_folder(f'{segmentation_maps_pics}GAIA/{i}', device=device))

    return prob_clusters, prob_randoms, prob_gaia


def printSegMaps():
    prob_clusters, prob_randoms, prob_gaia = formSegmentationMaps()

    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(10, 6))

    for i in range(len(prob_clusters)):
        ax[0, i].imshow(prob_clusters[i].reshape(20,20), cmap = cm.Blues)

    ax[0, 2].set_title('Clusters')
    ax[1, 2].set_title('Random')
    ax[2, 2].set_title('Stars')

    for i in range(len(prob_randoms)):
        ax[1, i].imshow(prob_randoms[i].reshape(20,20), cmap = cm.Blues)

    for i in range(len(prob_gaia)):
        ax[2, i].imshow(prob_gaia[i].reshape(20,20), cmap = cm.Blues)

    for j in range(3):
        for i in range(5):
            ax[j, i].axis('off')
            ax[j, i].plot(10, 10, 'x', ms=7, color='red')

    plt.show()


def printBigSegMap():
    test_dr5, test_madcows = data.train_val_test_split()[2:4]
    test_dr5_0, test_dr5_1 = test_dr5
    test_madcows_0, test_madcows_1 = test_madcows
    df = pd.concat([test_dr5_0, test_dr5_1, test_madcows_0, test_madcows_1], ignore_index=True)
    cl0 = df[df.target==1].sample(1, random_state=1).reset_index(drop=True)

    bigSegMapLocation = f'{segmentation_maps_pics}Big'
    os.makedirs(bigSegMapLocation, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    createSegMap(1, cl0.loc[0, 'RA'], cl0.loc[0, 'DEC'], cl0.loc[0, 'Component_name'], dire=bigSegMapLocation)
    prob_big = predict_folder(bigSegMapLocation, device=device)
    plt.imshow(prob_big.reshape(30, 30), cmap=cm.Blues)
    plt.axis('off')

# config
working_path = "./"
location = "data/"
segmentation_maps_pics = working_path + location + "example/"
samples_location = working_path + location + "Data224/"
dr5_sample_location = samples_location + "test_dr5/"

clusters_out = segmentation_maps_pics + "cl5.csv"
randoms_out = segmentation_maps_pics + "r5.csv"
stars_out = segmentation_maps_pics + "gaia5.csv"
