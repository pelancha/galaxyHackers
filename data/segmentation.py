'''Script to create segmentation maps for models'''
'''
To create segmentation maps for randomly chosen clusters, random objects and stars use function saveSegMaps()
To create a segmentation map with larger scale for a randomly chosen cluster use function saveBigSegMap()
'''

import data.legacy_for_img as legacy_for_img
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
import data.data as data
from data.config import *
from zipfile import ZipFile 
import wget


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


def prepare_gaia():
    data_gaia = data.read_gaia()
# gaia might have already been collected with the rest of data during dataloaders creation
    if (not os.path.exists(gaia_sample_location) or 
        len(os.listdir(gaia_sample_location)) == 0):
      os.makedirs(gaia_sample_location, exist_ok=True)
      legacy_for_img.grab_cutouts(target_file=data_gaia, output_dir=gaia_sample_location,
                                          survey='unwise-neo7', imgsize_pix = 224*8, file_format='jpg' )
    return data_gaia


def predict_folder(folder, model, optimizer_name, device='cuda:0'):
    model = model.to(device)
    loaded_model = torch.load(f"{working_path}state_dict/best_{model.__class__.__name__}_{optimizer_name}_weights.pth", map_location=device)
    model.load_state_dict(loaded_model)
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
    return np.array(probs)


def predict_tests(model, optimizer_name):
    test_dr_0, test_dr_1 = data.train_val_test_split()[2]
    if (not os.path.exists(location) or 
        len(os.listdir(f'{madcows_sample_location}1')) == 0):
        os.makedirs(data_out, exist_ok=True)
        try:
            wget.download(url=data_wget, out=data_out)
            with ZipFile(f"{zipped_data_out}", 'r') as zObject: 
                zObject.extractall(path=f"{data_out}")
            os.remove(f"{zipped_data_out}")
        except:
            data.ddos()
        else:
            if (not os.path.exists(location) or 
                    len(os.listdir(location)) == 0):
                data.ddos()

    gaia = prepare_gaia()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clust = test_dr_1
    clust['prob'] = predict_folder(f'{dr5_sample_location}1', model, optimizer_name, device=device)
    rand = test_dr_0
    rand['prob'] = predict_folder(f'{dr5_sample_location}0', model, optimizer_name, device=device)

    gaia['prob'] =  predict_folder(f'{samples_location}test_gaia', model, optimizer_name, device=device)
    samples = [clust, rand, gaia]
    return samples


def create_samples(model, optimizer_name):
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

    samples = predict_tests(model, optimizer_name)
    samples5_final = []
    for test in samples:
        sample5 = test.sample(5, random_state=5).reset_index(drop=True)
        max_ra = sample5['RA'].max()
        max_de = sample5['DEC'].max()
        required_space = 15 / 60 #15 minutes including shift
        while (max_ra + required_space) > 360 or (max_de - required_space) < -90:
            sample5 = test.sample(5).reset_index(drop=True)
            max_ra = sample5['RA'].max()
            max_de = sample5['DEC'].max()
        samples5_final.append(sample5)

    samples5_final[0].to_csv(clusters_out, index=False)
    samples5_final[1].to_csv(randoms_out, index=False)
    samples5_final[2].to_csv(stars_out, index=False)


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


def prepare_samples():
    '''Function to create segmentation maps for chosen samples'''
    cl5 = pd.read_csv(clusters_out)
    r5 =  pd.read_csv(randoms_out)
    gaia5 = pd.read_csv(stars_out)
    all_samples = [("Clusters", cl5), ("Random", r5), ("Stars", gaia5)]
    for i in range(5):
        createSegMap(0, all_samples[0][1].loc[i, 'RA'], all_samples[0][1].loc[i, 'DEC'], all_samples[0][1].loc[i, 'Component_name'], dire = f'{segmentation_maps_pics}Cl/{i}')
        createSegMap(0, all_samples[1][1].loc[i, 'RA'], all_samples[1][1].loc[i, 'DEC'], all_samples[1][1].loc[i, 'Component_name'], dire = f'{segmentation_maps_pics}R/{i}')
        createSegMap(0, all_samples[2][1].loc[i, 'RA'], all_samples[2][1].loc[i, 'DEC'], all_samples[2][1].loc[i, 'Component_name'], dire = f'{segmentation_maps_pics}GAIA/{i}')
    return all_samples


def formSegmentationMaps(model, optimizer_name):
    if (not os.path.exists(segmentation_maps_pics) or 
        len(os.listdir(f'{segmentation_maps_pics}GAIA/4')) == 0):
        os.makedirs(data_out, exist_ok=True)
        try:
            wget.download(url=example_wget, out=data_out)
            with ZipFile(f"{zipped_example_out}", 'r') as zObject: 
                zObject.extractall(path=f"{data_out}")
            os.remove(f"{zipped_example_out}")
            cl5 = pd.read_csv(clusters_out)
            r5 =  pd.read_csv(randoms_out)
            gaia5 = pd.read_csv(stars_out)
            all_samples = [("Clusters", cl5), ("Random", r5), ("Stars", gaia5)]
        except:
            create_samples(model, optimizer_name) #returns csvs
            all_samples = prepare_samples()
        else:
            if (not os.path.exists(segmentation_maps_pics) or 
                len(os.listdir(segmentation_maps_pics)) == 0):
                create_samples(model, optimizer_name) #returns csvs
                all_samples = prepare_samples()
    else:
            cl5 = pd.read_csv(clusters_out)
            r5 =  pd.read_csv(randoms_out)
            gaia5 = pd.read_csv(stars_out)
            all_samples = [("Clusters", cl5), ("Random", r5), ("Stars", gaia5)]

    prob_clusters, prob_randoms, prob_gaia = [], [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(5):
        prob_clusters.append(predict_folder(f'{segmentation_maps_pics}Cl/{i}', model, optimizer_name, device=device))
        prob_randoms.append(predict_folder(f'{segmentation_maps_pics}R/{i}', model, optimizer_name, device=device))
        prob_gaia.append(predict_folder(f'{segmentation_maps_pics}GAIA/{i}', model, optimizer_name, device=device))

    predictions = [prob_clusters, prob_randoms, prob_gaia]
    return all_samples, predictions


def saveSegMaps(selected_models, optimizer_name):
    '''
    Creates segmentation maps in 10x10 boxes with 0.5 minute step for 5 randomly chosen clusters from ACT_dr5 dataset, 
    objects from its negative class, stars from GAIA catalogue and saves these three samples separately in segmentation_maps folder 
    '''
    for model_name, model in selected_models:
        all_samples, predictions = formSegmentationMaps(model, optimizer_name)
        for i in range(len(all_samples)):
            fig, axs = plt.subplots(nrows=1, ncols=len(all_samples[i][1]), figsize=(15, 5))
            for j in range(len(axs)):
                axs[j].plot()
                axs[j].set_title("{:.4f}".format(all_samples[i][1].loc[j, "prob"]))
                axs[j].imshow(predictions[i][j].reshape(20,20), cmap = cm.PuBu)
                axs[j].axis('off')
                axs[j].plot(10, 10, 'x', ms=7, color='red')
            plt.suptitle(all_samples[i][0], size='xx-large')
            os.makedirs(seg_maps, exist_ok=True)
            plt.savefig(f"{working_path}{seg_maps}{model_name}_{optimizer_name}_{all_samples[i][0]}.png")
            plt.close()
 

def create_sample_big(test_dr5_1, test_madcows_1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if (not os.path.exists(location) or 
        len(os.listdir(f'{madcows_sample_location}1')) == 0):
        os.makedirs(data_out, exist_ok=True)
        try:
            wget.download(url=data_wget, out=data_out)
            with ZipFile(f"{zipped_data_out}", 'r') as zObject: 
                zObject.extractall(path=f"{data_out}")
            os.remove(f"{zipped_data_out}")
        except:
            data.ddos()
        else:
            if (not os.path.exists(location) or 
                    len(os.listdir(location)) == 0):
                data.ddos()
    test_dr5_1['prob'] = predict_folder(f'{dr5_sample_location}1', model, device=device)
    test_madcows_1['prob'] = predict_folder(f'{madcows_sample_location}1', model, device=device)
    df = pd.concat([test_dr5_1, test_madcows_1], ignore_index=True)

    cl0 = df.sample(1, random_state=1).reset_index(drop=True)
    max_ra = cl0['RA'].max()
    max_de = cl0['DEC'].max()
    required_space = 15 / 60 #15 minutes including shift
    while (max_ra + required_space) > 360 or (max_de - required_space) < -90:
        cl0 = df.sample(1).reset_index(drop=True)
        max_ra = cl0['RA'].max()
        max_de = cl0['DEC'].max()
    os.makedirs(bigSegMapLocation, exist_ok=True)
    cl0.to_csv(cl_bigSegMap_out, index=False)

    os.makedirs(bigSegMap_pics, exist_ok=True)
    createSegMap(1, cl0.loc[0, 'RA'], cl0.loc[0, 'DEC'], cl0.loc[0, 'Component_name'], dire=bigSegMap_pics)


def saveBigSegMap(selected_models, optimizer_name):
    '''
    Creates a segmentation map in a 30x30 box with 1 minute step for a cluster randomly chosen from MaDCoWS or ACT_dr5 datasets 
    and saves it in segmentation_maps folder
    '''
    if (not os.path.exists(bigSegMapLocation) or 
        len(os.listdir(bigSegMap_pics)) == 0):
        os.makedirs(data_out, exist_ok=True)
        try:
            wget.download(url=bigSegMap_wget, out=data_out)
            with ZipFile(f"{zipped_bigSegMap_out}", 'r') as zObject: 
                zObject.extractall(path=f"{data_out}")
            os.remove(f"{zipped_bigSegMap_out}")
            cl0 = pd.read_csv(cl_bigSegMap_out)
        except:
            test_dr5, test_madcows = data.train_val_test_split()[2:4]
            test_dr5_0, test_dr5_1 = test_dr5
            test_madcows_0, test_madcows_1 = test_madcows
            create_sample_big(test_dr5_1, test_madcows_1)
            cl0 = pd.read_csv(cl_bigSegMap_out)
        else:
            if (not os.path.exists(bigSegMapLocation) or 
                len(os.listdir(bigSegMap_pics)) == 0):
                test_dr5, test_madcows = data.train_val_test_split()[2:4]
                test_dr5_0, test_dr5_1 = test_dr5
                test_madcows_0, test_madcows_1 = test_madcows
                create_sample_big(test_dr5_1, test_madcows_1)
                cl0 = pd.read_csv(cl_bigSegMap_out)
    else:
        cl0 = pd.read_csv(cl_bigSegMap_out)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model_name, model in selected_models:
        prob_big = predict_folder(bigSegMap_pics, model, optimizer_name, device=device)
        plt.imshow(prob_big.reshape(30, 30), cmap=cm.PuBu)
        plt.title("{:.4f}".format(cl0.loc[0, "prob"]))
        plt.axis('off')
        os.makedirs(seg_maps, exist_ok=True)
        plt.savefig(f"{working_path}{seg_maps}{model_name}_{optimizer_name}_Big.png")
        plt.close()
