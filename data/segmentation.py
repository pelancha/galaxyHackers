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

'''As pictures of stars from GAIA are not used in training, they should be obtained here'''

def prepare_gaia():
    data_gaia = data.read_gaia()
    path = samples_location + 'test_gaia'
    os.makedirs(path, exist_ok=True)

    legacy_for_img.grab_cutouts(target_file=data_gaia, output_dir=path,
                                          survey='unwise-neo7', imgsize_pix = 224*8, file_format='jpg' )
    return data_gaia


def predict_folder(folder, model, device='cuda:0'):
    model = model.to(device)
    loaded_model = torch.load(f"{working_path}state_dict/{model.__class__.__name__}_weights.pth", map_location=device)
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


def predict_tests(model):
    test_dr_0, test_dr_1 = data.train_val_test_split()[2]
    gaia = prepare_gaia()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clust = test_dr_1
    clust['prob'] = predict_folder(f'{dr5_sample_location}1', model, device=device)
    rand = test_dr_0
    rand['prob'] = predict_folder(f'{dr5_sample_location}0', model, device=device)

    gaia['prob'] =  predict_folder(f'{samples_location}test_gaia', model, device=device)
    samples = [clust, rand, gaia]
    return samples


def create_samples(model):
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

    samples = predict_tests(model)
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


def formSegmentationMaps(model):
    create_samples(model) #returns csvs
    cl5 = pd.read_csv(clusters_out)
    r5 =  pd.read_csv(randoms_out)
    gaia5 = pd.read_csv(stars_out)

    all_samples = [("Clusters", cl5), ("Random", r5), ("Stars", gaia5)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(5):
        createSegMap(0, all_samples[0][1].loc[i, 'RA'], all_samples[0][1].loc[i, 'DEC'], all_samples[0][1].loc[i, 'Component_name'], dire = f'{segmentation_maps_pics}Cl/{i}')
        createSegMap(0, all_samples[1][1].loc[i, 'RA'], all_samples[1][1].loc[i, 'DEC'], all_samples[1][1].loc[i, 'Component_name'], dire = f'{segmentation_maps_pics}R/{i}')
        createSegMap(0, all_samples[2][1].loc[i, 'RA'], all_samples[2][1].loc[i, 'DEC'], all_samples[2][1].loc[i, 'Component_name'], dire = f'{segmentation_maps_pics}GAIA/{i}')

    prob_clusters, prob_randoms, prob_gaia = [], [], []

    for i in range(5):
        prob_clusters.append(predict_folder(f'{segmentation_maps_pics}Cl/{i}', model, device=device))
        prob_randoms.append(predict_folder(f'{segmentation_maps_pics}R/{i}', model, device=device))
        prob_gaia.append(predict_folder(f'{segmentation_maps_pics}GAIA/{i}', model, device=device))

    predictions = [prob_clusters, prob_randoms, prob_gaia]
    return all_samples, predictions


def printSegMaps(selected_models):
    for model_name, model in selected_models:
        all_samples, predictions = formSegmentationMaps(model)

        fig = plt.figure(constrained_layout=True)
        subfigs = fig.subfigures(nrows=len(all_samples), ncols=1)
        for i in range(len(subfigs)):
            subfigs[i].suptitle(all_samples[i][0])
            axs = subfigs[i].subplots(nrows=1, ncols=len(all_samples[i][1]))
            for j in range(len(axs)):
                axs[j].plot()
                axs[j].set_title("{:.4f}".format(all_samples[i][1].loc[j, "prob"]))
                axs[j].imshow(predictions[i][j].reshape(20,20), cmap = cm.PuBu)
                axs[j].axis('off')
                axs[j].plot(10, 10, 'x', ms=7, color='red')
        plt.show()


def printBigSegMap(selected_models):
    test_dr5, test_madcows = data.train_val_test_split()[2:4]
    test_dr5_0, test_dr5_1 = test_dr5
    test_madcows_0, test_madcows_1 = test_madcows
    df = pd.concat([test_dr5_0, test_dr5_1, test_madcows_0, test_madcows_1], ignore_index=True)

    cl0 = df[df.target==1].sample(1, random_state=1).reset_index(drop=True)
    max_ra = cl0['RA'].max()
    max_de = cl0['DEC'].max()
    required_space = 15 / 60 #15 minutes including shift
    while (max_ra + required_space) > 360 or (max_de - required_space) < -90:
        cl0 = df[df.target==1].sample(1, random_state=1).reset_index(drop=True)
        max_ra = cl0['RA'].max()
        max_de = cl0['DEC'].max()

    bigSegMapLocation = f'{segmentation_maps_pics}Big'
    os.makedirs(bigSegMapLocation, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model_name, model in selected_models:
        createSegMap(1, cl0.loc[0, 'RA'], cl0.loc[0, 'DEC'], cl0.loc[0, 'Component_name'], dire=bigSegMapLocation)
        prob_big = predict_folder(bigSegMapLocation, model, device=device)
        plt.imshow(prob_big.reshape(30, 30), cmap=cm.PuBu)
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
