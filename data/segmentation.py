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
    folderlocation = './data/Data224/'
    path = folderlocation + 'test_gaia'
    os.makedirs(path, exist_ok=True)

    legacy_for_img.grab_cutouts(target_file=data_gaia, output_dir=path,
                                          survey='unwise-neo7', imgsize_pix = 224*8, file_format='jpg' )
    return data_gaia


def predict_tests(): #+ input - model (str) for predict_folder
    test_dr_0, test_dr_1 = data.train_val_test_split()[2]
    gaia = prepare_gaia()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clust = test_dr_1
    clust['prob'] = predict_folder('./data/Data224/test_dr5/1', device=device)
    rand = test_dr_0
    rand['prob'] = predict_folder('./data/Data224/test_dr5/0', device=device)

    gaia['prob'] =  predict_folder('./data/Data224/test_gaia', device=device)
    return clust, rand, gaia


def create_samples(): #+ input - model (str) for predict_folder
    folderlocation = './data/example/'

    path = folderlocation
    os.makedirs(path, exist_ok=True)

    path = folderlocation + 'Cl/'
    os.makedirs(path, exist_ok=True)

    path = folderlocation + 'R/'
    os.makedirs(path, exist_ok=True)

    path = folderlocation + 'GAIA/'
    os.makedirs(path, exist_ok=True)

    for iter1 in range(5):    # 100 = number of classes
        path = folderlocation + 'Cl/'+str(iter1)
        os.makedirs(path, exist_ok=True)

        path = folderlocation + 'R/'+str(iter1)
        os.makedirs(path, exist_ok=True)

        path = folderlocation + 'GAIA/'+str(iter1)
        os.makedirs(path, exist_ok=True)

    clust, rand, gaia = predict_tests()

    r5 = rand.sample(5, random_state=5).reset_index(drop=True)
    max_ra = r5['RA'].max()
    max_de = r5['DEC'].max()
    required_space = 15 / 60 #15 minutes including shift
    # print(max_ra, max_de)
    while (max_ra + required_space) > 360 or (max_de - required_space) < -90:
        r5 = rand[rand.target==0].sample(5).reset_index(drop=True)
        max_ra = r5['RA'].max()
        max_de = r5['DEC'].max()
    r5.to_csv('./data/example/r5.csv',index=False)

    cl5 = clust.sample(5, random_state=5).reset_index(drop=True)
    max_ra = cl5['RA'].max()
    max_de = cl5['DEC'].max()
    required_space = 15 / 60 #15 minutes including shift
    while (max_ra + required_space) > 360 or (max_de - required_space) < -90:
        cl5 = clust[clust.target==1].sample(5).reset_index(drop=True)
        max_ra = cl5['RA'].max()
        max_de = cl5['DEC'].max()
    cl5.to_csv('./data/example/cl5.csv',index=False)

    gaia5 = gaia.sample(5, random_state=5).reset_index(drop=True)
    max_ra = gaia5['RA'].max()
    max_de = gaia5['DEC'].max()
    required_space = 15 / 60 #15 minutes including shift
    # print(max_ra, max_de)

    while (max_ra + required_space) > 360 or (max_de - required_space) < -90:
        gaia5 = gaia.sample(5).reset_index(drop=True)
        max_ra = gaia5['RA'].max()
        max_de = gaia5['DEC'].max()
    gaia5.to_csv('./data/example/gaia5.csv',index=False)


def createSegMap(id, ra0, dec0, name, dire): #id: 0 for small segmentation maps, 1 - for a big one
    match id:
        case 0:
            shift = 5 / 60 #отступ на 5 минут в градусах
            step = 0.5 / 60 #шаг в 0.5 минуту
            name = []
            ras, decs = [], []
            ra1, dec_current = ra0 + shift, dec0 - shift #ra шагаем вправо, dec шагаем вниз
            switch = 0

            for i in range(0, 100, 5): #10 минут - максимальное расстояние подряд в одну сторону, 0.5 минута - один шаг, всё *10
                ra_current = ra1
                for j in range(0, 100, 5):
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
                                                  survey='unwise-neo7', imgsize_pix = 20*20, file_format='jpg' )
            # print(data)
            # return data.shape
        case 1:
            shift = 5 / 60 #отступ на 5 минут в градусах
            step = 1 / 60 #шаг в 1 минуту
            name = []
            ras, decs = [], []
            ra1, dec_current = ra0 + shift, dec0 - shift #ra шагаем вправо, dec шагаем вниз
            switch = 0

            for i in range(0, 30): #30 минут - максимальное расстояние подряд в одну сторону, 1 минута - один шаг
                ra_current = ra1
                for j in range(0, 30):
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
    cl5 = pd.read_csv('./data/example/cl5.csv')
    r5 =  pd.read_csv('./data/example/r5.csv')
    gaia5 = pd.read_csv('./data/example/gaia5.csv')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    createSegMap(0, cl5.loc[0, 'RA'], cl5.loc[0, 'DEC'], cl5.loc[0, 'Component_name'], dire = './data/example/Cl/0')
    createSegMap(0, cl5.loc[1, 'RA'], cl5.loc[1, 'DEC'], cl5.loc[1, 'Component_name'], dire = './data/example/Cl/1')
    createSegMap(0, cl5.loc[2, 'RA'], cl5.loc[2, 'DEC'], cl5.loc[2, 'Component_name'], dire = './data/example/Cl/2')
    createSegMap(0, cl5.loc[3, 'RA'], cl5.loc[3, 'DEC'], cl5.loc[3, 'Component_name'], dire = './data/example/Cl/3')
    createSegMap(0, cl5.loc[4, 'RA'], cl5.loc[4, 'DEC'], cl5.loc[4, 'Component_name'], dire = './data/example/Cl/4')

    createSegMap(0, r5.loc[0, 'RA'], r5.loc[0, 'DEC'], r5.loc[0, 'Component_name'], dire = './data/example/R/0')
    createSegMap(0, r5.loc[1, 'RA'], r5.loc[1, 'DEC'], r5.loc[1, 'Component_name'], dire = './data/example/R/1')
    createSegMap(0, r5.loc[2, 'RA'], r5.loc[2, 'DEC'], r5.loc[2, 'Component_name'], dire = './data/example/R/2')
    createSegMap(0, r5.loc[3, 'RA'], r5.loc[3, 'DEC'], r5.loc[3, 'Component_name'], dire = './data/example/R/3')
    createSegMap(0, r5.loc[4, 'RA'], r5.loc[4, 'DEC'], r5.loc[4, 'Component_name'], dire = './data/example/R/4')

    createSegMap(0, gaia5.loc[0, 'RA'], gaia5.loc[0, 'DEC'], gaia5.loc[0, 'Component_name'], dire = './data/example/GAIA/0')
    createSegMap(0, gaia5.loc[1, 'RA'], gaia5.loc[1, 'DEC'], gaia5.loc[1, 'Component_name'], dire = './data/example/GAIA/1')
    createSegMap(0, gaia5.loc[2, 'RA'], gaia5.loc[2, 'DEC'], gaia5.loc[2, 'Component_name'], dire = './data/example/GAIA/2')
    createSegMap(0, gaia5.loc[3, 'RA'], gaia5.loc[3, 'DEC'], gaia5.loc[3, 'Component_name'], dire = './data/example/GAIA/3')
    createSegMap(0, gaia5.loc[4, 'RA'], gaia5.loc[4, 'DEC'], gaia5.loc[4, 'Component_name'], dire = './data/example/GAIA/4')

    prob_clust0 = predict_folder('./data/example/Cl/0', device=device)
    prob_clust1 = predict_folder('./data/example/Cl/1', device=device)
    prob_clust2 = predict_folder('./data/example/Cl/2', device=device)
    prob_clust3 = predict_folder('./data/example/Cl/3', device=device)
    prob_clust4 = predict_folder('./data/example/Cl/4', device=device)

    prob_clusters = [prob_clust0, prob_clust1, prob_clust2, prob_clust3, prob_clust4]

    prob_r0 = predict_folder('./data/example/R/0', device=device)
    prob_r1 = predict_folder('./data/example/R/1', device=device)
    prob_r2 = predict_folder('./data/example/R/2', device=device)
    prob_r3 = predict_folder('./data/example/R/3', device=device)
    prob_r4 = predict_folder('./data/example/R/4', device=device)

    prob_randoms = [prob_r0, prob_r1, prob_r2, prob_r3, prob_r4]

    prob_gaia0 = predict_folder('./data/example/GAIA/0', device=device)
    prob_gaia1 = predict_folder('./data/example/GAIA/1', device=device)
    prob_gaia2 = predict_folder('./data/example/GAIA/2', device=device)
    prob_gaia3 = predict_folder('./data/example/GAIA/3', device=device)
    prob_gaia4 = predict_folder('./data/example/GAIA/4', device=device)

    prob_gaia = [prob_gaia0, prob_gaia1, prob_gaia2, prob_gaia3, prob_gaia4]

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
    test_dr5, test_macdows = data.train_val_test_split()[2:4]
    test_dr5_0, test_dr5_1 = test_dr5
    test_macdows_0, test_macdows_1 = test_macdows
    df = pd.concat([test_dr5_0, test_dr5_1, test_macdows_0, test_macdows_1], ignore_index=True)
    cl0 = df[df.target==1].sample(1, random_state=1).reset_index(drop=True)

    folderlocation = './data/example/Big'

    path = folderlocation
    os.makedirs(path, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    createSegMap(1, cl0.loc[0, 'RA'], cl0.loc[0, 'DEC'], cl0.loc[0, 'Component_name'], dire='./data/example/Big')
    prob_big = predict_folder('./data/example/Big', device=device)
    plt.imshow(prob_big.reshape(30, 30), cmap=cm.Blues)
    plt.axis('off')