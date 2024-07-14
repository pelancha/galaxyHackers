'''Script to create segmentation maps for models'''
'''
To create segmentation maps for randomly chosen clusters, random objects and stars use function saveSegMaps()
To create a segmentation map with larger scale for a randomly chosen cluster use function saveBigSegMap()
'''

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
from config import settings
from zipfile import ZipFile 
import wget
from pathlib import Path

from train import Predictor

def predict_loader(loader: DataLoader, model, optimizer_name, device):
    
    model = model.to(device)
    weights_name = f'best_weights_{model.__class__.__name__}_{optimizer_name}.pth'
    weights_path = Path(settings.BEST_MODELS_PATH, weights_name )
    loaded_model = torch.load(weights_path, map_location=device)

    model.load_state_dict(loaded_model)

    predictor = Predictor(model=model, device=device)

    _, probs = predictor.predict(dataloader=loader)

    return np.array(probs)


def predict_tests(model, optimizer_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not (os.path.exists(clusters_out) and
            os.path.exists(randoms_out)):
        test_dr_0, test_dr_1 = data.train_val_test_split()[2]
        
        if not os.path.exists(clusters_out):
            clust = test_dr_1
            clust['prob'] = predict_folder(f'{dr5_sample_location}1', model, optimizer_name, device=device)
        else:
            clust['prob'] = np.zeros(len(clust["RA"])) #only needed to create non-existing samples
        if not os.path.exists(randoms_out):
            rand = test_dr_0
            rand['prob'] = predict_folder(f'{dr5_sample_location}0', model, optimizer_name, device=device)
        else:
            rand['prob'] = np.zeros(len(rand["RA"])) #only needed to create non-existing samples
    
    if not os.path.exists(randoms_out):
        gaia = prepare_gaia()
        gaia['prob'] =  predict_folder(f'{samples_location}test_gaia', model, optimizer_name, device=device)
    else:
        gaia['prob'] = np.zeros(len(gaia["RA"])) #only needed to create non-existing samples
    samples = [clust, rand, gaia]

    clust = test_dr_1
    clust['prob'] = predict_folder(f'{dr5_sample_location}1', model, optimizer_name, device=device)
    return samples

class SamplesPart(str, Enum):

    CLUSTER = "cluster"
    RANDOM = "random"
    GAIA = "gaia"

def create_samples(model, optimizer_name):
    
    for part in list(SamplesPart):
        path = Path(settings.SEGMENTATION_SAMPLES_PATH, part.value)
        os.makedirs(path, exist_ok=True)


    # for iter1 in range(10):    # 100 = number of classes
    #     path = segmentation_maps_pics + 'Cl/'+str(iter1)
    #     os.makedirs(path, exist_ok=True)
    # for iter1 in range(5):    # 100 = number of classes
    #     path = segmentation_maps_pics + 'R/'+str(iter1)
    #     os.makedirs(path, exist_ok=True)
    #     path = segmentation_maps_pics + 'GAIA/'+str(iter1)
    #     os.makedirs(path, exist_ok=True)

    # if not (os.path.exists(clusters_out) and
    #         os.path.exists(randoms_out) and
    #         os.path.exists(stars_out)):
    samples = predict_tests(model, optimizer_name)
    samples_final = []
    count = 0
    for test in samples:
        if count == 0:
            sample_size = 10
            count += 1
        else: 
            sample_size = 5
        sample_for_map = test.sample(sample_size).reset_index(drop=True)
        max_ra = sample_for_map['RA'].max()
        max_de = sample_for_map['DEC'].max()
        min_ra = sample_for_map['RA'].min()
        min_de =  sample_for_map['DEC'].min()
        required_space = 10 / 120 #shift for small segmentation maps / 2
        while ((max_ra + required_space) > 360 or
                (max_de + required_space) > 90 or
                (min_de - required_space) < -90 or
                (min_ra - required_space) < 0):
            sample_for_map = test.sample(sample_size).reset_index(drop=True)
            max_ra = sample_for_map['RA'].max()
            max_de = sample_for_map['DEC'].max()
            min_ra = sample_for_map['RA'].min()
            min_de = sample_for_map['DEC'].min()
        samples_final.append(sample_for_map)


    samples_final[0].to_csv(clusters_out, index=False)
    samples_final[1].to_csv(randoms_out, index=False)
    samples_final[2].to_csv(stars_out, index=False)

def createSegMap(id, ra0, dec0, name, dire): #id: 0 for small segmentation maps, 1 - for a big one
    name = []
    ras, decs = [], []
    match id:
        case 0:
            step = 0.5 / 60 #шаг в 0.5 минуту, выражено в градусах
            #10 минут - максимальное расстояние подряд в одну сторону, 0.5 минута - один шаг, всё *10
            distance = 10
            cycle_step = 5
        case 1:
            step = 1 / 60 #шаг в 1 минуту, выражено в градусах
            #30 минут - максимальное расстояние подряд в одну сторону, 1 минута - один шаг
            distance = 30
            cycle_step = 1

    shift = distance / 2 #подаётся центр карты сегментации, переводим начало в левый верхний угол
    ra1, dec_current = ra0 - shift, dec0 + shift
    #масштаб в case 0
    if step == 0.5 / 60:
        distance *= 10
    #ra шагаем вправо, dec шагаем вниз
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
            if (0 > ra_current or ra_current > 360):
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
    if (not os.path.exists(segmentation_maps_pics) or 
        len(os.listdir(f'{segmentation_maps_pics}GAIA/4')) == 0):
        for i in range(10):
            createSegMap(0, all_samples[0][1].loc[i, 'RA'], all_samples[0][1].loc[i, 'DEC'], all_samples[0][1].loc[i, 'Component_name'], dire = f'{segmentation_maps_pics}Cl/{i}')
        for i in range(5):
            createSegMap(0, all_samples[1][1].loc[i, 'RA'], all_samples[1][1].loc[i, 'DEC'], all_samples[1][1].loc[i, 'Component_name'], dire = f'{segmentation_maps_pics}R/{i}')
            createSegMap(0, all_samples[2][1].loc[i, 'RA'], all_samples[2][1].loc[i, 'DEC'], all_samples[2][1].loc[i, 'Component_name'], dire = f'{segmentation_maps_pics}GAIA/{i}')
    return all_samples


def formSegmentationMaps(model, optimizer_name):
    if not (os.path.exists(clusters_out) and 
            os.path.exists(randoms_out) and 
            os.path.exists(stars_out)):
        create_samples(model, optimizer_name) #returns csvs
    all_samples = prepare_samples()
    
    prob_clusters, prob_randoms, prob_gaia = [], [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(10):
        cl_elem_prob = predict_folder(f'{segmentation_maps_pics}Cl/{i}', model, optimizer_name, device=device)
        if i == 0:
            cl_prob_range = [cl_elem_prob.min(), cl_elem_prob.max()]
        else:
            cl_elem_prob_min, cl_elem_prob_max = cl_elem_prob.min(), cl_elem_prob.max()
            if cl_elem_prob_min < cl_prob_range[0]:
                cl_prob_range[0] = cl_elem_prob_min
            if cl_elem_prob_max > cl_prob_range[1]:
                cl_prob_range[1] = cl_elem_prob_max
        prob_clusters.append(cl_elem_prob)

    for i in range(5):
        rand_elem_prob = predict_folder(f'{segmentation_maps_pics}R/{i}', model, optimizer_name, device=device)
        gaia_elem_prob = predict_folder(f'{segmentation_maps_pics}GAIA/{i}', model, optimizer_name, device=device)

        if i == 0:
            rand_prob_range = [rand_elem_prob.min(), rand_elem_prob.max()]
            gaia_prob_range = [gaia_elem_prob.min(), gaia_elem_prob.max()]
        else:
            rand_elem_prob_min, rand_elem_prob_max = rand_elem_prob.min(), rand_elem_prob.max()
            gaia_elem_prob_min, gaia_elem_prob_max = gaia_elem_prob.min(), gaia_elem_prob.max()
            if rand_elem_prob_min < rand_prob_range[0]:
                rand_prob_range[0] = rand_elem_prob_min
            if rand_elem_prob_max > rand_prob_range[1]:
                rand_prob_range[1] = rand_elem_prob_max
            if gaia_elem_prob_min < gaia_prob_range[0]:
                gaia_prob_range[0] = gaia_elem_prob_min
            if gaia_elem_prob_max > gaia_prob_range[1]:
                gaia_prob_range[1] = gaia_elem_prob_max
        prob_randoms.append(rand_elem_prob)
        prob_gaia.append(gaia_elem_prob)

    ranges = [cl_prob_range, rand_prob_range, gaia_prob_range]
    predictions = [prob_clusters, prob_randoms, prob_gaia]
    return all_samples, predictions, ranges


def saveSegMaps(selected_models, optimizer_name):
    '''
    Creates segmentation maps in 10x10 boxes with 0.5 minute step for 5 randomly chosen clusters from ACT_dr5 dataset,
    objects from its negative class, stars from GAIA catalogue and saves these three samples separately in segmentation_maps folder
    '''
    for model_name, model in selected_models:
        all_samples, predictions, ranges = formSegmentationMaps(model, optimizer_name)
        for i in range(len(all_samples)):
            if i != 0:
                fig, axs = plt.subplots(nrows=1, ncols=len(all_samples[i][1]), figsize=(15, 5))
                for j in range(len(axs)):
                    axs[j].plot()
                    subtitle = "prob: " + "{:.4f}".format(all_samples[i][1].loc[j, "prob"])
                    axs[j].set_title(subtitle)
                    axs[j].imshow(predictions[i][j].reshape(20,20),
                                  cmap = cm.PuBu,
                                  vmin = ranges[i][0],
                                  vmax = ranges[i][1])
                    axs[j].axis('off')
                    axs[j].plot(10, 10, 'x', ms=7, color='red')
                pos = axs[0].imshow(predictions[i][0].reshape(20,20),
                                    cmap = cm.PuBu,
                                    vmin = ranges[i][0],
                                    vmax = ranges[i][1])
                fig.colorbar(pos, ax=axs, label="Cluster probability", orientation="horizontal", aspect=40)
                plt.suptitle(all_samples[i][0], size='xx-large')
                os.makedirs(seg_maps, exist_ok=True)
                plt.savefig(f"{working_path}{seg_maps}{model_name}_{optimizer_name}_{all_samples[i][0]}.png")
                plt.close()
            if i == 0:
                n_columns = len(all_samples[i][1]) // 2
                fig, axs = plt.subplots(nrows=2, ncols=n_columns, figsize=(15, 5))
                for k in range(2):
                    for j in range(n_columns):
                        axs[k, j].plot()
                        subtitle = ("prob: " + "{:.4f}".format(all_samples[i][1].loc[j + n_columns * k, "prob"]) +
                                "\nredshift: " + "{:.4f}".format(all_samples[i][1].loc[j + n_columns * k, "redshift"]) +
                                "\nweight: " + "{:.4f}".format(all_samples[i][1].loc[j + n_columns * k, "M500c"]))
                        axs[k, j].set_title(subtitle)
                        axs[k, j].imshow(predictions[i][j + n_columns * k].reshape(20,20),
                                      cmap = cm.PuBu,
                                      vmin = ranges[i][0],
                                      vmax = ranges[i][1])
                        axs[k, j].axis('off')
                        axs[k, j].plot(10, 10, 'x', ms=7, color='red')
                pos = axs[1, 0].imshow(predictions[i][0].reshape(20,20),
                                    cmap = cm.PuBu,
                                    vmin = ranges[i][0],
                                    vmax = ranges[i][1])
                fig.colorbar(pos, ax=axs, label="Cluster probability", orientation="horizontal", aspect=40)
                plt.suptitle(all_samples[i][0], size='xx-large')
                fig.set_size_inches(18, 11, forward=True)
                os.makedirs(seg_maps, exist_ok=True)
                plt.savefig(f"{working_path}{seg_maps}{model_name}_{optimizer_name}_{all_samples[i][0]}.png")
                plt.close()
 

def bigSegMap_cluster(model, optimizer_name, device):
    test_dr5, test_madcows = data.train_val_test_split()[2:4]
    test_dr5_0, test_dr5_1 = test_dr5
    test_madcows_0, test_madcows_1 = test_madcows

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
    test_dr5_1['prob'] = predict_folder(f'{dr5_sample_location}1', model, optimizer_name, device=device)
    test_madcows_1['prob'] = predict_folder(f'{madcows_sample_location}1', model, optimizer_name, device=device)
    df = pd.concat([test_dr5_1, test_madcows_1], ignore_index=True)

    cl0 = df.sample(1, random_state=1).reset_index(drop=True)
    max_ra = cl0['RA'].max()
    max_de = cl0['DEC'].max()
    min_ra = cl0['RA'].min()
    min_de = cl0['DEC'].min()
    required_space = 30 / 120 #shift for big segmentation map / 2
    while ((max_ra + required_space) > 360 or
            (max_de + required_space) > 90 or
            (min_de - required_space) < -90 or
            (min_ra - required_space) < 0):
        cl0 = df.sample(1).reset_index(drop=True)
        max_ra = cl0['RA'].max()
        max_de = cl0['DEC'].max()
        min_ra = cl0['RA'].min()
        min_de = cl0['DEC'].min()
    os.makedirs(bigSegMapLocation, exist_ok=True)
    cl0.to_csv(cl_bigSegMap_out, index=False)


def create_sample_big(model, optimizer_name, dire, device='cuda:0'):
    if not os.path.exists(cl_bigSegMap_out):
        bigSegMap_cluster(model, optimizer_name, device)
    os.makedirs(dire, exist_ok=True)
    if len(os.listdir(dire)) == 0:
        cl0 = pd.read_csv(cl_bigSegMap_out)
        createSegMap(1, cl0.loc[0, 'RA'], cl0.loc[0, 'DEC'], cl0.loc[0, 'Component_name'], dire=dire)


def saveBigSegMap(selected_models, optimizer_name):
    '''
    Creates a segmentation map in a 30x30 box with 1 minute step for a cluster randomly chosen from MaDCoWS or ACT_dr5 datasets
    and saves it in segmentation_maps folder
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model_name, model in selected_models:
        create_sample_big(model, optimizer_name, dire=bigSegMap_pics, device=device)
        cl0 = pd.read_csv(cl_bigSegMap_out)
        prob_big = predict_folder(bigSegMap_pics, model, optimizer_name, device=device)
        fig = plt.imshow(prob_big.reshape(30, 30), 
                   cmap=cm.PuBu)
        plt.title("{:.4f}".format(cl0.loc[0, "prob"]))
        plt.axis('off')
        plt.colorbar(fig, label="Cluster probability", orientation="horizontal", aspect=40)
        os.makedirs(seg_maps, exist_ok=True)
        plt.savefig(f"{working_path}{seg_maps}{model_name}_{optimizer_name}_Big.png")
        plt.close()