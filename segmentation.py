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

from enum import Enum

from data import DataPart, ClusterDataset
from train import Predictor

def load_model(model: torch.nn.Module, optimizer_name, device):

    model = model.to(device)
    weights_name = f'best_weights_{model.__class__.__name__}_{optimizer_name}.pth'
    weights_path = Path(settings.BEST_MODELS_PATH, weights_name )
    loaded_model = torch.load(weights_path, map_location=device)

    model.load_state_dict(loaded_model)

def predict_test(model: torch.nn.Module, optimizer_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    load_model(model, optimizer_name, device)

    predictor = Predictor(model=model, device=device)


    dataloaders = data.train_val_test_split()
    for part in [DataPart.TEST_DR5, DataPart.TEST_DR5, DataPart.GAIA]:
       
        predictions = predictor.predict(dataloader=dataloaders[part])

        predictions.to_csv(Path(settings.PREDICTIONS_PATH, f"{part}.csv"))


class SampleName(str, Enum):

    CLUSTER = "cluster"
    RANDOM = "random"
    GAIA = "gaia"


sample_sizes: dict = {
    SampleName.CLUSTER: 10,
    SampleName.RANDOM: 5,
    SampleName.GAIA: 5,
}

# Data part and target class for each sample
sample_soures = {
    SampleName.CLUSTER: (DataPart.TEST_DR5, 1) ,
    SampleName.RANDOM: (DataPart.TEST_DR5, 0),
    SampleName.GAIA: (DataPart.GAIA, 0),
}


class MapType(str, Enum):
    SMALL = 0
    BIG = 1

def create_sample(sample_name):

    sample_size = sample_sizes[sample_name]
    source, target_class = sample_soures[sample_name]

    description = pd.read_csv(Path(settings.DESCRIPTION_PATH, f"{source.value}.csv"), index_col=0)

    description = description.loc[description["target"] == target_class]


    min_ra, min_dec = -float("inf"), -float("inf")
    max_ra, max_dec = float("inf"), float("inf")
    

    required_space = 10 / 120 #shift for small segmentation maps / 2
    while ((max_ra + required_space) > 360 or
            (max_dec + required_space) > 90 or
            (min_dec - required_space) < -90 or
            (min_ra - required_space) < 0):
        
        sample = description.sample(sample_size, random_state=settings.SEED)
        max_ra = sample['ra_deg'].max()
        max_dec = sample['dec_deg'].max()
        min_ra = sample['ra_deg'].min()
        min_dec = sample['dec_deg'].min()

    path = Path(settings.SEGMENTATION_SAMPLES_DESCRIPTION_PATH, f"{sample_name.value}.csv", index=True)
    sample.to_csv(path)

    return sample


def create_map_dataloader(
        map_type: MapType, 
        ra_start: float, 
        dec_start: float,
        map_dir: Path): #id: 0 for small segmentation maps, 1 - for a big one

    name = []
    ras, decs = [], []

    match map_type:
        case MapType.SMALL:
            step = 0.5 / 60 #шаг в 0.5 минуту, выражено в градусах
            #10 минут - максимальное расстояние подряд в одну сторону, 0.5 минута - один шаг, всё *10
            distance = 10
            cycle_step = 5
        case MapType.BIG:
            step = 1 / 60 #шаг в 1 минуту, выражено в градусах
            #30 минут - максимальное расстояние подряд в одну сторону, 1 минута - один шаг
            distance = 30
            cycle_step = 1

    shift = distance / 2 #подаётся центр карты сегментации, переводим начало в левый верхний угол

    ra_corner = ra_start - shift
    dec_corner = dec_start + shift

    #масштаб в case 0
    if map_type == MapType.SMALL:
        distance *= 10


    dec_current = dec_corner

    idxs = []
    cur_idx = 0
    #ra шагаем вправо, dec шагаем вниз
    for _ in range(0, distance, cycle_step):
        ra_current = ra_corner

        for _ in range(0, distance, cycle_step):
            coords = coord.SkyCoord(ra=ra_current*u.degree, dec=dec_current*u.degree, frame='icrs')

            ras.append(coords.ra.degree)
            decs.append(coords.dec.degree)
            idxs.append(cur_idx)

            cur_idx += 1


            b = coords.galactic.b.degree
            l = coords.galactic.l.degree

            name.append(f'Map {l:.3f}{b:+.3f}')

            ra_current += step

            if (0 > ra_current or ra_current > 360):
              break

        dec_current -= step

        if (-90 > dec_current or dec_current > 90):
          break

    description_path = Path(map_dir, f"description.csv")

    map_data = pd.DataFrame({'name': name, 'ra_deg': ras, 'dec_deg': decs}, index=pd.Index(idxs, name="idx"))
    map_data.to_csv(description_path)

    legacy_for_img.grab_cutouts(
        target_file=map_data, 
        name_col="name",
        ra_col="ra_deg",
        dec_col="dec_deg",
        output_dir=map_dir,         
        survey='unwise-neo7', 
        imgsize_pix = 224, 
        file_format='jpg' )
    

    dataset = ClusterDataset(
        map_dir,
        description_path,
        transform=transforms.Compose(data.main_transforms),
    )

    dataloader = DataLoader(dataset, batch_size=settings.BATCH_SIZE)

    return dataloader
       

def prepare_sample_dataloaders(data: pd.DataFrame, sample_name: SampleName, map_type: MapType):

    dataloaders = []
    
    for idx, row in data.iterrows():

        directory = Path(settings.SEGMENTATION_SAMPLES_PATH, sample_name.value, str(idx))
        os.makedirs(directory, exist_ok=True)

        dataloader = create_map_dataloader(
            map_type=map_type, 
            ra_start=row['ra_deg'], 
            dec_start=row['dec_deg'], 
            map_dir=directory
        )

        dataloaders.append((idx, dataloader))


    return dataloaders 
    

def create_segmentation_plot(
        map_type: MapType, 
        model_name: str, 
        optimizer_name: str,
        predictor: Predictor, 
        sample_name: SampleName,
        n_cols = 5
        ):
    
    sample = create_sample(sample_name)

    dataloaders = prepare_sample_dataloaders(data=sample, sample_name=sample_name, map_type=map_type)

    n_rows = (len(sample) + 1) // n_cols

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5))

    
    for i, (idx, dataloader) in enumerate(dataloaders):

        cur_col = i % n_cols

        if n_rows > 1:
            cur_row = i // n_cols
            cur_position = (cur_row, cur_col)
        else:
            cur_position = cur_col

        predictions = predictor.predict(dataloader)

        path = Path(settings.SEGMENTATION_SAMPLES_PATH, sample_name.value, str(idx), "predictions.csv")

        predictions.to_csv(path)

        axes[cur_position].plot()
        # subtitle = "prob: " + "{:.4f}".format(start_probability)
        # axs[idx].set_title(subtitle)

        im = axes[cur_position].imshow(predictions["y_prob"].values.reshape(20,20).astype(float),
                        cmap = cm.PuBu,
                        vmin = 0,
                        vmax = 1)
        axes[cur_position].axis('off')
        axes[cur_position].plot(10, 10, 'x', ms=7, color='red')

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Cluster probability", orientation="horizontal", aspect=40)
    # plt.suptitle(all_samples[i][0], size='xx-large')

    plt.savefig(Path(settings.SEGMENTATION_MAPS_PATH, f"{map_type}_{model_name}_{optimizer_name}_{sample_name.value}.png"))
    plt.close()
        

def create_segmentation_plots(model, model_name, optimizer_name, map_type: MapType=MapType.SMALL):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    load_model(model, optimizer_name, device)

    predictor = Predictor(model, device=device)

    for sample_name in list(SampleName):
        create_segmentation_plot(
            map_type=map_type,
            model_name=model_name,
            optimizer_name=optimizer_name,
            predictor=predictor,
            sample_name=sample_name,

            
            )


# def bigSegMap_cluster(model, optimizer_name, device):
#     test_dr5, test_madcows = data.train_val_test_split()[2:4]
#     test_dr5_0, test_dr5_1 = test_dr5
#     test_madcows_0, test_madcows_1 = test_madcows

#     if (not os.path.exists(location) or
#         len(os.listdir(f'{madcows_sample_location}1')) == 0):
#         os.makedirs(data_out, exist_ok=True)
#         try:
#             wget.download(url=data_wget, out=data_out)
#             with ZipFile(f"{zipped_data_out}", 'r') as zObject:
#                 zObject.extractall(path=f"{data_out}")
#             os.remove(f"{zipped_data_out}")
#         except:
#             data.ddos()
#         else:
#             if (not os.path.exists(location) or
#                     len(os.listdir(location)) == 0):
#                 data.ddos()
#     test_dr5_1['prob'] = predict_folder(f'{dr5_sample_location}1', model, optimizer_name, device=device)
#     test_madcows_1['prob'] = predict_folder(f'{madcows_sample_location}1', model, optimizer_name, device=device)
#     df = pd.concat([test_dr5_1, test_madcows_1], ignore_index=True)

#     cl0 = df.sample(1, random_state=1).reset_index(drop=True)
#     max_ra = cl0['RA'].max()
#     max_de = cl0['DEC'].max()
#     min_ra = cl0['RA'].min()
#     min_de = cl0['DEC'].min()
#     required_space = 30 / 120 #shift for big segmentation map / 2
#     while ((max_ra + required_space) > 360 or
#             (max_de + required_space) > 90 or
#             (min_de - required_space) < -90 or
#             (min_ra - required_space) < 0):
#         cl0 = df.sample(1).reset_index(drop=True)
#         max_ra = cl0['RA'].max()
#         max_de = cl0['DEC'].max()
#         min_ra = cl0['RA'].min()
#         min_de = cl0['DEC'].min()
#     os.makedirs(bigSegMapLocation, exist_ok=True)
#     cl0.to_csv(cl_bigSegMap_out, index=False)


# def create_sample_big(model, optimizer_name, dire, device='cuda:0'):
#     if not os.path.exists(cl_bigSegMap_out):
#         bigSegMap_cluster(model, optimizer_name, device)
#     os.makedirs(dire, exist_ok=True)
#     if len(os.listdir(dire)) == 0:
#         cl0 = pd.read_csv(cl_bigSegMap_out)
#         createSegMap(1, cl0.loc[0, 'RA'], cl0.loc[0, 'DEC'], cl0.loc[0, 'Component_name'], dire=dire)


# def saveBigSegMap(selected_models, optimizer_name):
#     '''
#     Creates a segmentation map in a 30x30 box with 1 minute step for a cluster randomly chosen from MaDCoWS or ACT_dr5 datasets
#     and saves it in segmentation_maps folder
#     '''
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     for model_name, model in selected_models:
#         create_sample_big(model, optimizer_name, dire=bigSegMap_pics, device=device)
#         cl0 = pd.read_csv(cl_bigSegMap_out)
#         prob_big = predict_folder(bigSegMap_pics, model, optimizer_name, device=device)
#         fig = plt.imshow(prob_big.reshape(30, 30), 
#                    cmap=cm.PuBu)
#         plt.title("{:.4f}".format(cl0.loc[0, "prob"]))
#         plt.axis('off')
#         plt.colorbar(fig, label="Cluster probability", orientation="horizontal", aspect=40)
#         os.makedirs(seg_maps, exist_ok=True)
#         plt.savefig(f"{working_path}{seg_maps}{model_name}_{optimizer_name}_Big.png")
#         plt.close()