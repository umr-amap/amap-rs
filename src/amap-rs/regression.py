# general
import sys
import glob
import os
import warnings

# pytorch
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torchvision.transforms as T
import kornia.augmentation as K
import timm

# torchgeo
from torchgeo.transforms import AugmentationSequential

# geo
import geopandas as gpd

# data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import PredictionErrorDisplay
import umap.umap_ as umap
warnings.filterwarnings("ignore")

# custom modules
from datasets import ROIDataset
from utils.dl import check_false_dim
from utils.geo import get_geo_folds, get_mean_sd_by_band, get_random_points_on_raster
from utils.misc import get_colors_for_values
from custom_datasets import *


def train_loop(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
        verbose=False,
        ):

    model.train()  # Set the model to training mode
    running_loss = 0.0
    nsamples = 0

    for i, sample in enumerate(dataloader):

        if verbose:
            print(f'train\t{i/len(dataloader):.2%}', end='\r')
               
        images = check_false_dim(sample['image'])
        labels = torch.tensor(sample['gt'], dtype=torch.float32)
        images, labels = images.to(device), labels.to(device)  # Move data to the device

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # Zero the gradients

        running_loss += loss.item()
        nsamples += images.size(0)

    epoch_loss = running_loss / nsamples
    return epoch_loss


def test_loop(
        model, 
        dataloader, 
        criterion, 
        device,
        verbose=False,
        ):

    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    nsamples = 0
    predictions = []
    corrects = []

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if verbose:
                print(f'test\t{i/len(dataloader):.2%}', end='\r')

            images = check_false_dim(sample['image'])
            labels = torch.tensor(sample['gt'], dtype=torch.float32)
            corrects.extend(labels)
            images, labels = images.to(device), labels.to(device)  # Move data to the device

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions.extend(outputs.detach().cpu())

            # Statistics
            running_loss += loss.item()
            nsamples += images.size(0)
        
        epoch_loss = running_loss / nsamples
    
    return epoch_loss, predictions, corrects


def train_kfold(
        img_dir,
        gdf,
        arch,
        target_variable,
        loss_f=nn.MSELoss(),
        filename_glob = '*.[tT][iI][fF]',
        checkpoint_path = '',
        train_transform = None,
        test_transform = None,
        nfolds=None,
        fold_col = 'fold',
        batch_size = 10, 
        epochs = 100,
        lr = 1e-5,
        model_save_dir = "./",
        verbose=False,
        means=None,
        sds=None,
        device=None,
        ):

    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if nfolds:
        gdf = get_geo_folds(gdf, nfolds=nfolds)
        print(gdf.head())

    preds = []
    gts = []
    folds = []
    losses = {}

    if means==None :
        pathname = os.path.join(img_dir, "**", filename_glob)
        files = set(glob.iglob(pathname, recursive=True))
        # by default, only get the mean and sd of the first file matching filename_glob
        means, sds = get_mean_sd_by_band(next(iter(files)))

    model = timm.create_model(
            arch, 
            pretrained=True,
            in_chans=len(means), 
            num_classes=1 # because regression so only one value to predict
            )

    data_config = timm.data.resolve_model_data_config(model)
    _, h, w, = data_config['input_size']

    if train_transform==None:
        train_transform = AugmentationSequential(
                T.ConvertImageDtype(torch.float32), # change dtype for normalize to be possible
                K.Normalize(means,sds), # normalize occurs only on raster, not mask
                ## other data augmentation examples
                K.RandomVerticalFlip(p=0.5),
                K.RandomHorizontalFlip(p=0.5),
                K.Resize((h, w)),  # resize to 224*224 pixels, regardless of sampling size
                data_keys=["image"],
                )
    if test_transform==None:
        test_transform = AugmentationSequential(
                T.ConvertImageDtype(torch.float32), # change dtype for normalize to be possible
                K.Normalize(means,sds), # normalize occurs only on raster, not mask
                K.Resize((h, w)),  # resize to 224*224 pixels, regardless of sampling size
                data_keys=["image"],
                )


    filename_glob_tmp = filename_glob
    class CustomROIDataset(ROIDataset):
        filename_glob=filename_glob_tmp

    for fold in sorted(gdf[fold_col].unique()):


        if verbose:
            print(f'\n\n--------- fold {fold}------')

        best_loss = float('inf') ## start with infinite wrong loss
        train_indexes = gdf[gdf[fold_col]!= fold].index
        test_indexes = gdf[gdf[fold_col]== fold].index

        train_dataset = CustomROIDataset(
                naip_root, 
                gdf, 
                target_var=target_variable, 
                target_indexes=train_indexes,
                transforms = train_transform,
                )
        test_dataset = CustomROIDataset(
                naip_root, 
                gdf, 
                target_var=target_variable, 
                target_indexes=test_indexes,
                transforms = test_transform,
                )

        ## init via timm
        model = timm.create_model(
                arch, 
                pretrained=True,
                checkpoint_path=checkpoint_path,
                in_chans=len(means), 
                num_classes=1 # because regression so only one value to predict
                )
        model = model.to(device)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=stack_samples)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=stack_samples)
        optimizer = Adam(model.parameters(), lr=lr)

        train_losses=[]
        test_losses=[]
        losses[fold] = {}

        for epoch in range(epochs):
            if verbose:
                print(f'epoch {epoch}---------')
            train_loss = train_loop(
                    model, 
                    train_dataloader, 
                    loss_f, 
                    optimizer, 
                    device,
                    verbose=verbose,
                    )
            train_losses.append(train_loss)

            res = test_loop(
                    model, 
                    test_dataloader, 
                    loss_f, 
                    device,
                    verbose=verbose,
                    )

            test_loss, current_preds, current_gts = res
            test_losses.append(test_loss)

            if test_loss < best_loss:
                ## only keep track of best results
                best_gts = [x.item() for x in current_gts]
                best_preds = [x.item() for x in current_preds]
                model_save_path = os.path.join(model_save_dir,f'best_{fold}.pth')
                torch.save(model.state_dict(),model_save_path) # save model to disk

        losses[fold]['train'] = train_losses
        losses[fold]['test'] = test_losses

        preds.extend(best_preds)
        gts.extend(best_gts)
        ## keep a list with only fold number of the same size as the predictions
        folds.extend([fold for _ in best_preds]) 

    preds = np.asarray(preds)
    gts = np.asarray(gts)
    folds = np.asarray(folds)

    return preds, gts, folds, losses


if __name__ == "__main__":

    torch.manual_seed(42)

    ## Example workflow
    import os
    import tempfile
    from torchgeo.datasets import NAIP, ChesapeakeDE
    from torchgeo.datasets.utils import download_url
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    naip_root = os.path.join(tempfile.gettempdir(), "naip")
    chesapeake_root = os.path.join(tempfile.gettempdir(), "chesapeake")

    ############# data download and creation ############
    #naip_url = (
    #    "https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/"
    #)

    ## tile may not download properly, leading to a `TIFFReadEncodedTile() failed` error
    ## a simple wget with this url probably will solve the issue
    #tile = "m_3807511_ne_18_060_20181104.tif"
    #naip_path=os.path.join(naip_root, tile)
    #download_url(naip_url + tile, naip_root)
    #naip = NAIP(naip_root)

    ####ground truth of soil occupancy
    #os.makedirs(chesapeake_root, exist_ok=True)
    ## chesapeake = ChesapeakeDE(chesapeake_root, crs=naip.crs, res=naip.res, download=True)
    ## chesapeake_path = os.path.join(chesapeake_root,'DE_STATEWIDE.tif')
    
    ## gdf = get_random_points_on_raster_template(chesapeake_path,naip_path,1000)
    #gdf = get_random_points_on_raster(naip_path, 1000)
    #gdf.to_file(os.path.join(chesapeake_root,'random_chesapeake_reg.shp'), index=False)
    #print(gdf.head())
    ############################################ 

    MEANS = [122.39, 118.23, 98.1, 120.]
    SDS = [39.81, 37.33, 33.04, 30.]

    train_transform = AugmentationSequential(
            T.ConvertImageDtype(torch.float32), # change dtype for normalize to be possible
            K.Normalize(MEANS,SDS), # normalize occurs only on raster, not mask
            ## other data augmentation examples
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            K.Resize((224, 224)),  # resize to 224*224 pixels, regardless of sampling size
            data_keys=["image"],
            )
    test_transform = AugmentationSequential(
            T.ConvertImageDtype(torch.float32), # change dtype for normalize to be possible
            K.Normalize(MEANS,SDS), # normalize occurs only on raster, not mask
            K.Resize((224, 224)),  # resize to 224*224 pixels, regardless of sampling size
            data_keys=["image"],
            )

    gdf = gpd.read_file(os.path.join(chesapeake_root,'random_chesapeake_reg.shp'))

    ### if we need to normalize the target variable
    # gdf['pixel_valu'] = (gdf['pixel_valu'] - gdf['pixel_valu'].mean()) / gdf['pixel_valu'].std()

    gdf = gdf.head(200)

    res = train_kfold(
            img_dir=naip_root,
            gdf=gdf,
            arch='vit_base_patch16_224.dino',
            nfolds=5,
            target_variable='pixel_valu',
            batch_size=100,
            epochs=5,
            model_save_dir= chesapeake_root,
            verbose=True,
            means=MEANS,
            sds=SDS,
            lr=1e-4
            )
    preds, gts, folds, losses = res

    scatter_kwargs = {
        'color': get_colors_for_values(folds),  # Use categories for coloring
        'marker': '+',    # Marker style (you can change it as needed)
        'alpha': 0.5      # Transparency level (you can change it as needed)
    }

    display = PredictionErrorDisplay(y_true=gts, y_pred=preds)
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        gts,
        y_pred=preds,
        kind="actual_vs_predicted",
        ax=axs[0],
        random_state=0,
        scatter_kwargs=scatter_kwargs,
    )
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        gts,
        y_pred=preds,
        kind="residual_vs_predicted",
        ax=axs[1],
        random_state=0,
        scatter_kwargs=scatter_kwargs,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.suptitle("Plotting cross-validated predictions")
    plt.tight_layout()
    plt.savefig('/tmp/chesapeake/pred_obs.png')
    sys.exit(1)







    
    ## separate the dataset into 5 folds
    gdf = get_geo_folds(gdf, nfolds=5)
    print(gdf.head())

    all_preds = []
    all_gts = []
    all_folds = []
    best_loss = float('inf') ## start with infinite wrong loss

    for fold in gdf['fold'].unique():

        print(f'\n\n--------- fold n°{fold}------')

        train_indexes = gdf[gdf['fold']!= fold].index
        test_indexes = gdf[gdf['fold']== fold].index

        train_dataset = ROIDataset(
                naip_root, 
                gdf, 
                target_var='pixel_valu', 
                target_indexes=train_indexes,
                transforms = train_transform,
                )
        test_dataset = ROIDataset(
                naip_root, 
                gdf, 
                target_var='pixel_valu', 
                target_indexes=test_indexes,
                transforms = test_transform,
                )

        ## init via timm
        model = timm.create_model(
                'vit_base_patch16_224.dino', 
                pretrained=True,
                in_chans=len(MEANS), 
                num_classes=1 # because regression so only one value to predict
                )
        model = model.to(device)
        
        train_dataloader = DataLoader(train_dataset, batch_size=10, collate_fn=stack_samples)
        test_dataloader = DataLoader(test_dataset, batch_size=10, collate_fn=stack_samples)
        optimizer = Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        epochs = 2
        for epoch in range(epochs):
            print(f'epoch {epoch}---------')
            train_loss = train_loop(
                    model, 
                    train_dataloader, 
                    criterion, 
                    optimizer, 
                    device,
                    verbose=True,
                    )

            res = test_loop(
                    model, 
                    test_dataloader, 
                    criterion, 
                    device,
                    verbose=True,
                    )

            test_loss, predictions, corrects = res

            if test_loss < best_loss:
                ## only keep track of best results
                best_corrects = [x.item() for x in corrects]
                best_predictions = [x.item() for x in predictions]
                torch.save(model, '/tmp/chesapeake/best_model.pth') # save model to disk


        all_preds.extend(best_predictions)
        all_gts.extend(best_corrects)
        ## keep a list with only fold number of the same size as the predictions
        all_folds.extend([fold for _ in best_predictions]) 


    ######### plot pred vs obs

    all_preds = np.asarray(all_preds)
    all_gts = np.asarray(all_gts)
    all_folds = np.asarray(all_folds)

    scatter_kwargs = {
        'color': get_colors_for_values(all_folds),  # Use categories for coloring
        'marker': '+',    # Marker style (you can change it as needed)
        'alpha': 0.5      # Transparency level (you can change it as needed)
    }

    display = PredictionErrorDisplay(y_true=all_gts, y_pred=all_preds)
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        all_gts,
        y_pred=all_preds,
        kind="actual_vs_predicted",
        ax=axs[0],
        random_state=0,
        scatter_kwargs=scatter_kwargs,
    )
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        all_gts,
        y_pred=all_preds,
        kind="residual_vs_predicted",
        ax=axs[1],
        random_state=0,
        scatter_kwargs=scatter_kwargs,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.suptitle("Plotting cross-validated predictions")
    plt.tight_layout()
    plt.savefig('/tmp/chesapeake/pred_obs.png')

