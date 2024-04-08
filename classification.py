# general
import sys
import os
import glob
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
from torchgeo.datasets import stack_samples
from torchgeo.transforms import AugmentationSequential

# geo
import geopandas as gpd


# data
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import umap.umap_ as umap
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# custom modules
from datasets import ROIDataset
from utils.dl import check_false_dim, convert_classes_to_idx
from utils.geo import get_random_points_on_raster_template, get_geo_folds, get_mean_sd_by_band


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
        labels = torch.tensor(sample['gt'], dtype=torch.int64)
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
    correct_preds = 0
    nsamples = 0
    predictions = []
    corrects = []

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if verbose:
                print(f'test\t{i/len(dataloader):.2%}', end='\r')

            images = check_false_dim(sample['image'])
            labels = torch.tensor(sample['gt'], dtype=torch.int64)
            corrects.extend(labels)
            images, labels = images.to(device), labels.to(device)  # Move data to the device

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions.extend(outputs.detach().cpu().numpy().argmax(1))

            # Statistics
            running_loss += loss.item()
            nsamples += images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / nsamples
        epoch_accuracy = correct_preds.double() / nsamples

        if verbose:
            print(classification_report(corrects, predictions, labels=np.unique(corrects)))
            print(confusion_matrix(corrects, predictions, labels=np.unique(corrects)))
    
    return epoch_loss, predictions, corrects



def train_kfold(
        img_dir,
        gdf,
        arch,
        target_variable,
        loss_f=nn.CrossEntropyLoss(),
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
        
    num_classes=len(gdf[target_variable].unique())

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
            num_classes=num_classes 
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
                num_classes=num_classes
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

    ############ data download and creation ############
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
    #chesapeake = ChesapeakeDE(chesapeake_root, crs=naip.crs, res=naip.res, download=True)
    #chesapeake_path = os.path.join(chesapeake_root,'DE_STATEWIDE.tif')
    
    #gdf = get_random_points_on_raster_template(chesapeake_path,naip_path,1000)
    #gdf.to_file(os.path.join(chesapeake_root,'random_chesapeake.shp'), index=False)
    #print(gdf.head())
    ############################################ 


    gdf = gpd.read_file(os.path.join(chesapeake_root,'random_chesapeake.shp'))

    gdf = convert_classes_to_idx(gdf,'pixel_valu')

    MEANS = [122.39, 118.23, 98.1, 120.]
    SDS = [39.81, 37.33, 33.04, 30.]

    res = train_kfold(
            img_dir=naip_root,
            gdf=gdf,
            arch='vit_base_patch16_224.dino',
            nfolds=5,
            target_variable='class_idx',
            batch_size=100,
            epochs=2,
            model_save_dir= chesapeake_root,
            verbose=True,
            means=MEANS,
            sds=SDS,
            lr=1e-4
            )
    preds, gts, folds, losses = res
    print(classification_report(gts, preds, labels=np.unique(gts)))
    print(confusion_matrix(gts, preds, labels=np.unique(gts)))
    matrix = confusion_matrix(gts, preds, labels=np.unique(gts))
    # display = ConfusionMatrixDisplay(matrix, labels=np.unique(gts))
    fig, axs = plt.subplots(ncols=1, figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        gts,
        y_pred=preds,
        ax=axs,
    )
    axs.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('/tmp/chesapeake/confusion.png')
