import sys
import warnings
from collections import OrderedDict

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import kornia.augmentation as K
import timm

# torchgeo
from torchgeo.datasets import BoundingBox,GeoDataset, stack_samples, unbind_samples
from torchgeo.samplers import GridGeoSampler
from torchgeo.transforms import AugmentationSequential

# data
import numpy as np
warnings.filterwarnings("ignore")

# custom modules
from utils.geo import change_tif_resolution
from utils.geo import array_to_geotiff


def inference(
        dataset:GeoDataset, 
        model:nn.Module,
        size:int,
        batch_size:int, 
        roi=None,
        ):
    """
    Infer a model on a ROI

    Args:
        dataset:GeoDataset
            The trochgeo dataset to infer on.
        model:nn.Module,
            The model to produce features
        size:int,
            Sampling size (in pixels). Also defines the stride for now.
        batch_size:int, 
        roi=None,
            Subregion to sample on the dataset

    Returns:
        type and description of the returned object.macro_img, bboxes, patch_size
        macro_img: 
            Reconstructed output raster as a numpy array
        bboxes: 
            List of bounding boxes passed in the dataloader.
            Useful to get the top left corner afterward and reproject the array as a geotiff. 
        patch_size:
            size of the output pixels in native pixels. Also useful to reproject the array.
    """

    model.eval()
    model = model.cuda()

    if roi:
        sampler = GridGeoSampler(dataset, size = size, stride=size, roi = roi)
    else:
        sampler = GridGeoSampler(dataset, size = size, stride=size)

    dataloader = DataLoader(
            dataset, 
            sampler=sampler, 
            collate_fn=stack_samples, 
            shuffle=False, 
            batch_size = batch_size
            )

    # Initialize tensor used to store the projected feature space
    feat_img = None
    bboxes = []

    for i, batch in enumerate(dataloader):
        print(f"{i / len(dataloader):.2%}", end="\r")

        images = batch['image']
        if len(images.shape) > 4:
            images = images.squeeze(1)
        images = images.type(torch.float)
        images = images.cuda()

        for sample in unbind_samples(batch):
            bboxes.append(sample['bbox'])

        with torch.no_grad():
            feat = model(images)
            _,preds = torch.max(feat, 1)

            if i == 0 :
                patch_size = images.shape[2]

            preds = preds.detach().cpu().numpy()

            # Concatenate the transformed features to the existing tensor
            if feat_img is None:
                feat_img = preds
            else:
                feat_img = np.concatenate((feat_img, preds), axis=0)

    del model

    Nx = len(bboxes)
    for i in range(1, len(bboxes)):
        if bboxes[i][0] < bboxes[i - 1][0]:
            Nx = i
            break
    Ny = int(len(bboxes) / Nx)
    print('\n')
    print(Nx, Ny)

    macro_img = reconstruct_img(feat_img, Nx, Ny)

    return macro_img, bboxes, patch_size


def reconstruct_img(div_images, Nx, Ny):
    """
    recustruct numpy array of agregated cls_token features into a raster.

    Args:
        div_image:
            numpy array containing agregated features.
        Nx:
            Number of samples in a row in the GridGeoSampler
        Ny:
            Number of samples in a col in the GridGeoSampler

    Returns:
        aggregated_image:
            numpy array of shape (Ny,Nx,feature_dim).
    """
    # Initialize the aggregated image
    aggregated_image = np.zeros((Ny, Nx, 1), dtype=np.float16)

    # Iterate over rows of the original grid
    for j in range(Ny):
        print(f"{j / Ny:.2%}", end="\r")
        # Iterate over columns of the original grid
        for i in range(Nx):
            idx = (Ny-1-j) * Nx + i
            if idx < len(div_images):
                x_start = i 
                x_end = i + 1 
                y_start = j
                y_end = j + 1
                aggregated_image[y_start:y_end, x_start:x_end, :] = div_images[idx]
    
    return aggregated_image

def load_weights(
        model, 
        checkpoint_path ,
        ):

    checkpoint = torch.load(checkpoint_path)
    if 'teacher' in checkpoint:
        d = checkpoint['teacher']
        d2 = OrderedDict([(k[9:], v) for k, v in d.items() if ('backbone' in k)])
        model.load_state_dict(d2, strict=False)
    if 'model' in checkpoint:
        d = checkpoint['model']
        d2 = OrderedDict([(k, v) for k, v in d.items() if ('decoder_blocks' not in k)])
        model.load_state_dict(d2, strict=False)

    return model

if __name__ == "__main__":

    
    # ## Example workflow
    import os
    import tempfile
    from torchgeo.datasets import NAIP
    from torchgeo.datasets.utils import download_url
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    naip_root = os.path.join(tempfile.gettempdir(), "naip")
    proj_dir = os.path.join(tempfile.gettempdir(), "proj")
    proj_dir = os.path.expanduser(proj_dir)
    os.makedirs(proj_dir, exist_ok=True)

    naip_url = (
        "https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/"
    )

    # tile may not download properly, leading to a `TIFFReadEncodedTile() failed` error
    # a simple wget with this url probably will solve the issue
    tile = "m_3807511_ne_18_060_20181104.tif"
    download_url(naip_url + tile, naip_root)


    dataset = NAIP(naip_root)

    MEANS = [122.39, 118.23, 98.1, 120.]
    SDS = [39.81, 37.33, 33.04, 30.]

    transform = AugmentationSequential(
            T.ConvertImageDtype(torch.float32), # change dtype for normalize to be possible
            K.Normalize(MEANS,SDS), # normalize occurs only on raster, not mask
            K.Resize((224, 224)),  # resize to 224*224 pixels, regardless of sampling size
            data_keys=["image"],
            )
    dataset.transforms = transform

    bb=dataset.bounds
    xlim = bb[0] + (bb[1]-bb[0])* 0.1
    xlim_max = bb[0] + (bb[1]-bb[0])* 0.5
    ylim = bb[2] + (bb[3]-bb[2])* 0.9
    ylim_max = bb[2] + (bb[3]-bb[2])* 0.1
    roi=BoundingBox(xlim, xlim_max, ylim_max, ylim, bb[4], bb[5])

    # ## init via timm
    model = timm.create_model(
            'vit_base_patch16_224.dino', 
            in_chans=len(MEANS), 
            checkpoint_path='/tmp/chesapeake/best_5.pth',
            num_classes=9
            )

    # checkpoint_path = '/tmp/chesapeake/best_5.pth'
    # state_dict = torch.load(checkpoint_path)
    # # print(state_dict)
    # model.load_state_dict(state_dict)
    # model = load_weights(model, checkpoint_path)

    macro_img, bboxes, patch_size = inference(dataset,model,size=224,batch_size=100)

    print('to geotiff')
    top_left = (bboxes[0].minx, bboxes[-1].maxy)
    array_to_geotiff(
            array=macro_img,
            output_file='./out/proj.tif',
            top_left_corner_coords=top_left,
            pixel_width=dataset.res * patch_size,
            pixel_height=dataset.res * patch_size,
            crs=dataset.crs,
            # dtype='int8', ## if clusters
            )

    # # # go back to the original spatial resolution if needed
    # # with rasterio.open(os.path.join(naip_root,tile)) as template_src:
    # #     # Get the template resolution
    # #     orig_resolution = template_src.transform[0]

    # # change_tif_resolution(f'{proj_dir}/proj.tif',f'{proj_dir}/proj_rescale.tif', orig_resolution)


