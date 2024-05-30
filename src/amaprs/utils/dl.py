import torch
import torch.nn as nn
import itertools
from collections import OrderedDict
import timm
import pandas as pd

class FCHead(nn.Module):
    """
    Simple fully connected neural network to serve as head on top of a backbone.

    Attributes:
        attribute_name: type and description.
        embed_dim (int):
            Dimension of the feature space of the backbone (i.e. input dim of the head)
        num_classes (int):
            Number of classes outputed by the network
        num_layers (int):
            Number of hidden layers of the network (default 3), 
        hidden_dim (int):
            Dimension of the hidden layers (default 512), 
        activation (str or callable, optional): 
            The activation function to use.
            Can be a string representing a built-in activation function ('relu', 'sigmoid', 'tanh', 'identity', 'softmax'),
            or a callable object representing a custom activation function. Default is 'identity'.
    """
    def __init__(
            self, 
            embed_dim: int, 
            num_classes: int, 
            num_layers: int=3, 
            hidden_dim: int=512, 
            activation=None
            ):
        super().__init__()
        self.flatten = nn.Flatten()

         # Check if an activation function is provided
        if activation is not None:
            if callable(activation):  # Check if it's a callable object
                activation_func = activation
            elif isinstance(activation, str):  # Check if it's a string
                if activation == 'relu':
                    activation_func = nn.ReLU()
                elif activation == 'sigmoid':
                    activation_func = nn.Sigmoid()
                elif activation == 'tanh':
                    activation_func = nn.Tanh()
                elif activation == 'identity':
                    activation_func = nn.Identity()
                elif activation == 'softmax':
                    activation_func = nn.Softmax(dim=-1)  # Softmax needs a dimension argument
                else:
                    raise ValueError("Invalid activation function")
            else:
                raise ValueError("Activation must be either a string or a callable object")
        else:
            activation_func = nn.Identity()  # Default to identity function if not specified


        layers = []

        if num_layers > 1:  # Add intermediate layers if num_layers is greater than 1
            layers.append(nn.Linear(embed_dim, hidden_dim))
            layers.append(nn.ReLU())

            for _ in range(num_layers - 2):  # -2 because we already added the first layer and ReLU
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim if num_layers > 1 else embed_dim, num_classes))
        layers.append(activation_func)

        self.linear_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        cls = self.linear_stack(x)
        return cls


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class MergedDataLoader:
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders
        self.dataset = None  # You may need to define a custom dataset for this

    def __iter__(self):
        return itertools.chain(*[iter(dataloader) for dataloader in self.dataloaders])

    def __len__(self):
        return sum(len(dataloader) for dataloader in self.dataloaders)


def check_empty(tensor):
    """
    Find images with a single value inside a tensor.

    Args:
        tensor (torch.Tensor): input tensor of shape (B,C,W,H).

    Returns:
        mask (list): List of booleans with False value if image at this index along the batch dimension is empty.
        empty_idx (list): List of indexes where the empty images are located.
    """

    ##TODO verify that it works well with more than one band 

    mtensor = tensor.view(tensor.size(0), -1)
    mins, _ = torch.min(mtensor, dim=1)
    maxs, _ = torch.max(mtensor, dim=1)
    empty_idx = torch.where(mins == maxs)
    mask = torch.ones(tensor.size(0), dtype=torch.bool)
    mask[empty_idx] = False

    return mask, empty_idx

def check_false_dim(tensor, expected_dim=4, false_index=1):
    """
    Kornia creates an empty dimension after some transforms.
    This function removes this dimension if it exists.

    Args:
        tensor: batched images 
            With Kornia they may be of shape (B,1,C,W,H) instead of (B,C,W,H).
        expected_dim:
            expected number of dimensions of the tensor.
        false_index:
            empty dimension to remove.

    Returns:
        squeezed tensor if num of dimension exceeds the expected.
    """
    if len(tensor.shape) > expected_dim:
        return tensor.squeeze(false_index)
    else:
        return tensor

def save_if_best(current_loss, best_loss, model, model_save_path):
    if current_loss < best_loss:
        best_loss = current_loss # update best_loss value only if current is better
        torch.save(model, model_save_path)
    return best_loss


def convert_classes_to_idx(gdf, target_variable, idx_col_name='class_idx'):
    # Extract unique values from the specified column
    unique_values = sorted(gdf[target_variable].unique())

    # Factorize the unique values to integers between 0 and num_classes
    labels, levels = pd.factorize(unique_values)
    
    # Create a mapping dictionary to map original values to integer labels
    mapping_dict = {orig_val: int_val for orig_val, int_val in zip(unique_values, labels % len(unique_values))}
    
    # Map the original column values to the integer labels
    gdf[idx_col_name] = gdf[target_variable].map(mapping_dict)
    
    return gdf

def correct_json_format(json_path):    #for dinov2 training metrics json
    with open(json_path, 'r+') as file : 
        a = file.read()
        splitted = a.split('}')
        updated_str = splitted[0]
        for i in range(1, len(splitted)-1):
            updated_str += '},' + splitted[i]
        updated_str = '[' + updated_str + '}]'
        file.truncate(0)
        
        file.close()
    file = open(json_path, 'w+')
    file.write(updated_str)
    file.close()

def show_loss(train_metrics_path):


    try:
        data = json.load(open(train_metrics_path))
    except :
        correct_json_format(train_metrics_path)
        data = json.load(open(train_metrics_path))
    N = len(data)
    losses = ['total_loss', 'dino_local_crops_loss', 'dino_global_crops_loss', 'ibot_loss']             #koleo loss

    fig, ax = plt.subplots()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    X = np.array([i for i in range(N)])
    for loss in losses :
        Y = np.zeros(N)
        for j in range(N):
            Y[j] = data[j][loss]             
        plt.plot(X, Y, label = loss)
    plt.legend()
    plt.show()


def attention_stats_shp(shp_path, attention_tif_path):
    gdf = gpd.read_file(shp_path, driver='ESRI Shapefile')
    gdf = gdf.loc[gdf['geometry']!=None]
    
    with rasterio.open(attention_tif_path) as ds :
        tf = ds.meta.copy()['transform']
        bb = (tf[2], ds.width*tf[0]+tf[2], ds.height*tf[4]+tf[5], tf[5])
        gdf = gdf.loc[((gdf['geometry'].bounds['maxx']>bb[0]) &(gdf['geometry'].bounds['maxy']>bb[2]) &(gdf['geometry'].bounds['minx']<bb[1]) &(gdf['geometry'].bounds['miny']<bb[3]))]
        gdf.index = [i for i in range(len(gdf))]
        gdf['bboxes'] = [polygon_to_bbox(gdf['geometry'][i]) for i in range(len(gdf))]
        im = np.transpose(ds.read(), (1,2,0))
        print("image shape : ", im.shape)
        nb_chan = im.shape[-1]
        mean = []
        std = []
        for roi in gdf['bboxes']:
            bottom_left = ds.index(roi[0], roi[3])
            top_right = ds.index(roi[1], roi[2])
            roi_attention = im[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]]
            mean.append(np.array([np.mean(roi_attention[:,:,k]) for k in range(nb_chan)]))
            std.append(np.array([np.std(roi_attention[:,:,k]) for k in range(nb_chan)]))

        gdf['attention_mean'] = mean
        gdf['attention_std'] = std
            
    return gdf, np.array(mean), np.array(std)


###### Following functions will most likely be obsolete with new versions of timm

def load_pretrained_vit(
        model, 
        checkpoint_path ,
        nchannels=1,
        patch_size=14, 
        feat_dim=1024, 
        pos_embed_size=257, 
        ):

    # kernel_size = model.patch_embed.proj.kernel_size
    # stride = model.patch_embed.proj.stride
    # embed_dim = model.patch_embed.proj.out_channels # corresponds to embed_dim
    # print(model.pos_embed)

    model.patch_embed.proj = nn.Conv2d(nchannels, feat_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    model.pos_embed = nn.Parameter(torch.tensor(model.pos_embed[:, :pos_embed_size]))

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

def load_pretrained_vit_and_add_custom_head(
        checkpoint_path=None, 
        model_name='vit_large_patch16_224', 
        patch_size=14, 
        feat_dim=1024, 
        pos_embed_size=257, 
        in_chans=1, 
        out_classes=6, 
        freeze_backbone = True, 
        freeze_proj = False,
        head=None,
        activation='identity'
        ):


    # pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    if checkpoint_path:
        model = timm.create_model(model_name, num_classes=out_classes)
        model.patch_embed.proj = nn.Conv2d(in_chans, feat_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        # model.pos_embed = nn.Parameter(torch.tensor(pretrained.pos_embed[:, :pos_embed_size]))
        checkpoint = torch.load(checkpoint_path)
        try:
            d = checkpoint['teacher']
            d2 = OrderedDict([(k[9:], v) for k, v in d.items() if ('backbone' in k)])
            model.load_state_dict(d2, strict=False)
        except:
            d = checkpoint
            d2 = OrderedDict([(k[9:], v) for k, v in d.items() if ('backbone' in k)])
            model.load_state_dict(d2, strict=False)

    else :
        model = pretrained
        model= vit_first_layer_with_nchan(model, in_chans)

    if head == None:
        model.head = FCHead(feat_dim, out_classes, activation=activation)
    else:
        model.head = head

    # print(model)

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name[:4]=="head"

    if not freeze_proj:
        for name, param in model.named_parameters():
            if ('patch_embed'in name) and ('proj' in name):
                param.requires_grad = True

    return model


def vit_first_layer_with_nchan(model, in_chans=1):

    kernel_size = model.patch_embed.proj.kernel_size
    stride = model.patch_embed.proj.stride
    embed_dim = model.patch_embed.proj.out_channels # corresponds to embed_dim
    # copy the original patch_embed.proj config 
    # except the number of input channels
    new_conv = torch.nn.Conv2d(
            in_chans, 
            out_channels=embed_dim,
            kernel_size=kernel_size, 
            stride=stride
            )
    # copy weigths and biases
    weight = model.patch_embed.proj.weight.clone()
    bias = model.patch_embed.proj.bias.clone()
    with torch.no_grad():
        for i in range(0,in_chans):
            j = i%3 # cycle every 3 bands
            new_conv.weight[:,i,:,:] = weight[:,j,:,:] #band i takes old band j (blue) weights
            new_conv.bias[:] = bias[:]
    model.patch_embed.proj = new_conv

    return model

#######
