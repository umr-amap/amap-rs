from collections import OrderedDict
import torch

import timm

def vit_first_layer_with_nchan(
        model,
        in_chans=1,
        embed_dim=768,
        patch_size=16,
        ):

    # cf. https://github.com/facebookresearch/dino/issues/214
    # create empty proj layer
    new_conv = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    weight = model.patch_embed.proj.weight.clone()
    bias = model.patch_embed.proj.bias.clone()
    with torch.no_grad():
        for i in range(0,in_chans):
            j = i%3 # cycle every 3 bands
            new_conv.weight[:,i,:,:] = weight[:,j,:,:] #band i takes old band j (blue) weights
        new_conv.bias[:] = bias[:]
    model.patch_embed.proj = new_conv

    return model

def load_dino_weights(
        model, 
        checkpoint_path ,
        strict_loading,
        map_location=torch.device('cpu'),
        ):

    checkpoint = torch.load(checkpoint_path,map_location=map_location)
    if 'teacher' in checkpoint:
        d = checkpoint['teacher']
        ## this removes the 'backbone.backbone.' part in param names (18 char long)
        d2 = OrderedDict([(k[18:], v) for k, v in d.items() if ('backbone' in k)])

        model.load_state_dict(d2, strict=strict_loading)

    return model


if __name__ == "__main__":
    

    ## Load model pretrained with DINO with correct number of bands
    ## but initialized via timm

    checkpoint_path = "/path/to/pretrained/model.pth"

    model = timm.create_model('vit_base_patch16_224',
                              in_chans=6,
                              num_classes=1,
                              )

    model = load_dino_weights(
            model,
            checkpoint_path=checkpoint_path,
            strict_loading=False, ## Check with strict_loading=True first.
            )
    print(model)

    
    ## Load model trained on RGB and initialized via timm
    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m',
                              in_chans=3,
                              num_classes=1,
                              checkpoint_path=checkpoint_path,
                              )
    
    model =  vit_first_layer_with_nchan(
            model,
            in_chans=6,
            embed_dim=768, # check with model arch
            patch_size=14, # check with model arch
            )
    print(model)
