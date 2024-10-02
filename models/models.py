def select_model(model: str, *args, **kwargs):
    match model:
        case 'UNet':
            from models.samples.unet import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], kwargs['use_bilinear'])
        case 'RASW-UNet':
            from models.like.unet.RASW import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], kwargs['use_bilinear'])
        case 'SWA-UNet':
            from models.like.unet.SWA import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], kwargs['use_bilinear'])
        case _:
            raise ValueError(f'Not supported model: {model}')
