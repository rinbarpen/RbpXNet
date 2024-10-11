def select_model(model: str, *args, **kwargs):
    match model:
        case 'UNet':
            from models.samples.unet import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], False)
        case 'RASW-UNet':
            from models.like.unet.RASW import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], False)
        case 'SWA-UNet':
            from models.like.unet.SWA import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], False)
        case 'ML-UNet':
            from models.like.unet.MultiLink import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], False)
        case 'Which-Way-UNet':
            from models.like.unet.WhichWayImportant import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], False)
        case 'WayAttention-UNet':
            from models.like.unet.WayAttention import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], False)
        case _:
            raise ValueError(f'Not supported model: {model}')
