def select_model(model: str, **kwargs):
    match model:
        case 'UNet':
            from .samples.unet import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], False)
        case 'RASW-UNet':
            from .like.unet.RASW import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], False)
        case 'SWA-UNet':
            from .like.unet.SWA import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], False)
        case 'Which-Way-UNet':
            from .like.unet.WhichWayImportant import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], False)
        case 'WayAttention-UNet':
            from .like.unet.WayAttention import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], False)
        case 'MixedUNet':
            from .like.unet.MixedUNet import UNet
            return UNet(kwargs['n_channels'], kwargs['n_classes'], False)
        case _:
            raise ValueError(f'Not supported model: {model}')
