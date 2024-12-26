import glob
import os

import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import *
from enum import Enum

class DataType(Enum):
    CSV = 0,  # CSV file
    IMAGE = 1,
    TEXT = 2,
    IMAGE_SERIAL = 3, # Video(no audio, no subtitles)
    AUDIO = 4, 
    VIDEO = 5, 



class DatasetTemplate(Dataset):
    def __init__(self, source_type: DataType, target_type: DataType, 
                 source_path, target_path, 
                 transformer=(None, None), *, source_suffix: str, target_suffix: str, **kwargs):
        super(DatasetTemplate, self).__init__()

        self.source_type = source_type
        self.target_type = target_type
        self.transformer = transformer

        self.sources = [path for path in glob.glob(source_path) if len(source_suffix) == 0 or os.path.splitext(path)[1].endswith(source_suffix)]
        self.targets = [path for path in glob.glob(target_path) if len(target_suffix) == 0 or os.path.splitext(path)[1].endswith(target_suffix)]
        self.config = kwargs

    def __getitem__(self, idx):
        source_path, target_path = self.sources[idx], self.targets[idx]

        return self.__get_data(source_path, self.source_type, True), self.__get_data(target_path, self.target_type, False)


    def __get_data(self, path, data_type: DataType, source_or_target: bool):
        indent = 'source' if source_or_target else 'target'
        transformer = self.transformer[0 if source_or_target else 1]
        config = self.config[indent]

        match data_type:
            case DataType.IMAGE:
                if config['rgb']:
                    image = Image.open(path).convert('RGB')
                    # image = cv2.imread(source_path, cv2.COLOR_BGR2RGB)
                else:
                    image = Image.open(path).convert('L')
                    # image = cv2.imread(source_path, cv2.COLOR_BGR2GRAY)
            case DataType.TEXT:
                pass
            case DataType.CSV:
                pass
            case DataType.VIDEO:
                pass
            case DataType.AUDIO:
                pass
            case DataType.IMAGE_SERIAL:
                pass


        if transformer:
            image = transformer(image)

        return image
