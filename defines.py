from typing import TypedDict, List, Union, Tuple, Dict
import numpy as np
import torch

class ScoreDict(TypedDict):
    data: Union[List[List[float]], Dict[str, List[float]]] 

class AllScoreDict(TypedDict):
    miou: ScoreDict
    dice: ScoreDict
    f1: ScoreDict
    f2: ScoreDict
    recall: ScoreDict
    precision: ScoreDict
    accuracy: ScoreDict
