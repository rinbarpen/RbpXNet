from typing import TypedDict

class ScoreDict(TypedDict):
    all: list[float]
    mean: float

class AllScoreDict(TypedDict):
    miou: ScoreDict
    dice: ScoreDict
    f1: ScoreDict
    f2: ScoreDict
    recall: ScoreDict
    precision: ScoreDict
    accuracy: ScoreDict
    