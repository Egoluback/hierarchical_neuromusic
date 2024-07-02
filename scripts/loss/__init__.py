from scripts.loss.CELossWrapper import CELossWrapper as CELoss
from scripts.loss.CELossWrapper import CosineCELossWrapper as CosineCELoss
from scripts.loss.CELossWrapper import ExpCosineCELossWrapper as ExpCosineCELoss
from scripts.loss.CELossWrapper import SoftmaxCosineCELossWrapper as SoftmaxCosineCELoss

__all__ = [
    "CELoss",
    "CosineCELoss",
    "ExpCosineCELoss",
    "SoftmaxCosineCELoss"
]
