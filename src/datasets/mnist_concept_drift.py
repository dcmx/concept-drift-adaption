import torchvision
from torchvision.datasets import MNIST, FashionMNIST
from src.datasets.concept_drift_transforms import AddGaussianNoise
from typing import Any, Callable, Optional, Tuple

class MnistConceptDrift(MNIST):
    def __init__(
            self,
            root: str,
            train: bool = True,
            total_stages: int = 50,
            drift_mode: str = 'hard',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download)
        self.drift_stage = 0
        self.drift_mode = drift_mode
        self.total_stages = total_stages
        self.original_transform = transform
        self.sudden_drift = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)

        if self.sudden_drift:
            if target == 3:
                target = 7
            elif target == 7:
                target = 3
            elif target == 9:
                target = 8
            elif target == 8:
                target = 9

        return img, target

    def increase_drift_stage(self):
        if self.drift_stage < self.total_stages:
            self.drift_stage += 1
            self.transform = torchvision.transforms.Compose(
                [
                    self.original_transform,
                    AddGaussianNoise(mean=0, std=self.drift_stage / self.total_stages, mode=self.drift_mode)
                ]
            )

class FashionMnistConceptDrift(FashionMNIST):
    def __init__(
            self,
            root: str,
            train: bool = True,
            total_stages: int = 50,
            drift_mode: str = 'hard',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download)
        self.drift_stage = 0
        self.drift_mode = drift_mode
        self.total_stages = total_stages
        self.original_transform = transform
        self.sudden_drift = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)

        if self.sudden_drift:
            if target == 3:
                target = 7
            elif target == 7:
                target = 3
            elif target == 9:
                target = 8
            elif target == 8:
                target = 9

        return img, target
    def increase_drift_stage(self):
        if self.drift_stage < self.total_stages:
            self.drift_stage += 1
            self.transform = torchvision.transforms.Compose(
                [
                    self.original_transform,
                    AddGaussianNoise(mean=0, std=self.drift_stage / self.total_stages, mode=self.drift_mode)
                ]
            )
