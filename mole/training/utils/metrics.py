from typing import Dict, Optional

import torch
from torch.types import Number
from torchmetrics import Accuracy
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics import Metric
from torchmetrics import PearsonCorrCoef
from torchmetrics import R2Score
from torchmetrics.classification.stat_scores import BinaryStatScores
from torchmetrics.functional.classification.precision_recall import (
    _precision_recall_reduce,
)
from torchmetrics.functional.classification.specificity import _specificity_reduce


class MetricsDict(torch.nn.ModuleDict):
    def __init__(self, **metrics: Metric) -> None:
        """
        An alternative to `torchmetrics.MetricCollection` that is more explicit.
        It does not define an `update` method. Rather, the user should explicitly call
        `update` on the dict's values.
        """
        super().__init__(metrics)

    def compute(self) -> Dict[str, Number]:
        """
        Calls `compute` in a loop over all the metrics in the dict.
        Returns
        -------
        Dict[str, Number]
            The computed metric values.
        """
        return {
            metric_name: metric.compute().item() for metric_name, metric in self.items()
        }

    def reset(self) -> None:
        """
        Calls `reset` in a loop over all metrics in the dict.
        """
        for metric in self.values():
            metric.reset()


class BinaryBalancedAccuracy(BinaryStatScores):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def compute(self) -> torch.Tensor:
        tp, fp, tn, fn = self._final_state()
        binary_specificity = _specificity_reduce(
            tp, fp, tn, fn, average="binary", multidim_average=self.multidim_average
        )
        binary_recall = _precision_recall_reduce(
            "recall",
            tp,
            fp,
            tn,
            fn,
            average="binary",
            multidim_average=self.multidim_average,
        )
        return (binary_specificity + binary_recall) * 0.5


# def get_classification_metrics() -> MetricsDict:
#     return MetricsDict(accuracy = Accuracy(task='binary'),
#                        balanced_accuracy =  BinaryBalancedAccuracy(),
#                        au_roc = AUROC(task="binary"),
#                        au_prc = AveragePrecision(task="binary"))


def get_regression_metrics() -> MetricsDict:
    return MetricsDict(
        mae=MeanAbsoluteError(),
        rmse=MeanSquaredError(squared=False),
        r_squared=R2Score(),
        pearson_corr_coef=PearsonCorrCoef(),
    )


def get_classification_metrics() -> MetricsDict:
    return MetricsDict(accuracy=Accuracy(task="binary"))
