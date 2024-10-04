from abc import ABCMeta
from abc import abstractmethod
from collections import OrderedDict
from functools import partial
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch

from mole.training.utils.metrics import MetricsDict

ConfiguredOptimizers = Tuple[
    List[torch.optim.Optimizer],
    Optional[List[Dict[str, Union[str, torch.optim.lr_scheduler._LRScheduler]]]],
]  # noqa: E231
TensorDict = Dict[str, torch.Tensor]

logger = logging.getLogger(__name__)


class Model(pl.LightningModule, metaclass=ABCMeta):
    """
    Defines an interface that provides:
    - appropriate, scalable metric computation
    """

    def __init__(
        self,
        optimizer: partial(torch.optim.Optimizer),  # type: ignore[valid-type]
        metrics: MetricsDict,
        lr_scheduler: Optional[partial(torch.optim.lr_scheduler._LRScheduler)] = None,  # type: ignore[valid-type]
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """This the base class MolE models extend. It is a subclass of the PyTorch Lightning
        LightningModule class which offers significant functionality and reduces considerable boilerplate.
        We subclass to include helper functions regarding model export, distributed metric computation,
        finetuning, callbacks, wrangling optimizers & LR schedulers.

        Parameters
        ----------
        optimizer : partial[torch.optim.Optimizer]
            a partially instantiated optimizer defined in Hydra (see Hydra ADR)
        metrics : MetricsDict
            dictionary of metrics - typically the subclass defines this in its __init__ when it calls super()
        lr_scheduler : Optional[partial[torch.optim.lr_scheduler._LRScheduler]], optional
            a partially instatiated Torch LR scheduler, by default None
        checkpoint_path : Optional[str], optional
            Path to a checkpoint from which we would like to fine-tune, by default None
        """
        super().__init__()
        self._optimizer = (
            optimizer  # optimizer is only partially instantiated here via hydra
        )
        self._lr_scheduler = (
            lr_scheduler  # lr_scheduler is only partially instantiated here via hydra
        )
        self.metrics = metrics
        self.checkpoint_path = checkpoint_path

    def prepare_data(self) -> None:
        # Download checkpoints for fine-tuning should be in here.
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        """Base behavior: load state dict from fine-tune ckpt, if applicable."""
        if self.checkpoint_path is not None:
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = [
                val for key, val in state_dict.items() if "state_dict" in key
            ][0]
            state_dict = state_dict[0] if isinstance(state_dict, list) else state_dict
            state_dict = OrderedDict(
                (
                    (
                        "prediction_head." + k
                        if k.startswith("dense.") or k.startswith("classifier.")
                        else k
                    ),
                    v,
                )
                for k, v in state_dict.items()
            )
            state_dict = OrderedDict(
                ("model." + k if not k.startswith("model.") else k, v)
                for k, v in state_dict.items()
            )

            if (
                "model.prediction_head.classifier.bias" in state_dict.keys()
                and state_dict["model.prediction_head.classifier.bias"].shape[0]
                != self.model.prediction_head.classifier.bias.shape[0]
            ):
                keys = [k for k in state_dict.keys() if "MolE" not in k]
                for k in keys:
                    state_dict.pop(k)
            self.load_state_dict(state_dict, strict=False)

        # set padding embeddings to 0 for training
        if stage == "train":
            with torch.no_grad():
                self.model.MolE.encoder.rel_embeddings.weight[0].fill_(0)
                self.model.MolE.embeddings.word_embeddings.weight[
                    getattr(self.model.config, "padding_idx", 0)
                ].fill_(0)

    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> Union[torch.optim.Optimizer, ConfiguredOptimizers]:
        """Pytorch Lightning Models must describe how to construct optimizer(s) & lr scheduler(s).
        Most of the time, the subclass will return a single list with one optimizer and one lr scheduler.
        see: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers
        However, this can also be used in more advanced scenarios, e.g. with multiple optimizers (for a GAN).
        """
        # here, the _partial_ instantiation of the optimizer by hydra is completed
        self.optimizer = self._optimizer(params=self.parameters())
        if self._lr_scheduler is None:
            return self.optimizer  # can either return a single optimizer
        else:
            # check if LR scheduler needs to know total number of steps, e.g. like OneCycleLR
            kwargs = {}
            if "total_steps" in inspect.signature(self._lr_scheduler).parameters:
                # +1 since an extra call to the scheduler may occur
                kwargs["total_steps"] = self.trainer.estimated_stepping_batches + 1
                logger.info(
                    f"Estimating {kwargs['total_steps']} total_steps during training"
                )
            # construct the lr scheduler
            self.lr_scheduler = self._lr_scheduler(self.optimizer, **kwargs)
            # PTL pattern is to return two lists, one with optimizer and one with lr_scheduler
            # subclassses may overwrite this function to use multiple optimizers and schedulers.
            return [self.optimizer], [
                {
                    "scheduler": self.lr_scheduler,
                    "interval": "step",  # force LR scheduler to always go on step, rather than epoch
                }
            ]

    @abstractmethod
    def update_metrics(self, outputs: TensorDict, batch: TensorDict) -> None:
        """
        Update your metric states here. This will be called for every training and validation batch.
        See `photosynthetic.training.core.metrics.MetricsDict` for more info.

        At the end of each training step, metrics will be computed globally, logged, and reset.
        For validation, metrics are accumulated on each rank over all validation batches. At the end of
        the validation epoch, they will be computed globally, logged, and reset.
        """
        pass

    def on_train_batch_end(  # type: ignore[override]
        self,
        outputs: TensorDict,
        batch: TensorDict,
        batch_idx: int,
    ) -> None:
        """
        1. Calls `update_metrics`
        2. Synchronizes / computes metrics across all ranks
        3. Logs the metric values
        4. Resets the metric states
        """
        self.update_metrics(outputs=outputs, batch=batch)
        self.log_dict({"train_" + k: v for k, v in self.metrics.compute().items()})
        self.metrics.reset()

    def on_validation_batch_end(  # type: ignore[override]
        self,
        outputs: TensorDict,
        batch: TensorDict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Calls `update_metrics`.
        """
        self.update_metrics(outputs=outputs, batch=batch)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> TensorDict:
        return self.validation_step(batch=batch, batch_idx=batch_idx)  # type: ignore[return-value]

    def on_validation_epoch_end(self) -> None:
        """
        1. Synchronizes / computes metrics across all ranks
        2. Logs the metric values
        3. Resets the metric states
        """
        self.log_dict({"val_" + k: v for k, v in self.metrics.compute().items()})
        self.metrics.reset()

    def export_onnx(self, filepath: str) -> None:
        filepath = filepath.replace(".ckpt", ".onnx")
        filepath = filepath.replace("lightning_checkpoint", "model")

        output_names = ["output"]
        input_names = ["input_ids", "input_mask", "relative_pos"]
        input_sample = {
            "input_ids": torch.ones((1, 5), dtype=torch.long),
            "input_mask": torch.ones((1, 5), dtype=torch.bool),
            "relative_pos": torch.ones((1, 5, 5), dtype=torch.long),
        }
        dynamic_axes_dict = {
            "input_ids": {0: "bs", 1: "max_num_atoms"},
            "input_mask": {0: "bs", 1: "max_num_atoms"},
            "relative_pos": {0: "bs", 1: "max_num_atoms", 2: "max_num_atoms"},
            "output": {0: "bs"},
        }
        self.to_onnx(
            filepath,
            input_sample,
            export_params=True,
            dynamic_axes=dynamic_axes_dict,
            input_names=input_names,
            output_names=output_names,
            verbose=False,
        )
