import logging
import os
from typing import Any, Dict

import hydra
from hydra.utils import instantiate
from lightning_fabric.utilities.seed import seed_everything
import numexpr
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pytorch_lightning as pl

import mole
from mole.training.data.data_modules import MolDataModule
from mole.training.models.base import Model

logger = logging.getLogger(__name__)
config_path = os.path.join("..", "training", "configs")

OmegaConf.register_new_resolver(
    "eval",
    lambda expr: numexpr.evaluate(expr).item(),
    replace=True,
)
OmegaConf.register_new_resolver(
    "get_path",
    lambda x: os.path.dirname(x) if os.path.isfile(x) else None,
    replace=True,
)
OmegaConf.register_new_resolver(
    "full_path", lambda x: os.path.realpath(x) if os.path.isfile(x) else x, replace=True
)
OmegaConf.register_new_resolver(
    "get_filename",
    lambda x: os.path.basename(x) if os.path.isfile(x) else x,
    replace=True,
)
OmegaConf.register_new_resolver(
    "get_data_path",
    lambda x: os.path.join(mole.__path__[0], "../data/", x),
    replace=True,
)
OmegaConf.register_new_resolver(
    "is_not_null",
    lambda x: False if x is None else True,
    replace=True,
)


@hydra.main(version_base="1.2", config_path=config_path, config_name="base_config")
def main(cfg: DictConfig):
    cfg_resolved: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    if cfg.keep_configs is True:
        check_missing(cfg)
        file_name = str("config_file.yaml")
        with open(file_name, "w") as file:
            OmegaConf.save(config=cfg_resolved, f=file)

    if cfg.submit_job is True:
        check_missing(cfg)
        train(config=cfg)


def train(config: DictConfig):
    trainer: pl.Trainer = instantiate(config.model.data.trainer)
    seed = config.model.hyperparameters.get("random_seed", 42)
    logging.info(f"Using RANDOM SEED {seed} to seed everything!")
    seed_everything(seed=seed, workers=True)

    # instantiate the datamodule
    datamodule: MolDataModule = instantiate(config.model.hyperparameters.datamodule)

    config.model.hyperparameters.pl_module.model.vocab_size_inp = len(
        datamodule.dictionary_inp
    )

    # Download TDC data from cloud if needed since it is run only in master node
    datamodule.prepare_data()

    # calculate number of warmup steps
    if not config.model.hyperparameters.pl_module.lr_scheduler.num_warmup_steps:
        num_training_steps, num_warmup_steps = get_num_warmup_steps(trainer, datamodule)
        config.model.hyperparameters.pl_module.lr_scheduler.num_warmup_steps = (
            num_warmup_steps
        )
        if "num_training_steps" in config.model.hyperparameters.pl_module.lr_scheduler:
            config.model.hyperparameters.pl_module.lr_scheduler.num_training_steps = (
                num_training_steps
            )

    # instantiate the model
    model: Model = instantiate(config.model.hyperparameters.pl_module)

    # Log hparams
    datamodule.save_hyperparameters(
        config.model.hyperparameters.datamodule, logger=False
    )
    model.save_hyperparameters(config.model.hyperparameters.pl_module, logger=False)

    # log hyperparameters in all loggers for record keeping
    [logger.log_hyperparams(params=config) for logger in trainer.loggers]  # type: ignore[arg-type]
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


def check_missing(cfg: DictConfig):
    cfg.model.hyperparameters.datamodule.data
    cfg.checkpoint_path
    cfg.model.hyperparameters.pl_module.optimizer.lr
    cfg.model.hyperparameters.pl_module.lr_scheduler.num_warmup_steps


def get_num_warmup_steps(
    trainer: pl.Trainer, datamodule: MolDataModule, frac_warmup_steps: float = 0.1
):
    """
    Returns the total number of training steps and the number of warmup steps
    """
    if frac_warmup_steps < 0 or frac_warmup_steps > 1:
        raise ValueError("frac_warmup_steps should be float between 0 and 1")

    if trainer.max_epochs:
        datamodule.setup("fit")
        num_steps_per_epoch = len(datamodule.train_dataloader())
        num_training_steps = num_steps_per_epoch * trainer.max_epochs
        num_warmup_steps = int(num_training_steps * frac_warmup_steps)
    elif trainer.max_steps > 0:
        num_training_steps = trainer.max_steps
        num_warmup_steps = int(num_training_steps * frac_warmup_steps)
    else:
        raise ValueError(
            "Either max_epochs or max_steps should be specified in the Trainer"
        )

    logging.info(f"num_warmup_steps: {num_warmup_steps}")
    logging.info(f"num_training_steps: {num_training_steps}")

    return num_training_steps, num_warmup_steps


if __name__ == "__main__":
    main()
