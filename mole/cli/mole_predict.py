import argparse
import logging
import os
from typing import List
import warnings

from hydra import compose
from hydra import initialize_config_dir
from hydra.utils import instantiate
import numpy as np
import onnxruntime
import pandas as pd
import pytorch_lightning as pl
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_batch

import mole
from mole.training.data.data_modules import MolDataModule

warnings.filterwarnings("ignore")
mole_path = mole.__path__[0]  # type: ignore[attr-defined]
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=str, help="smiles used as input for the model")
    parser.add_argument(
        "--task", default="regression", help="classification or regression"
    )
    parser.add_argument("--num_tasks", default=1, type=int, help="number of outputs")
    parser.add_argument(
        "--num_classes",
        default=1,
        type=int,
        help="number of classes in classification task",
    )
    parser.add_argument(
        "--pretrained_model", default="null", help="path to pretrained model"
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size use for loader"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of CPU workers"
    )
    parser.add_argument(
        "--accelerator", default="auto", type=str, help="Choose accelerator to use"
    )
    args = parser.parse_args(argv)
    return args


def encode(
    smiles: List[str],
    pretrained_model: str = "null",
    batch_size: int = 32,
    num_workers: int = 4,
    accelerator: str = "auto",
) -> np.ndarray:
    """
    Example in a jupyter notebook:

    from mole import mole_predict
    import pandas as pd

    smiles= ['CCC', 'CCCCCC', 'CC', 'CCCCC']  # list of smiles

    embeddings = mole_predict.encode(smiles=smiles, pretrained_model=<PATH_TO_CHECKPOINT>)

    """
    # Initialize configuration
    pretrained_model = (
        pretrained_model if pretrained_model == "null" else "'" + pretrained_model + "'"
    )
    config_path = os.path.join(mole_path, "training", "configs")
    override_list = [
        "model=finetune",
        "dropout=null",
        "checkpoint_path=" + str(pretrained_model),
        "model.hyperparameters.datamodule.batch_size=" + str(batch_size),
        "model.hyperparameters.datamodule.num_workers=" + str(num_workers),
        "model.hyperparameters.pl_module._target_=mole.training.models.Encoder",
        "model.hyperparameters.pl_module.model._target_=mole.training.models.encoder",
    ]
    with initialize_config_dir(version_base="1.2", config_dir=config_path):
        cfg = compose(config_name="base_config", overrides=override_list)

    # Initialize data module
    datamodule = instantiate(cfg.model.hyperparameters.datamodule, data=smiles)

    # Initialize model and trainer
    model = instantiate(cfg.model.hyperparameters.pl_module)
    trainer = pl.Trainer(
        accelerator=accelerator, enable_progress_bar=False, logger=False
    )

    # Encode SMILES
    outputs = trainer.predict(model, datamodule)
    embeddings = np.concatenate([output.numpy() for output in outputs])  # type: ignore[union-attr]

    return embeddings


def predict_onnx(
    smiles: List[str],
    pretrained_model: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> np.ndarray:
    # Configure ONNX session
    so = onnxruntime.SessionOptions()
    so.inter_op_num_threads = num_workers
    so.intra_op_num_threads = 2
    ort_session = onnxruntime.InferenceSession(pretrained_model, sess_options=so)

    # Initialize data module
    datamodule = MolDataModule(
        data=smiles,
        vocabulary_inp="vocabulary_207atomenvs_radius0_ZINC_guacamole.pkl",
        batch_size=batch_size,
        num_workers=num_workers,
    )
    datamodule.setup("predict")
    dataloader = datamodule.predict_dataloader()

    # Predict and transform outputs
    outputs = []
    for batch in dataloader:
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "input_mask": input_mask.numpy(),
            "relative_pos": relative_pos.numpy(),
        }
        outputs.append(np.concatenate(ort_session.run(None, ort_inputs)))
    predictions = np.concatenate(outputs)

    return predictions


def predict_ckpt(
    smiles: List[str],
    task: str = "regression",
    num_tasks: int = 1,
    num_classes: int = 1,
    pretrained_model: str = "null",
    batch_size: int = 32,
    num_workers: int = 4,
    accelerator: str = "auto",
) -> np.ndarray:
    # Initialize configuration
    pretrained_model = (
        pretrained_model if pretrained_model == "null" else "'" + pretrained_model + "'"
    )
    config_path = os.path.join(mole_path, "training", "configs")
    override_list = [
        "model=finetune",
        "task=" + str(task),
        "num_tasks=" + str(num_tasks),
        "dropout=null",
        "checkpoint_path=" + str(pretrained_model),
        "model.hyperparameters.datamodule.batch_size=" + str(batch_size),
        "model.hyperparameters.datamodule.num_workers=" + str(num_workers),
        "model.hyperparameters.pl_module.model.num_classes=" + str(num_classes),
    ]
    with initialize_config_dir(version_base="1.2", config_dir=config_path):
        cfg = compose(config_name="base_config", overrides=override_list)

    # Initialize data module
    datamodule = instantiate(cfg.model.hyperparameters.datamodule, data=smiles)

    # Initialize model and trainer
    model = instantiate(cfg.model.hyperparameters.pl_module)
    trainer = pl.Trainer(
        accelerator=accelerator, enable_progress_bar=False, logger=False
    )

    # Predict and transform outputs
    outputs = trainer.predict(model, datamodule)
    predictions = np.concatenate(
        [output["logits"].numpy() for output in outputs]  # type: ignore[call-overload, union-attr]
    )

    return predictions


def predict(
    smiles: List[str],
    task: str = "regression",
    num_tasks: int = 1,
    num_classes: int = 1,
    pretrained_model: str = "null",
    batch_size: int = 32,
    num_workers: int = 4,
    accelerator: str = "auto",
) -> np.ndarray:
    """
    Example in a jupyter notebook:

    from mole import mole_predict
    import pandas as pd

    smiles= ['CCC', 'CCCCCC', 'CC', 'CCCCC']  # list of smiles

    predictions = mole_predict.predict(smiles=smiles, pretrained_model=<PATH_TO_CHECKPOINT>)

    df = pd.DataFrame(predictions)
    df.insert (0, 'smiles', smiles)
    df.head()
    """

    if ".onnx" in pretrained_model:
        predictions = predict_onnx(
            smiles, pretrained_model, batch_size=batch_size, num_workers=num_workers
        )
    else:
        predictions = predict_ckpt(
            smiles=smiles,
            task=task,
            num_tasks=num_tasks,
            num_classes=num_classes,
            pretrained_model=pretrained_model,
            batch_size=batch_size,
            num_workers=num_workers,
            accelerator=accelerator,
        )

    return predictions


def main(argv=None):
    args = parse_args(argv)
    filename = None
    if os.path.isfile(args.smiles):
        filename = args.smiles
        loader = getattr(pd, str("read_" + filename.split(".")[-1]))
        df = loader(filename)
        args.smiles = df.smiles.to_list()
        output = predict(**vars(args))
        result = pd.concat(
            [df, pd.DataFrame(output)], ignore_index=True, sort=False, axis=1
        )
        result.to_csv("predictions.csv", index=False)
    else:
        args.smiles = args.smiles.split()
        output = predict(**vars(args))
        return output


if __name__ == "__main__":
    print(main())
