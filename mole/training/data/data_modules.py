from typing import List, Optional, Union

import pandas as pd
from pytorch_lightning import LightningDataModule
import tdc
from torch_geometric.loader import DataLoader

from mole.training.data.datasets import MolDataset
from mole.training.data.utils import open_dictionary


class MolDataModule(LightningDataModule):
    def __init__(
        self,
        data: Union[str, pd.Series, List[str]],
        vocabulary_inp: str,
        validation_data: Optional[str] = None,
        MASK_token: Optional[str] = None,
        UNK_token: Optional[str] = None,
        CLS_token: Optional[str] = None,
        radius_inp: int = 0,
        useFeatures_inp: bool = False,
        use_class_weights: bool = False,
        batch_size: Optional[int] = None,
        num_workers: int = 4,
        folds: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.vocabulary_inp = vocabulary_inp
        self.validation_data = validation_data
        self.MASK_token = MASK_token
        self.UNK_token = UNK_token
        self.CLS_token = CLS_token
        self.radius_inp = radius_inp
        self.useFeatures_inp = useFeatures_inp
        self.use_class_weights = use_class_weights
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.folds = folds
        self.prepare_data_per_node = False

        self.dictionary_inp = open_dictionary(
            self.vocabulary_inp,
            mask_token=self.MASK_token,
            unk_token=self.UNK_token,
            cls_token=self.CLS_token,
        )
        self.tdc_benchmark_dataset_names = [
            y
            for x in tdc.benchmark_deprecated.benchmark_names["admet_group"].values()
            for y in x
        ]

    def prepare_data(self):
        # Reserve to download data from cloud since it is run only in master node
        # Download TDC data for ADMET Group
        if isinstance(self.data, str):
            if self.data.lower() in self.tdc_benchmark_dataset_names:
                tdc.BenchmarkGroup(name="ADMET_Group")
        return None

    def setup(self, stage: str):
        if stage == "fit" and isinstance(self.data, str):
            # Assign train/val datasets for use in dataloaders
            smiles_train, labels_train, smiles_val, labels_val = self.get_smiles_labels(
                self.data
            )
            if isinstance(self.validation_data, str):
                _, _, smiles_val, labels_val = self.get_smiles_labels(
                    self.validation_data
                )

            # Tran / Val datasets
            self.mol_train = MolDataset(
                smiles_train,
                self.dictionary_inp,
                labels=labels_train,
                cls_token=True,
                radius_inp=self.radius_inp,
                useFeatures_inp=self.useFeatures_inp,
                use_class_weights=self.use_class_weights,
            )

            self.mol_val = MolDataset(
                smiles_val,
                self.dictionary_inp,
                labels=labels_val,
                cls_token=True,
                radius_inp=self.radius_inp,
                useFeatures_inp=self.useFeatures_inp,
                use_class_weights=self.use_class_weights,
            )

        if stage == "test" and isinstance(self.data, str):
            smiles_test, labels_test, _, _ = self.get_smiles_labels(self.data)
            self.mol_test = MolDataset(
                smiles_test,
                self.dictionary_inp,
                labels=labels_test,
                cls_token=True,
                radius_inp=self.radius_inp,
                useFeatures_inp=self.useFeatures_inp,
                use_class_weights=self.use_class_weights,
            )

        if stage == "predict":
            if isinstance(self.data, str):
                smiles, _, _, _ = self.get_smiles_labels(self.data)
            else:
                smiles = pd.Series(self.data).astype("string[pyarrow]")  # type: ignore[call-overload]
            self.mol_predict = MolDataset(
                smiles,
                self.dictionary_inp,
                labels=None,
                cls_token=True,
                radius_inp=self.radius_inp,
                useFeatures_inp=self.useFeatures_inp,
                use_class_weights=self.use_class_weights,
            )

    def train_dataloader(self):
        return DataLoader(
            self.mol_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mol_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mol_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mol_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def get_smiles_labels(self, data_path: str):
        if data_path.lower() in self.tdc_benchmark_dataset_names and isinstance(
            self.folds, int
        ):
            tdc_ADMET_group = tdc.BenchmarkGroup(name="ADMET_Group")
            train, valid = tdc_ADMET_group.get_train_valid_split(
                benchmark=data_path.lower(), split_type="default", seed=self.folds
            )
            smiles_train = train.Drug.astype("string[pyarrow]")
            labels_train = train.Y.to_numpy()
            smiles_val = valid.Drug.astype("string[pyarrow]")
            labels_val = valid.Y.to_numpy()
            return smiles_train, labels_train, smiles_val, labels_val

        else:
            loader = getattr(pd, str("read_" + data_path.split(".")[-1]))
            df = loader(data_path)

            if "folds" in df.columns and isinstance(self.folds, int):
                df_train = df[df.folds != self.folds]
                df_val = df[df.folds == self.folds]

                df_train.drop(columns="folds", inplace=True)
                df_val.drop(columns="folds", inplace=True)

                smiles_train = df_train.smiles.astype("string[pyarrow]")
                labels_train = (
                    df_train.iloc[:, 1:].to_numpy()
                    if len(df_train.columns[1:]) > 0
                    else None
                )
                smiles_val = df_val.smiles.astype("string[pyarrow]")
                labels_val = (
                    df_val.iloc[:, 1:].to_numpy()
                    if len(df_val.columns[1:]) > 0
                    else None
                )

                return smiles_train, labels_train, smiles_val, labels_val
            else:
                if "folds" in df.columns:
                    df.drop(columns="folds", inplace=True)
                smiles_train = df.smiles.astype("string[pyarrow]")
                labels_train = (
                    df.iloc[:, 1:].to_numpy() if len(df.columns[1:]) > 0 else None
                )
                return smiles_train, labels_train, smiles_train, labels_train
