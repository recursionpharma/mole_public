import os
import pickle  # nosec: B403
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import sparse
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset


class MolDataset(Dataset):
    def __init__(
        self,
        smiles: pd.Series,
        dictionary_inp: dict,
        radius_inp: int = 0,
        labels: Optional[np.ndarray] = None,
        cls_token: bool = False,
        useFeatures_inp: bool = False,
        use_class_weights: bool = False,
    ) -> None:
        """
        A dataset that takes SIMLES as `pandas.Series` and lables (in case of supervised learning) and returns inputs
        as atom environments (or functional atom environments).

        Parameters
        ----------
        smiles : pd.Series
            A pandas Series with SIMLES to be use for training.
        dictionary_inp : str
            Dictionary containing atom environments identifiers as keys and tokes as values.
            This will be used to compute input tokens
        radius_inp : int
            Radius of input atom environments
        labels : numpy.array, optional
            Labels used to train a supervised model
        cls_token : bool
            Flag to add a class (CLS) token to each molecule. Used in supervised training or auxiliary tasks
        useFeatures_inp: bool
            Use functional atom environments for computing input tokens
        """
        self.smiles = smiles
        self.radius_inp = radius_inp
        self.dictionary_inp = dictionary_inp
        self.cls_token = cls_token
        self.labels = labels
        self.useFeatures_inp = useFeatures_inp

        if self.cls_token:
            if "CLS" not in self.dictionary_inp.keys():
                raise KeyError(
                    "cls_token=True but there is no CLS key in the dictionary.",
                    "Either set cls_token=False or add a CLS key to the dictionary",
                )

        self.class_weights = None
        if use_class_weights and self.labels is not None:
            self.class_weights = np.apply_along_axis(
                compute_class_weights, 0, self.labels
            )

    def len(self) -> int:
        raise NotImplementedError

    def get(self, idx: int) -> Data:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Data:
        data_dict = {}
        mol = Chem.MolFromSmiles(self.smiles.iloc[idx])
        dist_mat = Chem.GetDistanceMatrix(mol)
        atomenv_inp = getAtomEnvironments(
            mol, self.dictionary_inp, self.radius_inp, self.useFeatures_inp
        )

        # Get unmasked input tokens and labels for supervised training
        tokens = atomenv_inp
        target_labels = self.labels[idx] if self.labels is not None else []
        data_dict.update(
            {
                "x": torch.tensor(tokens, dtype=torch.long),
                "target_labels": torch.tensor(target_labels),
            }
        )
        if self.class_weights is not None:
            data_dict.update({"class_weights": torch.tensor(self.class_weights[idx])})

        # Transform distance matrix into a sparse matrix
        dist_mat[dist_mat == 1.0e08] = -1
        dist_mat = sparse.coo_matrix(dist_mat + 1)

        if self.cls_token:
            # Add CLS token at position '0' for input tokens, lables and distance matrix
            tokens = np.insert(tokens, 0, self.dictionary_inp["CLS"])
            data_dict.update(
                {
                    "x": torch.tensor(tokens, dtype=torch.long),
                }
            )

            dist_mat = sparse.vstack(
                ((np.zeros(dist_mat.shape[0])[None, :]), dist_mat)
            )  # use 0 as padding index
            dist_mat = sparse.hstack(
                ((np.zeros(dist_mat.shape[0])[:, None]), dist_mat)
            )  # use 0 as padding index

        if data_dict["target_labels"].nelement() == 0:
            data_dict.pop("target_labels")
        data_dict.update(
            {
                "edge_index": torch.tensor(
                    np.array([dist_mat.row, dist_mat.col]), dtype=torch.long
                ),
                "edge_attr": torch.tensor(dist_mat.data, dtype=torch.long),
            }
        )
        return Data.from_dict(data_dict)


def getAtomEnvironments(mol, dictionary, radius, useFeatures):
    disconnected_atoms = [
        i for i, atom in enumerate(mol.GetAtoms()) if atom.GetDegree() == 0
    ]

    info = {}
    atomenv = {}
    AllChem.GetMorganFingerprint(
        mol,
        radius,
        bitInfo=info,
        includeRedundantEnvironments=True,
        useFeatures=useFeatures,
    )
    for k, v in info.items():
        for e in v:
            if e[1] == radius or e[0] in disconnected_atoms:
                if k in dictionary:
                    atomenv[e[0]] = dictionary[k]
                else:
                    # Generic token ID
                    atomenv[e[0]] = dictionary["UNK"]

    atomenv = dict(sorted(atomenv.items()))
    atomenv = list(atomenv.values())
    if len(atomenv) == 0:
        atomenv = [dictionary["UNK"] for _ in range(mol.GetNumAtoms())]

    return atomenv


def open_dictionary(
    dictionary_path, mask_token=None, unk_token=None, cls_token=None, pad_token=None
):
    path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(dictionary_path):
        if os.path.isfile(os.path.join(path, "vocabularies", dictionary_path)):
            dictionary_path = os.path.join(path, "vocabularies", dictionary_path)
        else:
            print(
                "ERROR: Vocabulary in config should be the path to an existing file or the name of a file in",
                os.path.join(path, "vocabularies"),
            )

    with open(dictionary_path, "rb") as f:
        # TODO: don't use pickle
        dictionary = pickle.load(f)  # nosec: B301
    if "PAD" not in dictionary:
        dictionary["PAD"] = pad_token if pad_token is not None else 0
    if "MASK" not in dictionary:
        dictionary["MASK"] = (
            mask_token if mask_token is not None else max(dictionary.values()) + 1
        )
    if "UNK" not in dictionary:
        dictionary["UNK"] = (
            unk_token if unk_token is not None else max(dictionary.values()) + 1
        )
    if "CLS" not in dictionary:
        dictionary["CLS"] = (
            cls_token if cls_token is not None else max(dictionary.values()) + 1
        )

    return dictionary


def compute_class_weights(target: np.ndarray) -> np.ndarray:
    """
    Function that computes class weights for discrete labels.

    Parameters
    ----------
    target : np.ndarray
        Array with discrete labels for all training dataset.
    """
    target: np.ndarray = target.astype(np.float32)
    classes = np.unique(target)
    weight = compute_class_weight(class_weight="balanced", classes=classes, y=target)
    for i in range(len(classes)):
        target[target == classes[i]] = weight[i]
    return target
