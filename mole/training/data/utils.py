import os
import pickle  # nosec: B403
from typing import Dict

import torch

TensorDict = Dict[str, torch.Tensor]


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
