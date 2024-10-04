# Adapted from DeBERTa.DeBERTa.apps.models.masked_language_model.MaskedLanguageModel
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from typing import Any, Dict, Optional, Union

from DeBERTa.deberta.config import ModelConfig
import torch
from torch_geometric.data.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_batch

from mole.training.data.utils import TensorDict
from mole.training.models.base import Model
from mole.training.models.mole import AtomEnvEmbeddings
from mole.training.nn.bert import TaskPredictionHead
from mole.training.utils.metrics import MetricsDict


class encoder(torch.nn.Module):
    """Fine tunning of MolE to predict molecular properties"""

    def __init__(
        self,
        deberta_config: dict,
        vocab_size_inp: Optional[int] = None,
        freeze_encoder: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        config = ModelConfig.from_dict(deberta_config)
        config.vocab_size = (
            vocab_size_inp if vocab_size_inp is not None else config.vocab_size
        )
        self.MolE = AtomEnvEmbeddings(config)
        self.config = self.MolE.config

        self.prediction_head = TaskPredictionHead(
            self.config,
            loss_fn=None,
        )

        self.apply(self.init_weights)

        if freeze_encoder:
            for name, param in self.MolE.named_parameters():
                print("Freezing layer: ", name)
                param.requires_grad = False

    def init_weights(self, module):
        """Apply Gaussian(mean=0, std=`config.initializer_range`) initialization to the module.
        Args:
        module (:obj:`torch.nn.Module`): The module to apply the initialization.
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        input_mask=None,
        position_ids=None,
        attention_mask=None,
        relative_pos=None,
    ):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        type_ids = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        else:
            attention_mask = input_mask

        encoder_output = self.MolE(
            input_ids,
            input_mask,
            type_ids,
            output_all_encoded_layers=True,
            position_ids=position_ids,
            relative_pos=relative_pos,
        )
        hidden_states = encoder_output["hidden_states"]
        ctx_layer = hidden_states[-1]  # select last encoder layer
        context_token = ctx_layer[:, 0]  # select embedding of first token ie. CLS token

        return context_token


class Encoder(Model):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: partial(torch.optim.Optimizer),  # type: ignore[valid-type]
        metrics: MetricsDict,
        **kwargs: Any,
    ) -> None:
        super().__init__(optimizer=optimizer, metrics=metrics, **kwargs)
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, float, Any]]:
        output = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            relative_pos=relative_pos,
        )
        return output

    def training_step(self, batch: Data, batch_idx: int):
        print("Method not implemented")

    def validation_step(
        self, batch: Data, batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, float]]:
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)

        output = self(
            input_ids=input_ids,
            input_mask=input_mask,
            position_ids=None,
            attention_mask=None,
            relative_pos=relative_pos,
        )

        return output

    def update_metrics(self, outputs: TensorDict, batch: TensorDict) -> None:
        print("Method not implemented")
