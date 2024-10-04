# Adapted from DeBERTa.DeBERTa.apps.models.masked_language_model.MaskedLanguageModel
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from functools import partial
from typing import Any, Dict, Optional, Union

from DeBERTa.deberta.cache_utils import load_model_state
from DeBERTa.deberta.config import ModelConfig
import torch
from torch_geometric.data.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric

from mole.training.data.utils import TensorDict
from mole.training.models.base import Model
from mole.training.models.utils.masks import build_attention_span_mask
from mole.training.nn.bert import BertEmbeddings
from mole.training.nn.bert import BertEncoder
from mole.training.nn.bert import TaskPredictionHead
from mole.training.utils.metrics import MetricsDict

__all__ = ["AtomEnvEmbeddings"]


class AtomEnvEmbeddings(torch.nn.Module):
    """ AtomEnvEmbeddings is a DeBERTa encoder
  This module is composed of the input embedding layer with stacked transformer layers with disentangled attention.

  Parameters:
    config:
      A model config class instance with the configuration to build a new model. The schema is \
          similar to `BertConfig`, for more details, please refer :class:`~DeBERTa.deberta.ModelConfig`

    pre_trained:
      The pre-trained DeBERTa model, it can be a physical path of a pre-trained DeBERTa model or \
          a released configurations, i.e. [**base, large, base_mnli, large_mnli**]

  """

    def __init__(self, config=None, pre_trained=None):
        super().__init__()
        state = None
        if pre_trained is not None:
            state, model_config = load_model_state(pre_trained)
            if config is not None and model_config is not None:
                for k in config.__dict__:
                    if k not in [
                        "hidden_size",
                        "intermediate_size",
                        "num_attention_heads",
                        "num_hidden_layers",
                        "vocab_size",
                        "max_position_embeddings",
                    ]:
                        model_config.__dict__[k] = config.__dict__[k]
            config = copy.copy(model_config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.config = config
        self.pre_trained = pre_trained
        self.apply_state(state)

    def forward(
        self,
        input_ids,
        input_mask=None,
        attention_mask=None,
        token_type_ids=None,
        output_all_encoded_layers=True,
        position_ids=None,
        return_att=False,
        relative_pos=None,
    ):
        if input_mask is None:
            input_mask = torch.ones_like(input_ids)
        if attention_mask is None:
            attention_mask = input_mask
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        ebd_output = self.embeddings(
            input_ids.to(torch.long),
            token_type_ids.to(torch.long),
            position_ids,
            input_mask,
        )
        embedding_output = ebd_output["embeddings"]
        encoder_output = self.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            return_att=return_att,
            relative_pos=relative_pos,
        )
        encoder_output.update(ebd_output)
        return encoder_output

    def apply_state(self, state=None):
        """ Load state from previous loaded model state dictionary.

      Args:
        state (:obj:`dict`, optional): State dictionary as the state returned by torch.module.state_dict(), \
            default: `None`. \
            If it's `None`, then will use the pre-trained state loaded via the constructor to re-initialize \
            the `DeBERTa` model
    """
        if self.pre_trained is None and state is None:
            return
        if state is None:
            state, config = load_model_state(self.pre_trained)
            self.config = config

        prefix = ""
        for k in state:
            if "embeddings." in k:
                if not k.startswith("embeddings."):
                    prefix = k[: k.index("embeddings.")]
                break

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        self._load_from_state_dict(
            state,
            prefix=prefix,
            local_metadata=None,
            strict=True,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )


class Supervised(torch.nn.Module):
    """Fine tunning of MolE to predict molecular properties"""

    def __init__(
        self,
        deberta_config: dict,
        loss_fn: torch.nn.modules.loss._Loss,
        num_tasks: int = 1,
        num_classes: int = 1,
        dropout: Optional[float] = None,
        vocab_size_inp: Optional[int] = None,
        freeze_encoder: bool = False,
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
            loss_fn=loss_fn,
            num_tasks=num_tasks,
            num_classes=num_classes,
            dropout=dropout,
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
        labels=None,
        position_ids=None,
        attention_mask=None,
        relative_pos=None,
        aux_labels=None,
    ):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        type_ids = None
        if labels is not None:
            labels = labels.to(device)
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
        logits, loss, labels = self.prediction_head(context_token, labels)

        output = {"logits": logits}
        if labels is not None:
            output.update({"loss": loss, "labels": labels})

        return output


class MolE(Model):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: partial(torch.optim.Optimizer),  # type: ignore[valid-type]
        metrics: MetricsDict,
        attention_span_masking: Optional[int] = None,
        aux_loss_lambda: Optional[float] = 0.0,
        lr_scheduler: Optional[partial(torch.optim.lr_scheduler._LRScheduler)] = None,  # type: ignore[valid-type]
        **kwargs: Any,
    ) -> None:
        super().__init__(
            optimizer=optimizer, metrics=metrics, lr_scheduler=lr_scheduler, **kwargs
        )
        self.model = model
        self.attention_span_masking = attention_span_masking
        self.aux_loss_lambda = aux_loss_lambda
        self.metrics.update(
            MetricsDict(
                mean_loss=MeanMetric(),
                mean_main_loss=MeanMetric(),
                mean_aux_loss=MeanMetric(),
            )  # type:ignore[arg-type]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
        aux_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, float, Any]]:
        output = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
            labels=labels,
            position_ids=position_ids,
            attention_mask=attention_mask,
            relative_pos=relative_pos,
            aux_labels=aux_labels,
        )
        return output

    def training_step(
        self, batch: Data, batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, float]]:
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
        labels = batch.target_labels if "target_labels" in batch.keys() else None
        aux_labels = batch.aux_labels if "aux_labels" in batch.keys() else None

        attention_mask = None
        if self.attention_span_masking:
            attention_mask = build_attention_span_mask(
                labels, relative_pos, input_mask, span=self.attention_span_masking
            )

        output = self(
            input_ids=input_ids,
            input_mask=input_mask,
            labels=labels,
            position_ids=None,
            attention_mask=attention_mask,
            relative_pos=relative_pos,
            aux_labels=aux_labels,
        )
        dict_out = {"logits": output["logits"]}
        if labels is not None:
            if "class_weights" in batch.keys():
                output["loss"] *= batch.class_weights
            main_loss = output["loss"].mean()
            aux_loss = output["aux_loss"].mean() if "aux_loss" in output.keys() else 0
            aux_loss *= self.aux_loss_lambda
            loss: torch.Tensor = main_loss + aux_loss
            dict_out.update(
                {
                    "loss": loss,
                    "main_loss": main_loss,
                    "aux_loss": aux_loss,
                    "labels": output["labels"],
                }
            )

        return dict_out

    def validation_step(
        self, batch: Data, batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, float]]:
        return self.training_step(batch, batch_idx)

    def update_metrics(self, outputs: TensorDict, batch: TensorDict) -> None:
        for k in self.metrics:
            if k.replace("mean_", "") in outputs.keys():
                self.metrics[k].update(value=outputs[k.replace("mean_", "")])
            else:
                self.metrics[k].update(
                    preds=outputs["logits"], target=outputs["labels"]
                )
