import torch


def build_attention_span_mask(
    labels: torch.Tensor,
    relative_pos: torch.Tensor,
    input_mask: torch.Tensor,
    span: int = 3,
) -> torch.Tensor:
    """
    Build the attention span mask required by MolE. This specifies which tokens to
    attend based on their relative position to the masked token.
    """
    labels_mask = (labels == 0).unsqueeze(-1)
    position_mask = torch.logical_or(relative_pos <= 1, relative_pos > span)
    squared_input_mask = input_mask.unsqueeze(1) * input_mask.unsqueeze(2)

    mask = position_mask + labels_mask
    mask = mask * squared_input_mask
    return mask
