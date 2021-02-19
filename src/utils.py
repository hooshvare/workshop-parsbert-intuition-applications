import torch
import subprocess


def num_lines_in_file(fpath):
    return int(subprocess.check_output('wc -l %s' % fpath, shell=True).strip().split()[0])


def prepare_tgt(
    tgt: torch.Tensor,
    for_label_smoothing: bool = False
):
    # target shifting in order to learn next step
    # _tgt = tgt[:, :-1]

    if for_label_smoothing:
        return tgt[:, :-1], tgt[:, 1:].reshape(-1, 1)

    return tgt[:, :-1]


def make_src_mask(
    input_ids: torch.Tensor,
    pad_idx: int = 0
):
    # input_ids shape: (batch_size, seq_len)
    # mask shape: (batch_size, 1, 1, seq_len)

    batch_size = input_ids.shape[0]

    src_mask = (input_ids != pad_idx).view(batch_size, 1, 1, -1)

    return src_mask


def make_tgt_mask(
    input_ids: torch.Tensor,
    pad_idx: int = 0
):
    # input_ids shape: (batch_size, seq_len)
    # mask shape: (batch_size, 1, seq_len, seq_len)

    batch_size, seq_len = input_ids.shape[:2]
    device = input_ids.device

    # tgt_padding_mask shape: (batch_size, 1, 1, seq_len)
    tgt_padding_mask = (input_ids != pad_idx).view(batch_size, 1, 1, -1)

    # tgt_nolook_forward_mask shape: (1, 1, seq_len, seq_len)
    tgt_nolook_forward_mask = torch.triu(
        torch.ones((1, 1, seq_len, seq_len), device=device)
        ==
        1
    ).transpose(2, 3)

    tgt_mask = tgt_padding_mask & tgt_nolook_forward_mask
    return tgt_mask


def count_of_tokens_masks(
    mask: torch.Tensor
):
    return torch.sum(mask.long())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
