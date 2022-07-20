from collections import OrderedDict

import torch


def load_partial_state_dict(model, target_state_dict):

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in target_state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)

    return model


def convert_state_dict_from_ddp(pretrained_state_dict):
    # create new OrderedDict that does not contain string `module.`
    new_state_dict = OrderedDict()
    for k, v in pretrained_state_dict.items():
        if k[:7] == "module.":
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def old_truncated_normal(tensor, mean=0, std=1, n_truncted_stds=2):
    assert std >= 0, f"{std}"
    size = tensor.shape

    # for unit gaussian
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < n_truncted_stds) & (tmp > -1 * n_truncted_stds)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))

    # for target gaussian
    tensor.data.mul_(std).add_(mean)
    lower_bound = mean - 1 * n_truncted_stds * std
    upper_bound = mean + n_truncted_stds * std
    tensor[tensor <= lower_bound] = lower_bound
    tensor[tensor >= upper_bound] = upper_bound
    return tensor


def truncated_normal(tensor, mean=0, std=1, n_truncted_stds=2):
    assert std >= 0, f"{std}"
    size = tensor.shape

    # [n, 1] -> [n, 1, 4]
    tmp = tensor.new_empty(size + (4,), device=tensor.device).normal_()
    tmp.data.mul_(std).add_(mean)

    lower_bound = mean - 1 * n_truncted_stds * std
    upper_bound = mean + n_truncted_stds * std
    valid = (tmp < upper_bound) & (tmp > lower_bound)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))

    try:
        assert torch.all(tensor >= lower_bound), f"{torch.min(tensor)}"
        assert torch.all(tensor <= upper_bound), f"{torch.max(tensor)}"
    except:
        # fmt: off
        print("\nin truncated normal lower bound: ", tensor.shape, lower_bound, torch.min(tensor), torch.sum(tensor >= lower_bound))
        print("\nin truncated normal upper bound: ", tensor.shape, upper_bound, torch.max(tensor), torch.sum(tensor <= lower_bound))
        tensor[tensor <= lower_bound] = lower_bound
        tensor[tensor >= upper_bound] = upper_bound
        # fmt: on

    return tensor


def transform_vectors(matrix: torch.Tensor, vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = torch.matmul(vectors4, matrix.T)
    return res


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def torch_dot(x: torch.Tensor, y: torch.Tensor):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)
