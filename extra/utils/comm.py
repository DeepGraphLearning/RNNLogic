import os
import argparse

import torch
from torch import distributed as dist


def get_rank():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args, unknown = parser.parse_known_args()
    return args.local_rank


def get_world_size():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def reduce(obj, dst=None):

    def recursive_read(obj):
        values = []
        if isinstance(obj, torch.Tensor):
            values += [obj.flatten()]
        elif isinstance(obj, dict):
            for v in obj.values():
                values += recursive_read(v)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            for v in obj:
                values += recursive_read(v)
        else:
            raise ValueError("Unknown type `%s`" % type(obj))
        return values

    def recursive_write(obj, values):
        if isinstance(obj, torch.Tensor):
            new_obj = values[:obj.numel()].view_as(obj)
            values = values[obj.numel():]
            return new_obj, values
        elif isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                new_obj[k], values = recursive_write(v, values)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            new_obj = []
            for v in obj:
                new_v, values = recursive_write(v, values)
                new_obj.append(new_v)
        else:
            raise ValueError("Unknown type `%s`" % type(obj))
        return new_obj, values

    values = recursive_read(obj)
    values = torch.cat(values)

    if dst is None:
        dist.all_reduce(values)
    else:
        dist.reduce(values, dst=dst)

    return recursive_write(obj, values)[0]