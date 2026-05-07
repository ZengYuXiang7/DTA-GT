from torch import nn
import math

def init_tensor(tensor, init_type, nonlinearity):
    if tensor is None or init_type is None:
        return
    if init_type == "thomas":
        size = tensor.size(-1)
        stdv = 1.0 / math.sqrt(size)
        nn.init.uniform_(tensor, -stdv, stdv)
    elif init_type == "kaiming_normal_in":
        nn.init.kaiming_normal_(tensor, mode="fan_in", nonlinearity=nonlinearity)
    elif init_type == "kaiming_normal_out":
        nn.init.kaiming_normal_(tensor, mode="fan_out", nonlinearity=nonlinearity)
    elif init_type == "kaiming_uniform_in":
        nn.init.kaiming_uniform_(tensor, mode="fan_in", nonlinearity=nonlinearity)
    elif init_type == "kaiming_uniform_out":
        nn.init.kaiming_uniform_(tensor, mode="fan_out", nonlinearity=nonlinearity)
    elif init_type == "orthogonal":
        nn.init.orthogonal_(tensor, gain=nn.init.calculate_gain(nonlinearity))
    else:
        raise ValueError(f"Unknown initialization type: {init_type}")
