import torch
from torch import nn

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.misc import SlimFC

from typing import Callable, Optional, Dict

TensorInitFunc = Callable[[torch.Tensor], None]


def custom_normc_initializer(tensor, std=1.0):
    with torch.no_grad():
        nn.init.normal_(tensor, 0, 1)
        tensor *= std / tensor.pow(2).sum(1, keepdim=True).sqrt()


class TorchCustomWeightsModel(FullyConnectedNetwork):
    """
    use a weight initialization on the linear weights and biases
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 weight_init_func: Optional[TensorInitFunc] = None,
                 bias_init_func: Optional[TensorInitFunc] = None,
                 specific_weight_inits: Optional[Dict[int, TensorInitFunc]] = None,
                 specific_bias_inits: Optional[Dict[int, TensorInitFunc]] = None):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        if not specific_weight_inits:
            specific_weight_inits = dict()

        if not specific_bias_inits:
            specific_bias_inits = dict()

        for layer_idx, layer in enumerate(self._hidden_layers):
            if isinstance(layer, SlimFC):
                linear_part: nn.Linear = layer._model[0]
                self.init_tensor_if_needed(linear_part.weight, layer_idx, weight_init_func, specific_weight_inits)
                self.init_tensor_if_needed(linear_part.bias, layer_idx, bias_init_func, specific_bias_inits)

    @staticmethod
    def init_tensor_if_needed(tensor: torch.Tensor, layer_idx: int, tensor_init_func: Optional[TensorInitFunc],
                              specific_inits: Dict[int, TensorInitFunc]) -> None:
        if layer_idx in specific_inits:
            if specific_inits[layer_idx]:
                specific_inits[layer_idx](tensor)
        elif tensor_init_func:
            tensor_init_func(tensor)
