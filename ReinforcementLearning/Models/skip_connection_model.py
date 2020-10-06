import numpy as np
import torch
from torch import nn

from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.modules.skip_connection import SkipConnection


class TorchSkipConnectionModel(FullyConnectedNetwork):
    class FanInLayer(nn.Module):
        def __init__(self, kernel_size):
            super().__init__()
            self.kernel_size = kernel_size

        def forward(self, in_out):
            in_tensor, out_tensor = in_out

            pool_out = nn.functional.avg_pool1d(in_out[0].unsqueeze(1), self.kernel_size).squeeze()
            return pool_out + out_tensor

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        wrapped_layers = list()
        prev_size: int = self._hidden_layers[0]._model[0].out_features
        for layer in self._hidden_layers:
            # only wrap layers that are not the first one
            if isinstance(layer, SlimFC) and len(wrapped_layers) > 0:
                out_size = layer._model[0].out_features

                # if there is no difference in dimensions, no need for a fan in
                connection_fan_in_layer = None
                # if there is a difference, use an average pool and addition to match the output dim
                if prev_size != out_size:
                    connection_fan_in_layer = TorchSkipConnectionModel.FanInLayer(prev_size // out_size)

                skip_layer = SkipConnection(
                    layer,
                    fan_in_layer=connection_fan_in_layer
                )

                wrapped_layers.append(skip_layer)
                prev_size = out_size
            else:
                wrapped_layers.append(layer)

        self._hidden_layers = nn.Sequential(*wrapped_layers)
