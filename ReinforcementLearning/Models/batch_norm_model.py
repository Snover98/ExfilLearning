from torch import nn

from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override


class TorchBatchNormModel(FullyConnectedNetwork):
    """
    Adds batch normalization after every layer except for the last one
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        hidden_layers = list()
        for layer in self._hidden_layers:
            hidden_layers.append(layer)
            if isinstance(layer, SlimFC):
                layer_out_size = layer._model[0].out_features
                hidden_layers.append(nn.BatchNorm1d(layer_out_size))

        # do not add batch norm after the final layer (this is the reason for the indexing until -1)
        self._hidden_layers = nn.Sequential(*hidden_layers[:-1])

    @override(FullyConnectedNetwork)
    def forward(self, input_dict, state, seq_lens):
        # Set the correct train-mode for our hidden module (only important
        # b/c we have some batch-norm layers).
        self._hidden_layers.train(mode=input_dict.get("is_training", False))
        return super().forward(input_dict, state, seq_lens)
