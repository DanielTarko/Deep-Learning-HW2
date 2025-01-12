import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}

class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        for i, out_channels in enumerate(self.channels):
            #add layer
            layers.append(nn.Conv2d(in_channels, out_channels, **self.conv_params))
            in_channels = out_channels
            
            #reg
            if self.activation_type == "lrelu":
                layers.append(nn.LeakyReLU(**self.activation_params))
            elif self.activation_type == "relu":
                layers.append(nn.ReLU(**self.activation_params))
            else:
                raise ValueError(f"Unsupported activation type: {self.activation_type}")


            #pooling
            if (i + 1) % self.pool_every == 0:
                if self.pooling_type == "avg":
                    layers.append(nn.AvgPool2d(**self.pooling_params))
                elif self.pooling_type == "max":
                    layers.append(nn.MaxPool2d(**self.pooling_params))
                else:
                    raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        return nn.Sequential(*layers)

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            dummy_input = torch.zeros(1, *self.in_size)  
            output = self.feature_extractor(dummy_input) 
            #return and then finally
            return output.view(1, -1).shape[1]
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        # TODO:
        #  - Create the MLP part of the model: (FC -> ACT)*M -> Linear
        #  - Use the the MLP implementation from Part 1.
        #  - The first Linear layer should have an input dim of equal to the number of
        #    convolutional features extracted by the convolutional layers.
        #  - The last Linear layer should have an output dim of out_classes.
        mlp: MLP = None
        # ====== YOUR CODE: ======
        # Start with the input feature size
        print(torch.device)

        input_dim = self._n_features()

        # Pair hidden dimensions with the final output class count
        layer_dims = self.hidden_dims + [self.out_classes]

        # Create activations for each hidden layer (no activation for the last layer)
        activations = []
        for _ in range(len(self.hidden_dims)):
            if self.activation_type in ACTIVATIONS:
                activations.append(ACTIVATIONS[self.activation_type](**self.activation_params))
            else:
                raise ValueError(f"Unsupported activation type: {self.activation_type}")
        activations.append(None)  # No activation for the output layer

        # Construct the MLP using the layer dimensions and activations
        mlp = MLP(in_dim=input_dim, dims=layer_dims, nonlins=activations)
        # ========================
        return mlp

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        features = self.feature_extractor(x) 
        return self.mlp(features.view(features.size(0), -1))


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        # Build the main convolutional path
        layers = []
        curr_in_channels = in_channels

        # Add all layers except the last convolution
        for i in range(len(channels) - 1):
            # Add convolution layer
            padding = kernel_sizes[i] // 2
            layers.append(
                nn.Conv2d(
                    curr_in_channels,
                    channels[i],
                    kernel_sizes[i],
                    padding=padding,
                    bias=True
                )
            )
            
            
            # Add dropout if requested
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            
            # Add batch normalization if requested
            if batchnorm:
                layers.append(nn.BatchNorm2d(channels[i]))
            
            activation = ACTIVATIONS[activation_type](**activation_params)
            layers.append(activation)
            curr_in_channels = channels[i]

        # Add final convolution
        padding = kernel_sizes[-1] // 2
        layers.append(
            nn.Conv2d(
                curr_in_channels,
                channels[-1],
                kernel_sizes[-1],
                padding=padding,
                bias=True
            )
        )
        
        self.main_path = nn.Sequential(*layers)

        # Create the shortcut path
        # Only create a conv layer if we need to match dimensions
        if in_channels != channels[-1]:
            self.shortcut_path = nn.Conv2d(
                in_channels,
                channels[-1],
                kernel_size=1,
                bias=False
            )
        else:
            self.shortcut_path = nn.Identity()

    def forward(self, x: Tensor):
        # Compute main path
        main = self.main_path(x)
        
        # Compute shortcut path
        shortcut = self.shortcut_path(x)
        
        # Combine paths and apply final activation
        out = main + shortcut
        out = torch.relu(out)
        
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. NOT the outer projections)
            The length determines the number of convolutions, EXCLUDING the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->5->10.
            The first and last arrows are the 1X1 projection convolutions, 
            and the middle one is the inner convolution (corresponding to the kernel size listed in "inner kernel sizes").
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

      
        channels = [inner_channels[0]]  # First inner channel after input projection
        channels.extend(inner_channels)  # Rest of inner channels
        channels.append(in_out_channels)  # Output projection back to original channels

        # Create the complete kernel sizes list
        # 1x1 for input projection, specified inner kernels, 1x1 for output projection
        kernel_sizes = [1]  # Input projection kernel
        kernel_sizes.extend(inner_kernel_sizes)  # Inner convolution kernels
        kernel_sizes.append(1)  # Output projection kernel

        # Initialize the base ResidualBlock with our computed parameters
        super().__init__(
            in_channels=in_out_channels,  # Input channels
            channels=channels,  # Complete channel list
            kernel_sizes=kernel_sizes,  # Complete kernel size list
            **kwargs  # Pass through any additional arguments
        )


class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions (make sure to use the right stride and padding).
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        #    Reminder: the number of convolutions performed in the bottleneck block is:
        #    2 + len(inner_channels). [1 for each 1X1 proection convolution] + [# inner convolutions].
        # - Use batchnorm and dropout as requested.
        # ====== YOUR CODE: ======
        i = 0 
        num_convs = len(self.channels) 

        convs_remaining = num_convs 

        current_in_channels_count = in_channels 

        while convs_remaining > 0: #while convs_remaining
            group_size = min(self.pool_every, convs_remaining)


            group_channels = self.channels[i:i + group_size]

            use_bottleneck = self.bottleneck and current_in_channels_count == group_channels[-1]
            if not use_bottleneck:
                block = ResidualBlock(
                    in_channels=current_in_channels_count,
                    channels=group_channels,
                    kernel_sizes=[3] * len(group_channels),
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    activation_type=self.activation_type,
                    activation_params=self.activation_params,
                )
            else:
                block = ResidualBottleneckBlock(
                    in_out_channels=group_channels[-1],
                    inner_channels=group_channels[1:-1],  
                    inner_kernel_sizes=[3] * max(0, len(group_channels) - 2),
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    activation_type=self.activation_type,
                    activation_params=self.activation_params,
                )
                

            layers.append(block) # add layer

            if group_size == self.pool_every: #if time to add pool
                layers.append(POOLINGS[self.pooling_type](**self.pooling_params))

            current_in_channels_count = group_channels[-1]
            convs_remaining -= group_size
            i += group_size

        return nn.Sequential(*layers)
