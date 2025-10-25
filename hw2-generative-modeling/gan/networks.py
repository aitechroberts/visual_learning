import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.upscale_factor = upscale_factor

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Implement nearest neighbor upsampling
        # 1. Repeat x channel-wise upscale_factor^2 times
        # 2. Use torch.nn.PixelShuffle to form an output of dimension
        # (batch, channel, height*upscale_factor, width*upscale_factor)
        # 3. Apply convolution and return output
        ##################################################################
        r = self.upscale_factor

        # 1) Repeat channels r^2 times -> (B, C*r^2, H, W)
        x_rep = x.repeat(1, r * r, 1, 1)

        # 2) PixelShuffle spreads repeated channels into spatial dims
        #    -> (B, C, H*r, W*r)
        x_up = F.pixel_shuffle(x_rep, r)

        # 3) Convolution
        out = self.conv(x_up)
        return out
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class DownSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.downscale_ratio = downscale_ratio

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Implement spatial mean pooling
        # 1. Use torch.nn.PixelUnshuffle to form an output of dimension
        # (batch, channel*downscale_factor^2, height, width)
        # Think: using view/reshape instead of an explicit split
        # 2. Then split channel-wise and reshape into
        # (downscale_factor^2, batch, channel, height, width) images
        # 3. Take the average across dimension 0, apply convolution,
        # and return the output
        ##################################################################
        '''
        x.view doesn't shuffle data; it just reinterprets the same contiguous memory with a new shape/strides.

        After pixel_unshuffle(x, r), you have shape (B, C*r*r, H, W). The second axis is a product C·(r²).
        You're free to factor that axis into two adjacent axes in either order:

        (B, C, r*r, H, W) -> the r² part is at axis 2

        (B, r*r, C, H, W) -> the r² part is at axis 1

        Both are valid views because you're only splitting one axis into two factors whose product matches the original size. 
        You are not swapping different original axes — you're just deciding which factor changes faster (which becomes the inner sub-index).
        '''
        r = self.downscale_ratio
        # 1) PixelUnshuffle collects spatial blocks into channels or thought of another way brings subpixels into channels
        #    -> (B, C*r^2, H, W)
        x_unshuff = F.pixel_unshuffle(x, r)
        B, C_r2, H, W = x_unshuff.shape 
        C = C_r2 // (r * r) # Original number of channels before unshuffle 

        # 2) Reshape into (r^2, B, C, H, W) to separate the r^2 subpixel groups
        x_reshaped = x_unshuff.view(B, C, r * r, H, W) # shape: (B, C,  K, H, W)  where K = r*r
        # mean over the r^2 subpixels -> (B, C, H, W)
        x_pooled = x_reshaped.mean(dim=2) # reduce over axis-2 (the K axis)
        # 3) Average across dimension 0 (r^2), then convolve
        out = self.conv(x_pooled)
        return out

        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlockUp(torch.jit.ScriptModule):
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        '''
        Padding of 1 keeps the spatial dimensions the same for kernel size 3
        But would be pad =  kernel_size // 2 for other kernel sizes
        1x1 conv in upsample_residual does not need padding
        '''
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(
                input_channels, n_filters, kernel_size=kernel_size, padding=1, bias=False
            ),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            UpSampleConv2D(input_channels=n_filters, kernel_size=kernel_size, n_filters=n_filters, padding=1),
        )

        self.upsample_residual = UpSampleConv2D(
            input_channels=input_channels, kernel_size=1, n_filters=n_filters, upscale_factor=2, padding=0,
        )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward through the layers and implement a residual
        # connection. Make sure to upsample the residual before adding it
        # to the layer output.
        ##################################################################
        out = self.layers(x)
        res = self.upsample_residual(x)
        return out +  res
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlockDown(torch.jit.ScriptModule):
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d( input_channels, n_filters, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            DownSampleConv2D(input_channels=n_filters, kernel_size=kernel_size, n_filters=n_filters, padding=1),
        )  
        self.downsample_residual = DownSampleConv2D(
            input_channels=input_channels, kernel_size=1, n_filters=n_filters, downscale_ratio=2, padding=0,
        )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward through the layers and implement a residual
        # connection. Make sure to downsample the residual before adding
        # it to the layer output.
        ##################################################################
        out = self.layers(x)
        res = self.downsample_residual(x)
        return out + res
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlock(torch.jit.ScriptModule):
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=1),
        )
        # Assume input_channels == n_filters per the provided template.
        # If they ever differ, add a 1x1 projection:
        # self.proj = nn.Conv2d(input_channels, n_filters, kernel_size=1)  # and use it in forward.
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward the conv layers. Don't forget the residual
        # connection!
        ##################################################################
        out = self.layers(x)
        return out + x  # if you add a projection, use: out + self.proj(x)    
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class Generator(torch.jit.ScriptModule):
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        ##################################################################
        # TODO 1.1: Set up the network layers. You should use the modules
        # you have implemented previously above.
        ##################################################################
        self.dense = nn.Linear(128, 128 * starting_image_size * starting_image_size)

        self.layers = nn.Sequential(
            # 4x4 -> 8x8
            ResBlockUp(input_channels=128, n_filters=128, kernel_size=3),
            # 8x8 -> 16x16
            ResBlockUp(input_channels=128, n_filters=128, kernel_size=3),
            # 16x16 -> 32x32
            ResBlockUp(input_channels=128, n_filters=128, kernel_size=3),

            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self._start_sz = starting_image_size
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward_given_samples(self, z):
        ##################################################################
        # TODO 1.1: Forward the generator assuming a set of samples z has
        # been passed in. Don't forget to re-shape the output of the dense
        # layer into an image with the appropriate size!
        ##################################################################
        # z: (B, 128) -> (B, 128, 4, 4)
        batch_size = z.shape[0]
        x = self.dense(z)  # (B, 128*4*4)
        # reshape to (B, 128, 4, 4)
        x = x.view(batch_size, 128, self._start_sz, self._start_sz)  
        x = self.layers(x)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
        ##################################################################
        # TODO 1.1: Generate n_samples latents and forward through the
        # network.
        ##################################################################
        device = self.dense.weight.device
        z = torch.randn((n_samples, 128), device=device)
        return self.forward_given_samples(z)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class Discriminator(torch.jit.ScriptModule):
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (3): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        ##################################################################
        # TODO 1.1: Set up the network layers. You should use the modules
        # you have implemented previously above.
        ##################################################################
        self.dense = nn.Linear(128, 1)
        self.layers = nn.Sequential(
            # 32x32 -> 16x16
            ResBlockDown(input_channels=3,   n_filters=128, kernel_size=3),
            # 16x16 -> 8x8
            ResBlockDown(input_channels=128, n_filters=128, kernel_size=3),
            # keep 8x8 spatial size
            ResBlock(input_channels=128, n_filters=128, kernel_size=3),
            ResBlock(input_channels=128, n_filters=128, kernel_size=3),
            nn.ReLU(),
        )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward the discriminator assuming a batch of images
        # have been passed in. Make sure to sum across the image
        # dimensions after passing x through self.layers.
        ##################################################################
        # x: (B, 3, 32, 32) -> features: (B, 128, 8, 8)
        feats = self.layers(x)
        # global sum pooling over spatial dims -> (B, 128)
        feats = feats.sum(dim=3).sum(dim=2)
        # final score (B, 1)
        out = self.dense(feats)
        return out
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
