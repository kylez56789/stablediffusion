import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock



class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (batchsize, channel, height, width) -> (batchsize, 128, height, width)

            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (batchsize, 128, height, width) -> (batchsize, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (batchsize, 128, height, width) -> (batchsize, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (batchsize, 128, height, width) -> (batchsize, 128, height / 2, width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (batchsize, 128, height / 2, width / 2) -> (batchsize, 256, height / 2, width / 2)
            VAE_ResidualBlock(128, 256),

            # (batchsize, 256, height / 2, width / 2) -> (batchsize, 256, height / 2, width / 2)
            VAE_ResidualBlock(256, 256),

            # (batchsize, 256, height / 2, width / 2) -> (batchsize, 256, height / 4, width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (batchsize, 256, height / 4, width / 4) -> (batchsize, 512, height / 4, width / 4)
            VAE_ResidualBlock(256, 512),

            # (batchsize, 512, height / 4, width / 4) -> (batchsize, 512, height / 4, width / 4)
            VAE_ResidualBlock(512, 512),

            # (batchsize, 512, height / 4, width / 4) -> (batchsize, 512, height / 8, width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (batchsize, 512, height / 8, width / 8) -> (batchsize, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # (batchsize, 512, height / 8, width / 8) -> (batchsize, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # (batchsize, 512, height / 8, width / 8) -> (batchsize, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            # (batchsize, 512, height / 8, width / 8) -> (batchsize, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # (batchsize, 512, height / 8, width / 8) -> (batchsize, 512, height / 8, width / 8)
            nn.GroupNorm(32, 512),
            
            # (batchsize, 512, height / 8, width / 8) -> (batchsize, 512, height / 8, width / 8)
            nn.SiLU(),

            # (batchsize, 512, height / 8, width / 8) -> (batchsize, 8, height / 8, width / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (batchsize, 8, height / 8, width / 8) -> (batchsize, 8, height / 8, width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

        def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            # x: (batchsize, channel, height, width)
            # noise: (batchsize, out_channels, height / 8, width / 8)

            for module in self:
                if getattr(module, 'stride', None) == (2, 2):
                    # (padleft, padright, padtop, padbottom)
                    x = F.pad(x, (0, 1, 0, 1))
                x = module(x)

                
                # (batchsize, 8, height / 8, width / 8) -> two tensors of shape (batchsize, 4, height / 8, width / 8)
                mean, log_variance = torch.chunk(x, 2, dim=1)

                # (batchsize, 4, height / 8, width / 8) -> (batchsize, 4, height / 8, width / 8)
                log_variance = torch.clamp(log_variance, -30, 20)

                # (batchsize, 4, height / 8, width / 8) -> (batchsize, 4, height / 8, width / 8)
                variance = log_variance.exp()

                # (batchsize, 4, height / 8, width / 8) -> (batchsize, 4, height / 8, width / 8)
                stdev = variance.sqrt()

                # transformation to sample from a multivariate gaussian distribution with a diff mean and stdev
                x = mean + stdev * noise

                x *= 0.18215

                return x