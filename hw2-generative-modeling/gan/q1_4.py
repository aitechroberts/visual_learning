import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    '''
    Simpler to understand in terms of MSELoss especially since we have the
    least squares being used. Algorithm:
    Discriminator Loss (a = 0, b = 1): loss_D = 0.5 * [(D(x) - 1)^2 + (D(G(z)) - 0)^2]
    Generator Loss: (c = 1): 0.5 * (D(G(z)) - 1)^2
    This translates to:
        loss_real = 0.5 * MSEWithLogits(real_logit, 1s)
        loss_fake = 0.5 * MSEWithLogits(fake_logit, 0s)
    loss_D = 0.5 * (loss_real + loss_fake)
    '''
    loss_real = 0.5 * F.mse_loss(discrim_real, torch.ones_like(discrim_real))
    loss_fake = 0.5 * F.mse_loss(discrim_fake, torch.zeros_like(discrim_fake))
    loss = loss_real + loss_fake

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for generator.
    ##################################################################
    # LSGAN generator (c = 1): 0.5 * (D(G(z)) - 1)^2
    loss = 0.5 * F.mse_loss(discrim_fake, torch.ones_like(discrim_fake))
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss

if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
