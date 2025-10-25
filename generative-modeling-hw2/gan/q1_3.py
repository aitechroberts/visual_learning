import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    '''
    Making sure I have the formal math translated to code down
    Algorithm:
    Discriminator's Loss: loss_D = - [ log D(x) + log(1 - D(G(z))) ]
    Is the same as:
        loss_real = BCEWithLogits(real_logit, 1s)
        loss_fake = BCEWithLogits(fake_logit, 0s)
        loss_D = loss_real + loss_fake

    Because BCEWithLogits is BCEWithLogits(a,y)= -(y*log(sigma(a))+(1-y)*log(1-sigma(a)))
    If y=1: -log(D(x)) aka a real sample where we want to maximize log(D(x)) that's the same as minimizing -log(D(x))
    if y=0 -log(1-D(x)) aka a fake sample where we want to minimize log(1-D(x)) that's the same as minimizing the negative as well
    
    '''

    loss_real = F.binary_cross_entropy_with_logits(discrim_real, torch.ones_like(discrim_real))
    loss_fake = F.binary_cross_entropy_with_logits(discrim_fake, torch.zeros_like(discrim_fake))
    loss = loss_real + loss_fake

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    loss = F.binary_cross_entropy_with_logits(discrim_fake, torch.ones_like(discrim_fake))
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
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
