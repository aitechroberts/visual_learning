from __future__ import print_function

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
from voc_dataset import VOCDataset


'''Loss Functions as requested by instructions. 
Reasoning:
            
    PASCOL VOC is a multi-label classification dataset. Each image
    can have multiple labels (e.g. an image can contain both a dog
    and a cat). Therefore, the standard cross-entropy loss funtion
    is not suitable. Instead, I'm trying a binary cross-entropy (BCE)
    loss function that treats each class independently and then averages
    the loss over all classes aka the Logits loss. Meaning that we will
    use BCE with logits loss as found in https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    

Experiments:
1. Stable BCE with logits
    - Problem with this is that it seems to overfit quickly, where the loss
        decreases quickly but the map does not improve and often degrades after 
        the loss reaches a certain point.
    - No amount of regularization (weight decay, dropout, different data augmentations) seems to help
2. Stable BCE with logits and label smoothing
    - This seems to help a lot with the overfitting problem
    - The loss decreases more slowly, but the map improves more consistently, but ultimately collapses after 40-50 epochs
3. Focal Loss
    - This was tremendously better and more stable as it downweights easy examples
      and focuses training on hard negatives. This is important in VOC as most images
      have only a few classes present, so most classes are easy negatives without objects
      present. This is based on the paper "Focal Loss for Dense Object Detection"
      (https://arxiv.org/abs/1708.02002) and the implementation is based on
      https://www.kaggle.com/code/bigironsphere/loss-functions-focal-loss-bce-pytorch
    - This seems to be the best option and is what I'm using now.

'''

def smooth_bce_loss(output, target, wgt, ):
    """
    Manual implementation of BCE with label smoothing (no built-ins)
    Args:
        output: logits from the model (N, C)
        target: binary labels (N, C)
        wgt: class weights (N, C)
    """
    eps=0.05
    eps_safe=1e-8

    # Smooth targets: y_smooth = y*(1-eps) + 0.5*eps
    target_smooth = target * (1.0 - eps) + 0.5 * eps

    # Numerically stable BCE with logits
    max_val = torch.clamp(output, min=0)
    bce_stable = max_val - output * target_smooth + torch.log1p(torch.exp(-output.abs()))

    # Apply weights
    bce_stable = bce_stable * wgt

    # Normalize
    loss = bce_stable.sum() / (wgt.sum() + eps_safe)
    return loss

def focal_loss(output, target, wgt, ):
    """
    Manual implementation of focal loss (no PyTorch built-ins)
    Args:
        output: logits from the model (N, C)
        target: ground truth binary labels (N, C)
        wgt: class weights (N, C)
    """
    gamma=2.0
    alpha=0.25
    eps=1e-8

    # Compute sigmoid probabilities manually
    probs = 1.0 / (1.0 + torch.exp(-output))

    # BCE component
    bce = -(target * torch.log(probs + eps) + (1.0 - target) * torch.log(1.0 - probs + eps))

    # Focal scaling: focus on hard examples
    pt = target * probs + (1.0 - target) * (1.0 - probs)
    focal_weight = alpha * torch.pow((1.0 - pt), gamma)

    # Apply weights (difficulty mask)
    loss_terms = focal_weight * bce * wgt

    # Normalize
    loss = loss_terms.sum() / (wgt.sum() + eps)
    return loss


# Loss Functions above this

def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    print("saving model at ", filename)
    torch.save(model.state_dict(), filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    writer = SummaryWriter()
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0

    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            optimizer.zero_grad()
            output = model(data)

            ##################################################################
            # TODO: Implement a suitable loss function for multi-label
            # classification. You are NOT allowed to use any pytorch built-in
            # functions. Remember to take care of underflows / overflows.
            # Function Inputs:
            #   - `output`: Outputs from the network
            #   - `target`: Ground truth labels, refer to voc_dataset.py
            #   - `wgt`: Weights (difficult or not), refer to voc_dataset.py
            # Function Outputs:
            #   - `output`: Computed loss, a single floating point number
            ##################################################################
            # stable BCE with logits and smoothed.
            loss = focal_loss(output, target, wgt)
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################
            
            loss.backward()
            
            if cnt % args.log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                
                # Log gradients
                for tag, value in model.named_parameters():
                    if value.grad is not None:
                        writer.add_histogram(tag + "/grad", value.grad.cpu().numpy(), cnt)

            optimizer.step()
            
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                print("map: ", map)
                writer.add_scalar("map", map, cnt)
                model.train()
            
            cnt += 1

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map

