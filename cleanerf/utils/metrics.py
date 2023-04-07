"""
Metrics and losses.
"""

import torch


def mse(pred, gt):
    """
    Computes MSE between pred and gt.
    Args:
        pred: predicted voxels of shape (batch_size, c, h, w).
        gt: ground truth voxels of shape (batch_size, c, h, w).
    Returns:
        The MSE.
    """
    return torch.mean((pred - gt) ** 2)


def psnr(pred, gt):
    """
    Computes PSNR between pred and gt.
    Args:
        pred: predicted voxels of shape (batch_size, c, h, w).
        gt: ground truth voxels of shape (batch_size, c, h, w).
    Returns:
        The PSNR.
    """
    return 10 * torch.log10(1 / mse(pred, gt))


def voxel_iou(pred, gt):
    """
    Computes the intersection over union for y_true and y_pred.
    Args:
        y_true: ground truth voxels of shape (batch_size, x, y, z).
        y_pred: predicted voxels of shape (batch_size, x, y, z).
    Returns:
        The intersection over union.
    """
    # check that the inputs are valid
    if pred.shape != gt.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    # squeeze if possible
    shape = pred.shape
    if len(shape) == 5 and shape[1] == 1:
        pred = pred.squeeze(1)
        gt = gt.squeeze(1)
        shape = pred.shape
    if len(shape) != 4:
        raise ValueError("y_true and y_pred must have 4 dimensions.")
    x, y, z = shape[1:]
    if x != y or y != z:
        raise ValueError("x, y, and z must be equal.")

    pred = pred > 0
    gt = gt == 1

    intersection = torch.sum(pred & gt, dim=(1, 2, 3))
    union = torch.sum(pred | gt, dim=(1, 2, 3))
    # add a small epsilon to avoid division by zero
    return torch.mean(intersection / (union + 1e-6))


def voxel_acc(pred, gt):
    """
    Computes the accucary for y_true and y_pred.
    Args:
        y_true: ground truth voxels of shape (batch_size, x, y, z).
        y_pred: predicted voxels of shape (batch_size, x, y, z).
    Returns:
        The accuacy.
    """
    # check that the inputs are valid
    if pred.shape != gt.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    # squeeze if possible
    shape = pred.shape
    if len(shape) == 5 and shape[1] == 1:
        pred = pred.squeeze(1)
        gt = gt.squeeze(1)
        shape = pred.shape
    if len(shape) != 4:
        raise ValueError("y_true and y_pred must have 4 dimensions.")
    x, y, z = shape[1:]
    if x != y or y != z:
        raise ValueError("x, y, and z must be equal.")

    pred = pred > 0
    gt = gt == 1

    return (pred == gt).float().mean()


def voxel_f1(pred, gt):
    """
    Computes the F1 score for y_true and y_pred.
    Args:
        y_true: ground truth voxels of shape (batch_size, x, y, z).
        y_pred: predicted voxels of shape (batch_size, x, y, z).
    Returns:
        The F1 score.
    """
    # check that the inputs are valid
    if pred.shape != gt.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    # squeeze if possible
    shape = pred.shape
    if len(shape) == 5 and shape[1] == 1:
        pred = pred.squeeze(1)
        gt = gt.squeeze(1)
        shape = pred.shape
    if len(shape) != 4:
        raise ValueError("y_true and y_pred must have 4 dimensions.")
    x, y, z = shape[1:]
    if x != y or y != z:
        raise ValueError("x, y, and z must be equal.")

    pred = pred > 0
    gt = gt == 1

    intersection = torch.sum(pred & gt, dim=(1, 2, 3))
    precision = intersection / torch.sum(pred, dim=(1, 2, 3))
    recall = intersection / torch.sum(gt, dim=(1, 2, 3))
    # add a small epsilon to avoid division by zero
    return torch.mean(2 * precision * recall / (precision + recall + 1e-6))
