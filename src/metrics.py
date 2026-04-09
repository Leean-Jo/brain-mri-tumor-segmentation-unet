import torch


def dice_score(logits, targets, threshold: float = 0.5, smooth: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    score = (2.0 * intersection + smooth) / (
        preds.sum(dim=1) + targets.sum(dim=1) + smooth
    )
    return score.mean().item()


def iou_score(logits, targets, threshold: float = 0.5, smooth: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    score = (intersection + smooth) / (union + smooth)
    return score.mean().item()