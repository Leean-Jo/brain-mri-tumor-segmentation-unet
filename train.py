from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BrainMRIDataset, get_image_mask_paths
from src.losses import BCEDiceLoss
from src.metrics import dice_score, iou_score
from src.model import UNet




def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        running_loss += loss.item()
        running_dice += dice_score(outputs, masks)
        running_iou += iou_score(outputs, masks)

    return (
        running_loss / len(loader),
        running_dice / len(loader),
        running_iou / len(loader),
    )


def main():
    data_root = "data/lgg-mri-segmentation"

    lr = 1e-3
    batch_size = 8
    num_epochs = 1
    image_size = (128, 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    image_paths, mask_paths = get_image_mask_paths(data_root)
    print(f"Total samples: {len(image_paths)}")

    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths,
        mask_paths,
        test_size=0.2,
        random_state=42,
    )

    train_dataset = BrainMRIDataset(
        image_paths=train_images,
        mask_paths=train_masks,
        image_size=image_size,
        augment=True,
    )

    val_dataset = BrainMRIDataset(
        image_paths=val_images,
        mask_paths=val_masks,
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_dir = Path("outputs/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f} "
            f"val_iou={val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"Best model saved. Dice={best_dice:.4f}")


if __name__ == "__main__":
    main()