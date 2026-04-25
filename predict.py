from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import BrainMRIDataset, get_image_mask_paths
from src.model import UNet


def main():
    data_root = "data/lgg-mri-segmentation"
    image_size = (128, 128)
    batch_size = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths, mask_paths = get_image_mask_paths(data_root)

    _, val_images, _, val_masks = train_test_split(
        image_paths,
        mask_paths,
        test_size=0.2,
        random_state=42,
    )

    val_dataset = BrainMRIDataset(
        image_paths=val_images,
        mask_paths=val_masks,
        image_size=image_size,
        augment=False,
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(
        torch.load("outputs/epoch5/checkpoints/best_model.pth", map_location=device)
    )
    model.eval()

    output_dir = Path("outputs/epoch5/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = next(iter(val_loader))
    images = batch["image"].to(device)
    masks = batch["mask"]

    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float().cpu()

    images = images.cpu()

    for i in range(min(3, images.size(0))):
        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.title("MRI")
        plt.imshow(images[i].permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("GT")
        plt.imshow(masks[i].squeeze(), cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Pred")
        plt.imshow(preds[i].squeeze(), cmap="gray")
        plt.axis("off")

        plt.savefig(output_dir / f"pred_{i}.png")
        plt.close()

    print("Prediction images saved!")


if __name__ == "__main__":
    main()