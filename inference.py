from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.dataset import BrainMRIDataset, get_image_mask_paths
from src.model import UNet


def save_prediction(image, mask, pred_mask, save_path):
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred_mask = pred_mask.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    data_root = "data/lgg-mri-segmentation"
    checkpoint_path = "outputs/checkpoints/best_model.pth"
    save_dir = Path("outputs/predictions")
    save_dir.mkdir(parents=True, exist_ok=True)

    image_size = (128, 128)
    num_samples = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    image_paths, mask_paths = get_image_mask_paths(data_root)

    dataset = BrainMRIDataset(
        image_paths=image_paths[:num_samples],
        mask_paths=mask_paths[:num_samples],
        image_size=image_size,
        augment=False,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            save_path = save_dir / f"prediction_{idx}.png"

            save_prediction(
                images[0],
                masks[0],
                preds[0],
                save_path,
            )

            print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()