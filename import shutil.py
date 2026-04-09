import shutil
from pathlib import Path

# 이 파일은 train.py와 같은 위치(프로젝트 루트)에 둬야 함.
# 예:
# brain-mri-tumor-segmentation-unet/
# ├── prepare_data.py
# ├── train.py
# └── data/
#     └── lgg-mri-segmentation/
#         └── kaggle_3m/
#             ├── TCGA_...

src_root = Path("data/lgg-mri-segmentation/kaggle_3m")
dst_images = Path("data/lgg-mri-segmentation/images")
dst_masks = Path("data/lgg-mri-segmentation/masks")

dst_images.mkdir(parents=True, exist_ok=True)
dst_masks.mkdir(parents=True, exist_ok=True)

if not src_root.exists():
    raise FileNotFoundError(f"Source folder not found: {src_root}")

image_count = 0
mask_count = 0

for file_path in src_root.rglob("*"):
    if not file_path.is_file():
        continue

    if file_path.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        continue

    if "mask" in file_path.stem.lower():
        shutil.copy2(file_path, dst_masks / file_path.name)
        mask_count += 1
    else:
        shutil.copy2(file_path, dst_images / file_path.name)
        image_count += 1

print(f"Copied images: {image_count}")
print(f"Copied masks : {mask_count}")
print("Done!")