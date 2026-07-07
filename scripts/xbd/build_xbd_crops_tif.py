import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from shapely import wkt
from tqdm import tqdm

RAW = Path("datasets/xbd_raw")
OUT = Path("datasets/xbd_processed")
CROP_DIR = OUT / "crops"
CROP_DIR.mkdir(parents=True, exist_ok=True)

CONTINENT_MAP = {
    "guatemala-volcano": "central_america",
    "mexico-earthquake": "north_america",
    "hurricane-florence": "north_america",
    "hurricane-harvey": "north_america",
    "hurricane-matthew": "north_america",
    "hurricane-michael": "north_america",
    "midwest-flooding": "north_america",
    "santa-rosa-wildfire": "north_america",
    "socal-fire": "north_america",
    "palu-tsunami": "asia",
}

def read_tif_as_pil(path):
    arr = tifffile.imread(str(path))

    # Handle possible shapes
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        # If channel-first, convert to channel-last
        if arr.shape[0] in [3, 4] and arr.shape[-1] not in [3, 4]:
            arr = np.transpose(arr, (1, 2, 0))
        # Keep first 3 channels
        if arr.shape[-1] > 3:
            arr = arr[..., :3]

    # Convert to uint8 safely
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        arr = np.nan_to_num(arr).clip(0, 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")

def damage_to_binary(label):
    return 0 if label in ["no-damage", "un-classified", None, ""] else 1

def damage_to_four(label):
    return {
        "no-damage": 0,
        "minor-damage": 1,
        "major-damage": 2,
        "destroyed": 3,
        "un-classified": 0,
    }.get(label, 0)

def safe_crop(img, bounds, pad=4):
    w, h = img.size
    minx, miny, maxx, maxy = bounds
    x1 = max(int(minx) - pad, 0)
    y1 = max(int(miny) - pad, 0)
    x2 = min(int(maxx) + pad, w)
    y2 = min(int(maxy) + pad, h)

    if x2 <= x1 or y2 <= y1:
        return None

    return img.crop((x1, y1, x2, y2)).resize((128, 128))

rows = []
post_labels = sorted(RAW.rglob("*_post_disaster.json"))
print("Found post label files:", len(post_labels))

bad_images = 0

for post_label in tqdm(post_labels):
    with open(post_label, "r") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    disaster = meta.get("disaster") or post_label.name.split("_")[0]
    disaster_type = meta.get("disaster_type", "unknown")
    continent = CONTINENT_MAP.get(disaster, "unknown")

    post_img_name = post_label.name.replace(".json", ".tif")
    pre_img_name = post_label.name.replace("_post_disaster.json", "_pre_disaster.tif")

    post_imgs = list(RAW.rglob(post_img_name))
    pre_imgs = list(RAW.rglob(pre_img_name))

    if not post_imgs or not pre_imgs:
        continue

    try:
        post_img = read_tif_as_pil(post_imgs[0])
        pre_img = read_tif_as_pil(pre_imgs[0])
    except Exception as e:
        bad_images += 1
        print("Bad tif:", post_imgs[0], e)
        continue

    image_id = post_img_name.replace("_post_disaster.tif", "")
    buildings = data.get("features", {}).get("xy", [])

    for idx, feat in enumerate(buildings):
        subtype = feat.get("properties", {}).get("subtype", "no-damage")
        poly_wkt = feat.get("wkt")
        if not poly_wkt:
            continue

        try:
            geom = wkt.loads(poly_wkt)
            bounds = geom.bounds
        except Exception:
            continue

        pre_crop = safe_crop(pre_img, bounds)
        post_crop = safe_crop(post_img, bounds)

        if pre_crop is None or post_crop is None:
            continue

        sample_id = hashlib.md5(f"{image_id}_{idx}_{subtype}".encode()).hexdigest()[:16]

        pre_crop_path = CROP_DIR / f"{sample_id}_pre.png"
        post_crop_path = CROP_DIR / f"{sample_id}_post.png"

        pre_crop.save(pre_crop_path)
        post_crop.save(post_crop_path)

        shard = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % 3
        client_id = f"{continent}_{disaster_type}_{disaster}_shard{shard}"

        rows.append({
            "sample_id": sample_id,
            "image_id": image_id,
            "continent": continent,
            "disaster_type": disaster_type,
            "disaster": disaster,
            "client_id": client_id,
            "pre_crop_path": str(pre_crop_path),
            "post_crop_path": str(post_crop_path),
            "damage_label_raw": subtype,
            "label_binary": damage_to_binary(subtype),
            "label_four": damage_to_four(subtype),
        })

df = pd.DataFrame(rows)
df.to_csv(OUT / "xbd_crops_metadata.csv", index=False)

print("Saved:", OUT / "xbd_crops_metadata.csv")
print("Samples:", len(df))
print("Bad images:", bad_images)

if len(df):
    print("\nBinary labels:")
    print(df["label_binary"].value_counts())
    print("\nFour labels:")
    print(df["label_four"].value_counts())
    print("\nClients:")
    print(df["client_id"].value_counts().head(50))
