import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Manual category grouping map
category_to_group = {
    '001': 'Jackets', '002': 'Jackets', '003': 'Tops', '004': 'Jackets',
    '005': 'Tops', '006': 'Sweaters', '007': 'Tops', '008': 'Tops',
    '009': 'Tops', '010': 'Athleisure', '011': 'Jackets', '012': 'Tops',
    '013': 'Jackets', '014': 'Jackets', '015': 'Outerwear', '016': 'Sweaters',
    '017': 'Tops', '018': 'Tops', '019': 'Tops', '020': 'Sweaters',
    '021': 'Pants', '022': 'Pants', '023': 'Skirts', '024': 'Shorts',
    '025': 'Pants', '026': 'Pants', '027': 'Pants', '028': 'Pants',
    '029': 'Athleisure', '030': 'Pants', '031': 'Skirts', '032': 'Shorts',
    '033': 'Skirts', '034': 'Athleisure', '035': 'Shorts', '036': 'Shorts',
    '037': 'Outerwear', '038': 'Outerwear', '039': 'Jackets', '040': 'Outerwear',
    '041': 'Dresses', '042': 'Jumpsuits', '043': 'Outerwear', '044': 'Outerwear',
    '045': 'Dresses', '046': 'Jumpsuits', '047': 'Outerwear', '048': 'Dresses',
    '049': 'Dresses', '050': 'Dresses'
}

def load_eval_partition(filepath):
    split_map = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            img_path, split = line.strip().split()
            split_map[img_path] = split
    return split_map

def load_category_labels(filepath):
    category_map = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            img_path, category = line.strip().split()
            category_map[img_path] = category.zfill(3)  # ensure zero-padding
    return category_map

def create_grouped_symlink_dataset(images_dir, output_dir, eval_partition_file, category_file):
    split_map = load_eval_partition(eval_partition_file)
    category_map = load_category_labels(category_file)

    missing_count = 0
    uncategorized_count = 0

    print(f"Creating grouped symlink dataset into {output_dir}...")
    for img_rel_path in tqdm(split_map.keys(), desc="Processing images"):
        split = split_map[img_rel_path]
        img_path = images_dir / img_rel_path
        if not img_path.exists():
            missing_count += 1
            continue

        category_id = category_map.get(img_rel_path, None)
        if category_id is None or category_id not in category_to_group:
            uncategorized_count += 1
            continue

        group_label = category_to_group[category_id]
        target_dir = output_dir / split / group_label
        target_dir.mkdir(parents=True, exist_ok=True)

        target_symlink = target_dir / img_path.name
        if not target_symlink.exists():
            os.symlink(img_path.resolve(), target_symlink)

    print("Grouped symlink dataset created successfully!")
    print(f"Total missing images: {missing_count}")
    print(f"Total uncategorized images: {uncategorized_count}")

if __name__ == "__main__":
    base_dir = Path("data/deepfashion")
    images_dir = base_dir
    eval_partition_file = base_dir / "Eval" / "list_eval_partition.txt"
    category_file = base_dir / "Anno_coarse" / "list_category_img.txt"
    output_dir = Path("data/fashion_grouped")

    create_grouped_symlink_dataset(images_dir, output_dir, eval_partition_file, category_file)
