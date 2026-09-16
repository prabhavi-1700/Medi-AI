import os
import random
import shutil

SOURCE_DIR = "curated_dataset"
OUTPUT_DIR = "split_dataset"

CLASSES = [
    "Acne",
    "Eczema",
    "Healthy Skin",
    "Melanoma",
    "vitiligo"
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

random.seed(42)

# Create output folders
for split in ["train", "validation", "test"]:
    for class_name in CLASSES:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )

for class_name in CLASSES:

    source_class = os.path.join(SOURCE_DIR, class_name)

    image_files = [
        file for file in os.listdir(source_class)
        if file.lower().endswith(VALID_EXTENSIONS)
    ]

    random.shuffle(image_files)

    total = len(image_files)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    splits = {
        "train": train_files,
        "validation": val_files,
        "test": test_files
    }

    for split_name, files in splits.items():

        destination_class = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_name
        )

        for file in files:

            source_path = os.path.join(source_class, file)
            destination_path = os.path.join(
                destination_class,
                file
            )

            shutil.copy2(source_path, destination_path)

    print(
        f"{class_name}: "
        f"Train={len(train_files)}, "
        f"Validation={len(val_files)}, "
        f"Test={len(test_files)}"
    )

print("\n===================================")
print("DATASET SPLIT COMPLETE")
print("===================================")