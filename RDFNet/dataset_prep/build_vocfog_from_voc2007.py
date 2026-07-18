"""
Builds VOC-FOG for RDFNet using ONLY the local, already-downloaded VOC2007
trainval + test releases under datasets/voc/ (no VOC2012 download needed).

This departs from prepare_vocfog.py (which replays the repo's committed
dataset_split/{train,test}.txt -- IDs that only exist in the cumulative
VOC2012 "trainval" devkit, e.g. "2008_000129") in favour of the official
VOC2007 challenge train/test partition:
    - VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007  (5011 images) -> train pool
    - VOCtest_06-Nov-2007/VOCdevkit/VOC2007      (4952 images) -> test pool
Both pools are filtered down to images containing >=1 non-difficult object
in RDFNet's 5 classes (person, bicycle, car, motorbike, bus). This is a
smaller, non-paper-matching split, but it needs nothing beyond what's
already on disk and has no train/test leakage (official VOC partition).

Fog synthesis follows the same ASM the paper cites (McCartney 1977, via
IA-YOLO's data_make.py -- see fog_synthesis.py), but widens the augmentation
to close the synthetic->real domain gap that shows up as the big accuracy
drop from VOC-FOG (clean synthetic fog) to RTTS/FDD (real fog) in Table I of
the paper (78.39 -> 59.93 -> 36.99 mAP):

  - TRAIN: each clean image gets FOG_VARIANTS_TRAIN foggy realizations, beta
    ~ Uniform(*TRAIN_BETA_RANGE) and A ~ Uniform(*TRAIN_A_RANGE) sampled
    independently per realization. This is closer to IA-YOLO's original
    multi-level fog augmentation (10 fixed beta levels per image) than a
    single fixed-beta copy, and the wider beta/A ranges expose the detector
    to both light haze and dense fog with varying atmospheric brightness,
    instead of one narrow, always-gray-ish fog look.
  - TEST/VAL: exactly one foggy realization per image (paper's convention
    for a reproducible held-out evaluation set), beta ~ Uniform(*TEST_BETA_RANGE).

The clean and fog train lists are written index-aligned (clean image line i
pairs with fog image line i) because utils/dataloader.py's YoloDataset zips
annotation_lines[index] with clean_lines[index] -- so each of the
FOG_VARIANTS_TRAIN foggy copies of an image gets its own repeated clean line.

Usage:
    cd dataset_prep
    python build_vocfog_from_voc2007.py --out ../datasets/VOC_FOG
"""
import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET

import cv2

from fog_synthesis import add_fog

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES_PATH = os.path.join(REPO_ROOT, "model_data", "rtts_classes.txt")
VOC_TRAINVAL_ROOT = os.path.join(
    REPO_ROOT, "datasets", "voc", "VOCtrainval_06-Nov-2007", "VOCdevkit", "VOC2007"
)
VOC_TEST_ROOT = os.path.join(
    REPO_ROOT, "datasets", "voc", "VOCtest_06-Nov-2007", "VOCdevkit", "VOC2007"
)

FOG_VARIANTS_TRAIN = 3
TRAIN_BETA_RANGE = (0.03, 0.18)
TRAIN_A_RANGE = (0.45, 0.65)
TEST_BETA_RANGE = (0.05, 0.14)
TEST_A = 0.5


def get_classes():
    with open(CLASSES_PATH) as f:
        return [c.strip() for c in f if c.strip()]


def image_boxes(xml_path, classes):
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.iter("object"):
        difficult_node = obj.find("difficult")
        difficult = int(difficult_node.text) if difficult_node is not None else 0
        cls = obj.find("name").text
        if cls not in classes or difficult == 1:
            continue
        b = obj.find("bndbox")
        boxes.append(
            (
                int(float(b.find("xmin").text)),
                int(float(b.find("ymin").text)),
                int(float(b.find("xmax").text)),
                int(float(b.find("ymax").text)),
                classes.index(cls),
            )
        )
    return boxes


def box_suffix(boxes):
    return "".join(" " + ",".join(map(str, b)) for b in boxes)


def collect_ids(voc_root, classes):
    ann_dir = os.path.join(voc_root, "Annotations")
    kept = []
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".xml"):
            continue
        image_id = fname[:-4]
        boxes = image_boxes(os.path.join(ann_dir, fname), classes)
        if boxes:
            kept.append(image_id)
    return kept


def build_train(ids, out_dir, classes, seed):
    rng = random.Random(seed)
    jpeg_dir = os.path.join(out_dir, "VOC2007", "JPEGImages")
    fog_dir = os.path.join(out_dir, "VOC2007", "FOG")
    ann_dir = os.path.join(out_dir, "VOC2007", "Annotations")
    for d in (jpeg_dir, fog_dir, ann_dir):
        os.makedirs(d, exist_ok=True)

    clean_list = open(os.path.join(out_dir, "2007_train.txt"), "w")
    fog_list = open(os.path.join(out_dir, "2007_train_fog.txt"), "w")

    for image_id in ids:
        src_jpg = os.path.join(VOC_TRAINVAL_ROOT, "JPEGImages", image_id + ".jpg")
        src_xml = os.path.join(VOC_TRAINVAL_ROOT, "Annotations", image_id + ".xml")
        dst_jpg = os.path.join(jpeg_dir, image_id + ".jpg")
        dst_xml = os.path.join(ann_dir, image_id + ".xml")

        if not os.path.exists(dst_jpg):
            shutil.copy(src_jpg, dst_jpg)
        if not os.path.exists(dst_xml):
            shutil.copy(src_xml, dst_xml)

        boxes = image_boxes(dst_xml, classes)
        suffix = box_suffix(boxes)
        img = cv2.imread(dst_jpg)

        for v in range(FOG_VARIANTS_TRAIN):
            fog_id = f"{image_id}_f{v}"
            dst_fog = os.path.join(fog_dir, fog_id + ".jpg")
            if not os.path.exists(dst_fog):
                beta = rng.uniform(*TRAIN_BETA_RANGE)
                a = rng.uniform(*TRAIN_A_RANGE)
                foggy, _ = add_fog(img, beta=beta, A=a, rng=rng)
                cv2.imwrite(dst_fog, foggy)

            clean_list.write(os.path.abspath(dst_jpg) + suffix + "\n")
            fog_list.write(os.path.abspath(dst_fog) + suffix + "\n")

    clean_list.close()
    fog_list.close()
    print(f"[train] {len(ids)} source images -> {len(ids) * FOG_VARIANTS_TRAIN} fog/clean pairs "
          f"({FOG_VARIANTS_TRAIN} fog variants each)")


def build_test(ids, out_dir, classes, seed):
    rng = random.Random(seed)
    fog_dir = os.path.join(out_dir, "VOC2007", "FOG")
    ann_dir = os.path.join(out_dir, "VOC2007", "Annotations")
    os.makedirs(fog_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    fog_list = open(os.path.join(out_dir, "2007_val_fog.txt"), "w")

    for image_id in ids:
        src_jpg = os.path.join(VOC_TEST_ROOT, "JPEGImages", image_id + ".jpg")
        src_xml = os.path.join(VOC_TEST_ROOT, "Annotations", image_id + ".xml")
        dst_xml = os.path.join(ann_dir, image_id + ".xml")
        dst_fog = os.path.join(fog_dir, image_id + ".jpg")

        if not os.path.exists(dst_xml):
            shutil.copy(src_xml, dst_xml)

        boxes = image_boxes(dst_xml, classes)
        suffix = box_suffix(boxes)

        if not os.path.exists(dst_fog):
            img = cv2.imread(src_jpg)
            beta = rng.uniform(*TEST_BETA_RANGE)
            foggy, _ = add_fog(img, beta=beta, A=TEST_A, rng=rng)
            cv2.imwrite(dst_fog, foggy)

        fog_list.write(os.path.abspath(dst_fog) + suffix + "\n")

    fog_list.close()
    print(f"[test] {len(ids)} source images -> {len(ids)} fog images (1 variant each)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output VOC-FOG root")
    ap.add_argument("--seed", type=int, default=114514)
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)

    classes = get_classes()
    train_ids = collect_ids(VOC_TRAINVAL_ROOT, classes)
    test_ids = collect_ids(VOC_TEST_ROOT, classes)

    sets_dir = os.path.join(out_dir, "VOC2007", "ImageSets", "Main")
    os.makedirs(sets_dir, exist_ok=True)
    with open(os.path.join(sets_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_ids) + "\n")
    with open(os.path.join(sets_dir, "test.txt"), "w") as f:
        f.write("\n".join(test_ids) + "\n")

    build_train(train_ids, out_dir, classes, args.seed)
    build_test(test_ids, out_dir, classes, args.seed + 1)

    print("\nDone. In config.py set:")
    print(f"  train_annotation_path = '{os.path.join(out_dir, '2007_train_fog.txt')}'")
    print(f"  val_annotation_path   = '{os.path.join(out_dir, '2007_val_fog.txt')}'")
    print(f"  clear_annotation_path = '{os.path.join(out_dir, '2007_train.txt')}'")
    print(f"For get_map.py-style testing, VOCdevkit_path = '{out_dir}', dataname = 'VOC-FOG'.")


if __name__ == "__main__":
    main()
