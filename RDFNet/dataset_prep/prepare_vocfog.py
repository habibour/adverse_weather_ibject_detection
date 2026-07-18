"""
Builds the VOC-FOG dataset (train + test/val) expected by RDFNet's train.py /
get_map.py, using the exact image-ID splits already committed in this repo at
dataset_split/train.txt (9,578 ids) and dataset_split/test.txt (2,129 ids), and
the ASM fog synthesis in fog_synthesis.py.

Why reuse dataset_split/*.txt instead of re-deriving the split: those files
already encode which VOC2007-2012 images were selected (only images containing
>=1 of the 5 target classes) and the exact train/test partition RDFNet reports
78.39 mAP on -- redoing that selection ourselves risks a different, non-comparable
split.

Prerequisite: download and extract, then merge into flat roots that each
contain Annotations/ and JPEGImages/:
    - PASCAL VOC2007 trainval:  http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    - PASCAL VOC2007 test:      http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    - PASCAL VOC2012 trainval:  http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  (mirror if the host is down: replace host.robots.ox.ac.uk with pjreddie.com/media/files,
  or search Kaggle for "PASCAL VOC 2007" / "PASCAL VOC 2012")
  VOC image filenames are unique across 2007-2012 (they're year-prefixed), so the
  three extracted VOCdevkit/VOC{2007,2007,2012}/ folders can be passed directly
  as --sources without merging them by hand.

Usage:
    python prepare_vocfog.py \\
        --sources VOCdevkit_trainval2007/VOC2007 VOCdevkit_test2007/VOC2007 VOCdevkit_trainval2012/VOC2012 \\
        --out /path/to/VOCFOG_out
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
SPLIT_DIR = os.path.join(REPO_ROOT, "dataset_split")


def get_classes():
    with open(CLASSES_PATH) as f:
        return [c.strip() for c in f if c.strip()]


def find_source(image_id, source_roots, exts=(".jpg", ".jpeg", ".png")):
    for root in source_roots:
        for ext in exts:
            jpg = os.path.join(root, "JPEGImages", image_id + ext)
            xml = os.path.join(root, "Annotations", image_id + ".xml")
            if os.path.exists(jpg) and os.path.exists(xml):
                return jpg, xml
    return None, None


def write_box_line(xml_path, classes, list_file):
    root = ET.parse(xml_path).getroot()
    for obj in root.iter("object"):
        difficult_node = obj.find("difficult")
        difficult = int(difficult_node.text) if difficult_node is not None else 0
        cls = obj.find("name").text
        if cls not in classes or difficult == 1:
            continue
        b = obj.find("bndbox")
        coords = (
            int(float(b.find("xmin").text)),
            int(float(b.find("ymin").text)),
            int(float(b.find("xmax").text)),
            int(float(b.find("ymax").text)),
        )
        list_file.write(" " + ",".join(map(str, coords)) + "," + str(classes.index(cls)))


def build_split(split_name, ids, source_roots, out_dir, classes, seed):
    rng = random.Random(seed)
    jpeg_dir = os.path.join(out_dir, "VOC2007", "JPEGImages")
    fog_dir = os.path.join(out_dir, "VOC2007", "FOG")
    ann_dir = os.path.join(out_dir, "VOC2007", "Annotations")
    for d in (jpeg_dir, fog_dir, ann_dir):
        os.makedirs(d, exist_ok=True)

    clean_list = open(os.path.join(out_dir, f"clean_{split_name}.txt"), "w")
    fog_list = open(os.path.join(out_dir, f"fog_{split_name}.txt"), "w")

    missing = []
    for image_id in ids:
        src_jpg, src_xml = find_source(image_id, source_roots)
        if src_jpg is None:
            missing.append(image_id)
            continue

        dst_jpg = os.path.join(jpeg_dir, image_id + ".jpg")
        dst_xml = os.path.join(ann_dir, image_id + ".xml")
        dst_fog = os.path.join(fog_dir, image_id + ".jpg")

        img = cv2.imread(dst_jpg) if os.path.exists(dst_jpg) else cv2.imread(src_jpg)
        if not os.path.exists(dst_jpg):
            cv2.imwrite(dst_jpg, img)
        if not os.path.exists(dst_xml):
            shutil.copy(src_xml, dst_xml)
        if not os.path.exists(dst_fog):
            foggy, _beta = add_fog(img, rng=rng)
            cv2.imwrite(dst_fog, foggy)

        clean_list.write(os.path.abspath(dst_jpg))
        write_box_line(dst_xml, classes, clean_list)
        clean_list.write("\n")

        fog_list.write(os.path.abspath(dst_fog))
        write_box_line(dst_xml, classes, fog_list)
        fog_list.write("\n")

    clean_list.close()
    fog_list.close()
    if missing:
        print(f"[{split_name}] WARNING: {len(missing)} ids not found under --sources, e.g. {missing[:5]}")
    print(f"[{split_name}] wrote {len(ids) - len(missing)} / {len(ids)} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="One or more raw VOC roots, each containing Annotations/ and JPEGImages/",
    )
    ap.add_argument("--out", required=True, help="Output VOC-FOG root")
    ap.add_argument("--seed", type=int, default=114514)
    args = ap.parse_args()

    classes = get_classes()
    train_ids = open(os.path.join(SPLIT_DIR, "train.txt")).read().split()
    test_ids = open(os.path.join(SPLIT_DIR, "test.txt")).read().split()

    sets_dir = os.path.join(args.out, "VOC2007", "ImageSets", "Main")
    os.makedirs(sets_dir, exist_ok=True)
    shutil.copy(os.path.join(SPLIT_DIR, "train.txt"), os.path.join(sets_dir, "train.txt"))
    shutil.copy(os.path.join(SPLIT_DIR, "test.txt"), os.path.join(sets_dir, "test.txt"))

    build_split("train", train_ids, args.sources, args.out, classes, args.seed)
    build_split("val", test_ids, args.sources, args.out, classes, args.seed + 1)

    # Rename to the exact filenames config.py expects
    shutil.move(os.path.join(args.out, "clean_train.txt"), os.path.join(args.out, "2007_train.txt"))
    shutil.move(os.path.join(args.out, "fog_train.txt"), os.path.join(args.out, "2007_train_fog.txt"))
    shutil.move(os.path.join(args.out, "fog_val.txt"), os.path.join(args.out, "2007_val_fog.txt"))
    os.remove(os.path.join(args.out, "clean_val.txt"))  # only the train-side clean pairs are used

    print("\nDone. In config.py set:")
    print(f"  train_annotation_path = '{os.path.join(args.out, '2007_train_fog.txt')}'")
    print(f"  val_annotation_path   = '{os.path.join(args.out, '2007_val_fog.txt')}'")
    print(f"  clear_annotation_path = '{os.path.join(args.out, '2007_train.txt')}'")
    print("For get_map.py testing, set dataname='VOC-FOG' and VOCdevkit_path to this --out path.")


if __name__ == "__main__":
    main()
