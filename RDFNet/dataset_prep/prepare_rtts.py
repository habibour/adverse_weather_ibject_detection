"""
Reorganizes the RESIDE RTTS release ("HazeDetection") into the VOC2007-style
layout RDFNet's get_map.py expects.

RTTS ships as (per the RESIDE-beta page, sites.google.com/view/reside-dehaze-datasets/reside-b):
    HazeDetection/
        Annotations/*.xml
        ImageSets/Main/test.txt
        JPEGImages/*.png

Two fixes are needed versus using it as-is:
  1. get_map.py hardcodes the ".jpg" extension for the rtts/fdd branch, but RTTS
     images are .png -- this script re-encodes them to .jpg.
  2. RDFNet's own dataset_split/rtts_test.txt (4,321 ids) is used as the
     authoritative test list here instead of RESIDE's shipped test.txt, in case
     they differ slightly from what the paper's numbers were computed on.

Download RTTS first:
    Dropbox: https://bit.ly/3c4gl3z
    Baidu:   https://pan.baidu.com/s/1A0MMAnlWmuJ0dXhsbXk4Gg  (password: 4mv7)

Usage:
    python prepare_rtts.py --source /path/to/HazeDetection --out /path/to/RTTS_out
"""
import argparse
import os
import shutil

import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_FILE = os.path.join(REPO_ROOT, "dataset_split", "rtts_test.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to extracted HazeDetection/ folder")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids = open(SPLIT_FILE).read().split()
    jpeg_dir = os.path.join(args.out, "VOC2007", "JPEGImages")
    ann_dir = os.path.join(args.out, "VOC2007", "Annotations")
    sets_dir = os.path.join(args.out, "VOC2007", "ImageSets", "Main")
    for d in (jpeg_dir, ann_dir, sets_dir):
        os.makedirs(d, exist_ok=True)

    missing = []
    for image_id in ids:
        src_png = os.path.join(args.source, "JPEGImages", image_id + ".png")
        src_xml = os.path.join(args.source, "Annotations", image_id + ".xml")
        if not (os.path.exists(src_png) and os.path.exists(src_xml)):
            missing.append(image_id)
            continue

        dst_jpg = os.path.join(jpeg_dir, image_id + ".jpg")
        if not os.path.exists(dst_jpg):
            img = cv2.imread(src_png)
            cv2.imwrite(dst_jpg, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        dst_xml = os.path.join(ann_dir, image_id + ".xml")
        if not os.path.exists(dst_xml):
            shutil.copy(src_xml, dst_xml)

    kept = [i for i in ids if i not in missing]
    with open(os.path.join(sets_dir, "test.txt"), "w") as f:
        f.write("\n".join(kept) + "\n")

    if missing:
        print(f"WARNING: {len(missing)} ids from dataset_split/rtts_test.txt not found under --source, e.g. {missing[:5]}")
    print(f"Wrote {len(kept)} / {len(ids)} images to {args.out}")
    print(f'For get_map.py: dataname = "rtts", VOCdevkit_path = "{args.out}"')


if __name__ == "__main__":
    main()
