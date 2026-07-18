"""
Standalone, subprocess-isolated evaluation script.

Why this exists as a separate script instead of a function imported into the
notebook's long-lived kernel: this project's architecture (nets/model.py)
reads config.USE_DRM at import time to decide whether the P3 branch includes
the Detail Recovery Module. Within one notebook session we evaluate
checkpoints trained under different USE_DRM settings (the unchanged-baseline
control vs. the DRM+Wise-IoU+DWA proposed system) -- if both were evaluated
via `import nets.model` in the same Python process, only the FIRST import
would actually take effect (Python caches modules), silently evaluating the
wrong architecture for the second checkpoint. Running each evaluation as a
fresh `python eval_one.py ...` subprocess sidesteps that entirely: every
invocation re-reads config.py from disk, so it always builds the exact
architecture the checkpoint being evaluated was trained with.

Because of this, ALWAYS (re)write config.py with the correct USE_DRM value
for the checkpoint you are about to evaluate before calling this script.

Usage:
    python eval_one.py --dataname NAME --model_path PATH \
        --images_dir DIR --ann_dir DIR --image_ext .jpg \
        --out_json result.json \
        [--ids_file path_with_one_id_per_line] \
        [--classes_path model_data/rtts_classes.txt] \
        [--anchors_path model_data/yolo_anchors.txt]

If --ids_file is omitted, every file in --images_dir matching --image_ext is
used as the id list (id = filename without extension).
"""
import argparse
import json
import os
import re
import xml.etree.ElementTree as ET

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_map
from yolo import YOLO


def list_ids_from_dir(images_dir, ext):
    return sorted(os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith(ext))


def parse_per_class_ap(results_txt_path):
    """utils_map.get_map writes lines like '66.35% = bicycle AP ' into
    results.txt inside map_out_path -- pull those back out instead of
    modifying utils_map.py's already-working (and entangled) internals."""
    per_class = {}
    if not os.path.exists(results_txt_path):
        return per_class
    pattern = re.compile(r'^([\d.]+)% = (\S+) AP\b')
    with open(results_txt_path) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                per_class[m.group(2)] = float(m.group(1))
    return per_class


def evaluate(dataname, image_ids, images_dir, annotations_dir, model_path, image_ext,
             classes_path, anchors_path, min_overlap=0.5, confidence=0.001,
             nms_iou=0.5, score_threshold=0.5, map_out_path=None):
    map_out_path = map_out_path or f'map_out-{dataname}'
    for sub in ['ground-truth', 'detection-results', 'images-optional']:
        os.makedirs(os.path.join(map_out_path, sub), exist_ok=True)

    class_names, _ = get_classes(classes_path)
    print(f"[{dataname}] {len(image_ids)} images | loading model from {model_path}")
    yolo = YOLO(model_path=model_path, classes_path=classes_path, anchors_path=anchors_path,
                confidence=confidence, nms_iou=nms_iou, cuda=torch.cuda.is_available())

    skipped = []
    for image_id in tqdm(image_ids, desc=f'[{dataname}] predicting'):
        image_path = os.path.join(images_dir, image_id + image_ext)
        try:
            image = Image.open(image_path)
            image.load()
        except (UnidentifiedImageError, OSError) as e:
            skipped.append((image_id, str(e)))
            open(os.path.join(map_out_path, f"detection-results/{image_id}.txt"), "w").close()
            continue
        yolo.get_map_txt(image_id, image, class_names, map_out_path)

    if skipped:
        print(f"[{dataname}] WARNING: skipped {len(skipped)}/{len(image_ids)} unreadable image(s) "
              f"(treated as zero detections)")
        for image_id, err in skipped[:20]:
            print(f"    {image_id}{image_ext}: {err}")

    bad_xml = []
    for image_id in tqdm(image_ids, desc=f'[{dataname}] ground truth'):
        with open(os.path.join(map_out_path, f"ground-truth/{image_id}.txt"), "w") as new_f:
            try:
                root = ET.parse(os.path.join(annotations_dir, f"{image_id}.xml")).getroot()
            except ET.ParseError as e:
                bad_xml.append((image_id, str(e)))
                continue
            for obj in root.findall('object'):
                difficult_flag = obj.find('difficult') is not None and int(obj.find('difficult').text) == 1
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bnd = obj.find('bndbox')
                left, top = bnd.find('xmin').text, bnd.find('ymin').text
                right, bottom = bnd.find('xmax').text, bnd.find('ymax').text
                suffix = " difficult" if difficult_flag else ""
                new_f.write(f"{obj_name} {left} {top} {right} {bottom}{suffix}\n")

    if bad_xml:
        print(f"[{dataname}] WARNING: {len(bad_xml)} unparseable annotation(s) (treated as 0 objects)")

    print(f"[{dataname}] computing mAP")
    # NOTE: get_map returns mAP as a 0-1 fraction (e.g. 0.7839), not a
    # percentage -- despite the existing notebooks' evaluate() printing it
    # with a trailing '%' unscaled (a pre-existing formatting inconsistency
    # in this repo, not introduced here). We multiply by 100 explicitly below
    # so results in out_json are unambiguously percentages.
    mAP_fraction = get_map(min_overlap, True, score_threhold=score_threshold, path=map_out_path)
    per_class = parse_per_class_ap(os.path.join(map_out_path, 'results.txt'))
    return mAP_fraction * 100.0, per_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataname', required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--images_dir', required=True)
    p.add_argument('--ann_dir', required=True)
    p.add_argument('--image_ext', default='.jpg')
    p.add_argument('--ids_file', default=None,
                    help='optional file with one image id per line; defaults to listing images_dir')
    p.add_argument('--classes_path', default='model_data/rtts_classes.txt')
    p.add_argument('--anchors_path', default='model_data/yolo_anchors.txt')
    p.add_argument('--min_overlap', type=float, default=0.5)
    p.add_argument('--confidence', type=float, default=0.001)
    p.add_argument('--nms_iou', type=float, default=0.5)
    p.add_argument('--score_threshold', type=float, default=0.5)
    p.add_argument('--out_json', required=True)
    args = p.parse_args()

    if args.ids_file:
        # Matches rdfnet_baseline_eval.ipynb's read_test_ids() convention exactly
        # (whitespace-split, not line-split) for consistency with the already-
        # validated baseline evaluation.
        image_ids = open(args.ids_file).read().strip().split()
    else:
        image_ids = list_ids_from_dir(args.images_dir, args.image_ext)

    mAP, per_class = evaluate(
        dataname=args.dataname, image_ids=image_ids, images_dir=args.images_dir,
        annotations_dir=args.ann_dir, model_path=args.model_path, image_ext=args.image_ext,
        classes_path=args.classes_path, anchors_path=args.anchors_path,
        min_overlap=args.min_overlap, confidence=args.confidence,
        nms_iou=args.nms_iou, score_threshold=args.score_threshold,
        map_out_path=f'/kaggle/working/map_out-{args.dataname}',
    )

    result = {'dataname': args.dataname, 'model_path': args.model_path,
              'num_images': len(image_ids), 'mAP': mAP, 'per_class_AP': per_class}
    with open(args.out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n=== [{args.dataname}] mAP@{args.min_overlap}: {mAP:.2f}% ===")
    for cls, ap in sorted(per_class.items()):
        print(f"    {cls:12s}: {ap:.2f}%")
    print(f"Result written to {args.out_json}")


if __name__ == '__main__':
    main()
