"""
Converts the Foggy Driving Dataset (Sakaridis, Dai & Van Gool, IJCV 2018) into
VOC-XML annotations restricted to RDFNet's 5 classes (person, bicycle, car,
motorbike, bus), and lays it out the way get_map.py expects.

Download first (100 MB, direct link, no login required):
    https://data.vision.ee.ethz.ch/csakarid/shared/SFSU_synthetic/Downloads/Foggy_Driving.zip
Extract it, then:

    # 1) sanity-check the annotation format before converting anything --
    #    Foggy Driving ships Cityscapes-style *_polygons.json files; this
    #    script assumes that format and derives bounding boxes from the
    #    polygon extents. If your extracted copy looks different, this will
    #    tell you so instead of silently producing wrong boxes.
    python prepare_fdd.py --inspect --source /path/to/Foggy_Driving

    # 2) convert
    python prepare_fdd.py --source /path/to/Foggy_Driving --out /path/to/FDD_out

Cityscapes' 8 "thing" classes are person, rider, car, truck, bus, train,
motorcycle, bicycle -- FDD's bbox annotations cover exactly these 8 (per the
dataset's own description: "individual instances of the 8 classes ... afford
bounding box annotations"). We keep the 5 that overlap with RDFNet's class set
and map motorcycle -> motorbike; rider/truck/train are dropped.

FDD has no official train/test split file (it's a 101-image, eval-only
benchmark) and isn't listed in this repo's dataset_split/, so every annotated
image found becomes the test set.
"""
import argparse
import glob
import json
import os
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

CLASS_MAP = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorbike",
    "bus": "bus",
}


def find_polygon_jsons(source):
    return sorted(glob.glob(os.path.join(source, "**", "*polygons.json"), recursive=True))


def find_image_for(json_path, source):
    base = os.path.basename(json_path)
    stem = base
    for suffix in ("_gtFine_polygons.json", "_gtCoarse_polygons.json", "_polygons.json"):
        if base.endswith(suffix):
            stem = base[: -len(suffix)]
            break
    else:
        stem = os.path.splitext(base)[0]

    for img_suffix in ("_leftImg8bit", ""):
        for ext in (".png", ".jpg", ".jpeg"):
            candidates = glob.glob(os.path.join(source, "**", stem + img_suffix + ext), recursive=True)
            if candidates:
                return candidates[0], stem
    return None, stem


def polygon_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def write_voc_xml(out_path, filename, width, height, boxes):
    ann = ET.Element("annotation")
    ET.SubElement(ann, "filename").text = filename
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    for cls, (xmin, ymin, xmax, ymax) in boxes:
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = cls
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(xmin))
        ET.SubElement(bnd, "ymin").text = str(int(ymin))
        ET.SubElement(bnd, "xmax").text = str(int(xmax))
        ET.SubElement(bnd, "ymax").text = str(int(ymax))
    xml_str = minidom.parseString(ET.tostring(ann)).toprettyxml(indent="  ")
    with open(out_path, "w") as f:
        f.write(xml_str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    jsons = find_polygon_jsons(args.source)
    print(f"Found {len(jsons)} *polygons.json files under {args.source}")
    if not jsons:
        print("No Cityscapes-style polygon jsons found. Other json/xml/txt files present:")
        for ext in ("*.json", "*.xml", "*.txt"):
            for p in glob.glob(os.path.join(args.source, "**", ext), recursive=True)[:20]:
                print(" ", p)
        print("-> the format assumption in this script doesn't match your download; "
              "inspect one of the files above and adjust find_polygon_jsons / the parser.")
        return

    if args.inspect:
        sample = json.load(open(jsons[0]))
        print("Sample file:", jsons[0])
        print("Top-level keys:", list(sample.keys()))
        if sample.get("objects"):
            print("First object keys:", list(sample["objects"][0].keys()))
            print("First object:", sample["objects"][0])
        img, stem = find_image_for(jsons[0], args.source)
        print("Matched image:", img, "(stem:", stem, ")")
        if img is None:
            print("-> could not locate the matching image file; check the naming pattern.")
        return

    assert args.out, "--out is required unless --inspect"
    import cv2  # deferred import so --inspect works without opencv installed

    jpeg_dir = os.path.join(args.out, "VOC2007", "JPEGImages")
    ann_dir = os.path.join(args.out, "VOC2007", "Annotations")
    sets_dir = os.path.join(args.out, "VOC2007", "ImageSets", "Main")
    for d in (jpeg_dir, ann_dir, sets_dir):
        os.makedirs(d, exist_ok=True)

    ids = []
    skipped_no_image = 0
    skipped_no_boxes = 0
    for jp in jsons:
        img_path, stem = find_image_for(jp, args.source)
        if img_path is None:
            skipped_no_image += 1
            continue

        data = json.load(open(jp))
        h, w = data.get("imgHeight"), data.get("imgWidth")
        boxes = []
        for obj in data.get("objects", []):
            label = obj.get("label")
            if label not in CLASS_MAP:
                continue
            poly = obj.get("polygon", [])
            if len(poly) < 2:
                continue
            boxes.append((CLASS_MAP[label], polygon_to_bbox(poly)))
        if not boxes:
            skipped_no_boxes += 1
            continue  # keep FDD restricted to images with >=1 of the 5 target classes

        img = cv2.imread(img_path)
        if h is None or w is None:
            h, w = img.shape[:2]
        cv2.imwrite(os.path.join(jpeg_dir, stem + ".jpg"), img)
        write_voc_xml(os.path.join(ann_dir, stem + ".xml"), stem + ".jpg", w, h, boxes)
        ids.append(stem)

    with open(os.path.join(sets_dir, "test.txt"), "w") as f:
        f.write("\n".join(ids) + "\n")

    print(f"Wrote {len(ids)} annotated images "
          f"(skipped {skipped_no_image} with no matched image, {skipped_no_boxes} with 0 relevant boxes)")
    print(f'For get_map.py: dataname = "fdd", VOCdevkit_path = "{args.out}"')


if __name__ == "__main__":
    main()
