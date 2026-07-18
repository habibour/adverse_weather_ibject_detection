# Preparing VOC-FOG, RTTS, and FDD for RDFNet

RDFNet is trained once on VOC-FOG (5 classes: person, bicycle, car, motorbike,
bus) and evaluated zero-shot on all three benchmarks. Only VOC-FOG needs a
train split; RTTS and FDD are evaluation-only.

## 1. What to download

| Dataset | Source | Notes |
|---|---|---|
| PASCAL VOC2007 trainval | `http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar` | raw material for VOC-FOG (clean images to fog-ify) |
| PASCAL VOC2007 test | `http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar` | same |
| PASCAL VOC2012 trainval | `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar` | same |
| RTTS | Dropbox `https://bit.ly/3c4gl3z` or Baidu `pan.baidu.com/s/1A0MMAnlWmuJ0dXhsbXk4Gg` (pw `4mv7`), from the [RESIDE-β page](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2) | already VOC2007-formatted, images are `.png` |
| FDD (Foggy Driving) | `https://data.vision.ee.ethz.ch/csakarid/shared/SFSU_synthetic/Downloads/Foggy_Driving.zip` (100 MB, direct, no login) | Cityscapes-style polygon jsons, not VOC XML |

If `host.robots.ox.ac.uk` is unreachable (it goes down often), the same tars
are commonly mirrored at `pjreddie.com/media/files/...` with identical
filenames, or search Kaggle for "PASCAL VOC 2007" / "PASCAL VOC 2012".

You already have these three on Kaggle (`fdd-dataset`, `voc-fog`, `rtts-dataset`)
plus the checkpoint (`rdfnet-baseline`) — the scripts below tell you how to
check whether what's inside matches the layout RDFNet's code expects, and how
to fix it if not. Run `--inspect`/dry-run steps first rather than assuming.

## 2. VOC-FOG (train + test)

VOC-FOG is *synthetic* — there's no single file to download for it. It's built
by taking a subset of VOC images and adding fog with an atmospheric-scattering
model, following IA-YOLO's method (the paper cites this explicitly).

There are two ways to build it, depending on what raw VOC data you have:

**(a) `prepare_vocfog.py`** — replays this repo's committed
`dataset_split/train.txt` (9,578 ids) / `test.txt` (2,129 ids), the exact
split the paper's 78.39 mAP was computed on. **Requires VOC2012's "trainval"
devkit tar** (`VOCtrainval_11-May-2012.tar`), not just VOC2007 — those IDs are
in the form `2008_000129`, `2011_003456`, etc., which only exist in the
cumulative 2007-2012 devkit, not the standalone VOC2007 release.

```bash
cd dataset_prep
python prepare_vocfog.py \
    --sources /path/to/VOCdevkit/VOC2007 /path/to/VOCdevkit_test/VOC2007 /path/to/VOCdevkit2012/VOC2012 \
    --out /data/VOCFOG
```

**(b) `build_vocfog_from_voc2007.py`** — uses only the standalone VOC2007
trainval (5,011 imgs) + test (4,952 imgs) releases already under
`datasets/voc/`, filtered to images with ≥1 non-difficult object in the 5
target classes (2,724 train / 2,734 test after filtering). No download
needed, but it's the official VOC2007 split, not the paper's split, so
resulting mAP numbers aren't directly comparable to Table I.

```bash
cd dataset_prep
python build_vocfog_from_voc2007.py --out ../datasets/VOC_FOG
```

Both write clean images to `VOC2007/JPEGImages/` and fogged copies to
`VOC2007/FOG/`, plus three annotation-list files at the top of `--out`:

- `2007_train.txt` — clean images, → `config.py: clear_annotation_path`
- `2007_train_fog.txt` — foggy images, → `config.py: train_annotation_path`
- `2007_val_fog.txt` — foggy images (held-out test split), → `config.py: val_annotation_path`

The clean/foggy pair is required for training because LMDNet's restoration
loss needs the clean ground-truth image alongside the foggy one; testing
(`get_map.py`) only reads the `FOG/` folder.

`build_vocfog_from_voc2007.py` additionally augments the *train* split with
`FOG_VARIANTS_TRAIN=3` foggy realizations per clean image (beta ~
Uniform(0.03, 0.18), atmospheric light A ~ Uniform(0.45, 0.65), independently
per realization — vs. `prepare_vocfog.py`'s single beta ~ Uniform(0.05, 0.14)
at fixed A=0.5), closer to IA-YOLO's original 10-fixed-level augmentation than
one fixed-beta copy per image. The *test* split still gets exactly one
realization per image (beta ~ Uniform(0.05, 0.14), A=0.5) for a reproducible
held-out eval. The wider, more varied train-side fog is meant to narrow the
synthetic→real gap visible in Table I (78.39 mAP on synthetic VOC-FOG vs.
59.93 on RTTS / 36.99 on FDD, both real fog) — a detector only ever shown one
narrow fog "look" during training has less reason to generalize to the denser
or more unevenly-lit fog in RTTS/FDD.

## 3. RTTS (test only)

RTTS already ships in VOC2007 format (`Annotations/*.xml`,
`ImageSets/Main/test.txt`, `JPEGImages/*.png`) — but two things need fixing
before `get_map.py` will work: images are `.png` while `get_map.py` hardcodes
`.jpg` for this branch, and this repo's own split
(`dataset_split/rtts_test.txt`, 4,321 ids) should be used as the authoritative
list rather than whatever `ImageSets/Main/test.txt` RESIDE ships.

```bash
python prepare_rtts.py --source /path/to/HazeDetection --out /data/RTTS
```

## 4. FDD (test only, 101 images)

FDD's bounding boxes come as Cityscapes-style `*_polygons.json` (bbox =
enclosing rectangle of each labeled polygon), not VOC XML. **Verify this
assumption before converting anything:**

```bash
python prepare_fdd.py --inspect --source /path/to/Foggy_Driving
```

This prints the json's actual keys and the first object's fields, and tells
you whether it could locate the matching image file. If it looks as expected
(an `"objects"` list with `"label"`/`"polygon"` keys), convert:

```bash
python prepare_fdd.py --source /path/to/Foggy_Driving --out /data/FDD
```

Cityscapes' 8 "thing" classes are person, rider, car, truck, bus, train,
motorcycle, bicycle; the script keeps the 5 RDFNet trains on (dropping
rider/truck/train, mapping motorcycle→motorbike) and only keeps images that
end up with ≥1 box, since FDD has no official split file to defer to.

## 5. Final layout (all three datasets look like this to the code)

```
<out>/VOC2007/
    JPEGImages/<id>.jpg     # clean (VOC-FOG only)
    FOG/<id>.jpg            # foggy (VOC-FOG only; RTTS/FDD put fogged/real images in JPEGImages/ instead)
    Annotations/<id>.xml
    ImageSets/Main/{train,test}.txt
<out>/2007_train.txt         # VOC-FOG only
<out>/2007_train_fog.txt     # VOC-FOG only
<out>/2007_val_fog.txt       # VOC-FOG only
```

## 6. Wiring it into the code

- Training (`config.py`): set `train_annotation_path`, `val_annotation_path`,
  `clear_annotation_path` to the three VOC-FOG lists above; `classes_path`
  already points at `model_data/rtts_classes.txt` (the shared 5-class list).
- Evaluation (`get_map.py`): set `dataname` to `'VOC-FOG'` / `'rtts'` / `'fdd'`
  and `VOCdevkit_path` to the corresponding `--out` directory; `model_path` to
  the checkpoint under test.

## 7. Sanity checks before spending GPU time

- `wc -l <out>/VOC2007/ImageSets/Main/train.txt` should read 9578 (VOC-FOG),
  `test.txt` 2129 (VOC-FOG) / 4321 (RTTS).
- Open a couple of generated XMLs and confirm the boxes look sane against the
  image (e.g. via `predict.py` or a quick matplotlib overlay).
- Check `prepare_*.py`'s printed "WARNING: N ids not found" — if that number
  is large, the `--sources`/`--source` path doesn't actually contain what the
  split file expects.
