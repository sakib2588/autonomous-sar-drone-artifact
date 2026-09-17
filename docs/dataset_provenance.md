# Dataset Provenance

Verified 2026-08-04. GATE-1 (R2). Every dataset used in the paper must have a
resolvable source recorded here before training/eval begins.

## Training domain — VisDrone (urban-aerial)

- Source: Ultralytics auto-download via `VisDrone.yaml`.
- Status: **fetched and converted 2026-08-04**, `data/raw/VisDrone/` (3.7 GB).
  train 6,471 / val 548 / test-dev 1,610 images (official VisDrone-DET split).
  10 raw classes: pedestrian, people, bicycle, car, van, truck, tricycle,
  awning-tricycle, bus, motor — matches `VISDRONE_MAP` in the plan (R2 Step 3).
- Role: **training domain only.** Never described as SAR data in the paper.
- **Correction to plan v2 assumption (verified 2026-08-04):** the plan assumed
  "VisDrone's official test labels are not public," implying we'd need a
  held-out slice of train-val. That's true of *test-challenge* (withheld GT),
  but Ultralytics' downloader fetches **test-dev**, which has real, non-empty,
  public GT (75,102 instances, verified). We use official train/val/test-dev
  as-is. GATE-2 verified with a full md5 content-hash scan across all 8,629
  images: 0 cross-split duplicate content (3 intra-split dupes, harmless).
  Filename video-ids restart per split (24 coincidental collisions, e.g.
  "0000072" in both train and val) but carry zero shared bytes — confirmed not
  a leak. Proof: `results/split_manifest.json`.

## Cross-domain evaluation — aerial-SAR set

Three candidates checked live (2026-08-04). None is a drop-in — pick per below.

| Dataset | Access | Gated | Size | Status |
|---|---|---|---|---|
| HERIDAL (Bozic-Stulic et al., Split/Zagreb) | Official host `ipsar.fesb.unist.hr` | — | ~68,750 patches (29,050 person / 39,700 negative) + ~500 full-size 4000x3000 images; separate cited split 1,647 images / 3,229 person boxes, single class "person", VOC XML | **DEAD — official server refuses connection (both port 80/443).** Usable only via third-party mirrors: Zenodo `zenodo.org/records/5662351` (8.3 GB, PASCAL VOC, HTTP 200 verified), Roboflow Universe re-uploads (`heridal-human-detection`, `heridal-lrbkc`), Kaggle re-upload (`heridal-2-sar-1`). Licence CC BY 3.0 Unported on the original page. **Cite the mirror explicitly, never the dead canonical URL.** |
| SARD | IEEE DataPort DOI 10.21227/ahxm-k331 (HTTP 200) OR Kaggle mirror `nikolasgegenava/sard-search-and-rescue` (HTTP 200) | Free account required on either (DataPort subscription/account, Kaggle login); Kaggle route is instant after login | 1,981 manually labeled drone/video-frame images of simulated casualties (running/walking/lying/sitting); "SARD Corr" extension adds fog/snow/ice | **Live, obtainable.** Kaggle mirror is the lower-friction path. License likely CC BY on DataPort — not independently confirmed by fetch, verify before citing as CC BY in the paper. |
| WiSARD | `sites.google.com/uw.edu/wisard/` (HTTP 200) | **No — direct Google Drive download, no registration** | ~56,000 images: 26,862 visual-only, 29,989 thermal-only, ~15,453 synchronized visual-thermal pairs | **Live, freely obtainable, MIT licence.** Best access-friction of the three. Includes thermal — relevant to R9 characterisation as a reference point, but not used for RGB training/label space (taxonomy stays separate per the two-branch rule). |

### Decision

**Primary aerial-SAR eval pick: WiSARD** (visual subset, person-only) — zero
registration friction, MIT licence, largest verified-live size. **Fallback / secondary:
SARD via Kaggle** if WiSARD's visual-subset framing or label format doesn't map
cleanly to the person-only cross-domain table. **HERIDAL held in reserve** via the
Zenodo mirror only if both above are insufficient — the dead canonical host is a
red flag for citing it as the primary source in a double-blind submission (a dead
link cited by a reviewer looks bad regardless of mirror availability).

### WiSARD on disk — verified 2026-08-07 (R2 step 5 CLOSED)

Downloaded and extracted. It does **not** live under `data/raw/`; the archives and
the extracted visual subset are on the NVMe at
`sar-drone-data/raw/WiSARD/`:

| file | size | files |
|---|---|---|
| `wisard_v1.zip` | 41.5 GB | 100,874 |
| `wisard_multimodal_sample.zip` | 971 MB | 1,060 |
| `extracted_vis/` (the `*_VIS_*` subset) | 38 GB | 45,755 |

Thermal (`*_IR_*`, 2.16 GB across 54,042 files) is deliberately **not** extracted
— the RGB and thermal taxonomies are separate label spaces.

**Label format: no conversion needed.** Already YOLO `class cx cy w h`,
normalised. Verified across all 38 VIS sequences: **45,640 boxes, every one
class 0, zero malformed lines.** WiSARD is person-only, and person is index 0 in
`SAR_RGB_CLASSES`, so the remap is the identity — asserted in
`src/sar/data/sar_eval.py` rather than assumed.

| measure | value |
|---|---|
| sequences | 38 (6 verified human-free) |
| images | 25,735 |
| person boxes (raw) | 45,640 |
| person boxes (after clip policy) | 44,821 |
| explicit empty-label frames | 3,111 |
| frames with no label file | 5,748 |
| total negative frames | 8,859 (34.4%) |

**Resolution is mixed, not uniformly 4K:** 1920x1080 (16,000 images), 3840x2160
(3,796), 2720x1530 (3,169), 2704x1520 (2,226), 4096x2160 (544).

**Person size, native pixels:** mean 72.55, median 59.96, min 0.5, max 1024.6;
19.01% under 32 px, 4.40% under 16 px, 1.53% under 8 px.

Three properties the evaluation code must respect, all recorded in
`src/sar/data/sar_eval.py`:

1. **Tile, never resize.** Because resolution is mixed, one resize policy gives
   wildly different effective scales: 1080p to 640 is 3.0x (72.55 px to ~24 px)
   while 4K to 640 is 6.0x (to ~12 px, near the stride-8 floor). That would
   confound the domain-gap measurement with a resolution artefact.
2. **A missing label file means zero humans, not unannotated.** The 6 human-free
   sequences carry `count.txt` reading "number of humans: 0" with matching image
   counts. They are true negatives and must be kept — they are what makes the
   R8 Step 5 false-positive budget measurable.
3. **Bootstrap over sequences, not boxes.** Frames are consecutive video and
   strongly correlated within a flight. `sequence_id_of` exposes the correct
   resampling unit.

**Out-of-range boxes:** 829 of 45,640 (1.82%) extend past the frame edge, 813 of
them inside `200426_SkookumCreek_Mavic_Mini_VIS_0007` and `_0008` (8 to 10% of
those two sequences — systematic, not scattered). Policy: clip to the image and
keep the visible part; drop the 819 with zero overlap. The 5 `count.txt`
box-count mismatches are exactly those dropped boxes, so the accounting is
self-consistent.

Remaining caveat: min box side is 0.5 px, so some annotations are degenerate. A
minimum-size floor may be warranted before the cross-domain table is computed.

## Field set — self-captured

Own webcam capture (this PC first, per user 2026-08-04 — validate detection
pipeline locally before quantizing/deploying to Pi4). RGB branch only.
Three sessions planned: dawn / midday / dusk (B3, plan Section 5). Not yet captured.

## Thermal

OWLSHINE Dual Port Thermal Imager (UVC), procurement confirmed — see hardware
procurement memory. Characterisation only (GATE-5), not a training dataset.
