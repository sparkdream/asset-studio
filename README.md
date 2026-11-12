# asset-studio

From one picture to a rigged, animated game character: reference views, a textured 3D model, a skeleton with skin
weights, and motion clips, ready for a browser game (three.js).

Everything runs locally. The current setup targets an **Intel GPU (xpu) under WSL2**, and CUDA works for most steps.
The code is Apache 2.0, and every model it calls allows commercial use (see [Licences](#licences)).

## The pipeline

```
picture ──► reference views ──► TRELLIS.2 model ──► promote ──► rig ──► captures ──► game export
            (Qwen-Image-Edit)   (multi-view)        (scale,     (SAM 3D   (Kimodo text-to-motion,
                                                     archive)    Body,     or SAM 3D Body on a video)
                                                                MakeHuman
                                                                weights)
```

1. **Reference views:** [`reference/views.py`](reference/views.py) starts from a front image and makes the other
   views (3/4s, sides, back, turntable) with Qwen-Image-Edit-2511. Each view is versioned, and you can edit it with
   words or replace it with your own image.
2. **3D model:** TRELLIS.2 ([`external/trellis2`](external/trellis2), our Intel GPU fork) builds a textured mesh from
   one view or several. With several views, each voxel follows the views that can see it.
   [`reference/reproject.py`](reference/reproject.py) then repaints the texture with the real pixels from the views.
3. **Promote:** [`rigging/promote.py`](rigging/promote.py) makes a model the character's mesh. It scales the mesh to
   real height and moves the previous mesh, rig and captures to `archive/` (nothing is overwritten).
4. **Rig:** [`rigging/rig.sh`](rigging/rig.sh) renders the mesh at rest, and SAM 3D Body reads the joints from those
   renders. It then fits a skeleton template with twist bones (`rigging/templates/`) and transfers skin weights from
   MakeHuman's CC0 base body. Last, it adds spring bones for hair (found by SAM 3) and loose cloth, plus sockets where
   items attach.
5. **Motion:**
   - from words: [`rigging/kimodo_capture.py`](rigging/kimodo_capture.py) generates the move with NVIDIA's Kimodo and
     retargets it onto the rig. Recipes chain moves together, pin poses and add spins.
   - from a video: [`rigging/capture.sh`](rigging/capture.sh) reads the joint rotations of each frame with SAM 3D Body
     and retargets them.
   - clean-up tools: [`loop.py`](rigging/loop.py) (cycles), [`join_moves.py`](rigging/join_moves.py),
     [`kimodo_mirror.py`](rigging/kimodo_mirror.py), [`kimodo_posture.py`](rigging/kimodo_posture.py),
     [`kimodo_pose.py`](rigging/kimodo_pose.py), and [`copy_moves.py`](rigging/copy_moves.py) (another character's
     moves on this rig).
6. **Items:** [`item_grip.py`](rigging/item_grip.py) finds where a prop is held, and
   [`item_game.py`](rigging/item_game.py) makes a light copy aligned to the socket.
7. **Game export:** [`rigging/export_game.py`](rigging/export_game.py) writes the characters, clips, items and a
   `manifest.json`. [`studio/equip.js`](studio/equip.js) loads them in three.js
   (see [`studio/export_example.html`](studio/export_example.html)).

The **studio** ([`studio/server.py`](studio/server.py)) is a local web page that drives all of this. Use it to review
views, models, clips and captures, approve or reject them, and queue GPU jobs. Automatic checks
([`studio/checks/`](studio/checks)) look for skin weights that reach too far, limbs that collapse or sink into the
body, and video-model failures.

## Requirements

- Linux or WSL2, with Python 3.10+
- **Blender** on `PATH`, run headless (`blender -b`)
- `ffmpeg`
- A GPU with about 32 GB of memory (the setup is tuned on an Intel Arc Pro B70; Qwen peaks around 27 GB)
- Separate Python environments, which the scripts look for at these default paths:

| Environment | Default path | Used for | Override with |
|---|---|---|---|
| TRELLIS.2 | `~/venvs/trellis2-xpu` | TRELLIS.2, studio jobs, capture tooling | `STUDIO_PYTHON`, `PYTHON` |
| SAM 3D Body + SAM 3 | `~/venvs/sam3d-xpu` | joint reading, hair segmentation (transformers ≥ 5.20) | `SAM_PYTHON`, `BODY3D_PYTHON` |
| Kimodo + Qwen | `~/venvs/kimodo-xpu` | text-to-motion, reference views | `QWEN_PYTHON` |

Gated Hugging Face models (SAM 3, Llama 3 8B Instruct for Kimodo's text encoder) need an accepted licence and a
token in your Hugging Face cache.

## Setup

```bash
git clone --recurse-submodules https://github.com/sparkdream/asset-studio.git
cd asset-studio
source env.sh            # or let direnv load it through .envrc
```

`env.sh` selects the TRELLIS.2 backends that work without the CUDA-only extensions. Machine-specific settings (for
example `HF_HOME`) go in `env.local.sh`, which git ignores and `env.sh` loads.

For the TRELLIS.2 environment on an Intel GPU, follow the header of
[`external/trellis2/requirements-xpu.txt`](external/trellis2/requirements-xpu.txt).

## Quick start: the studio

```bash
python3 studio/server.py          # then open http://localhost:8765
STUDIO_DRY_RUN=1 python3 studio/server.py   # show each job's script without running it
```

Jobs run on two queues, one for the GPU and one for CPU-only checks. Qwen, TRELLIS and promotion jobs run inside a memory cap (`systemd-run … MemoryMax`), so a runaway job fails on its own
instead of freezing WSL. Each job writes its log next to its output. Verdicts and notes go to
`outputs/<project>/review.json`.

## From the command line

```bash
P=outputs/myproject/hero

# 1. views
python -m reference.views $P --front hero.png
python -m reference.views $P --set four

# 2. a TRELLIS.2 model from those views (inside external/trellis2)
(cd external/trellis2 && PYTHONPATH=o-voxel python example_xpu.py \
   ../../$P/views/front.png,../../$P/views/left.png,../../$P/views/back.png 1024 ../../$P/trellis_four)

# 3. make it the character's mesh, and rig it (promote runs rig.sh)
python -m rigging.promote $P $P/trellis_four/<model>.glb --height 1.75

# 4. a move from words
python -m rigging.kimodo_capture $P walk "A person walks forward." --samples 4

# 5. export for the game
python -m rigging.export_game plan.json
```

Every script explains its arguments in the docstring at the top of the file. Most also accept `--help`.

## Layout

```
reference/          views (Qwen-Image-Edit), reprojecting the views onto a model
rigging/            promote, fit, skeleton, retargeting, Kimodo, loops, items, game export
  blender/          scripts run inside headless Blender (bpy)
  templates/        skeleton templates and retarget maps (humanoid, Kimodo's SOMA, MHR)
  data/makehuman/   MakeHuman CC0 base mesh, skeleton and weights (data only)
studio/             local review and job server, page, checks, three.js equip viewer
external/           submodules: trellis2, flexgemm (Intel GPU forks), anymove (being phased out)
docs/               rigging-plan.md: design, progress log, licences
outputs/            projects (git-ignored)
```

Each character is one project folder, `outputs/<project>/<character>/`, holding its `views/`, its TRELLIS runs,
`anymove/` (the mesh and captures), `rigging/` (`character.json`, the fit, the rig) and `archive/`. Items live in
`outputs/<project>/items/<name>/`. See [docs/rigging-plan.md](docs/rigging-plan.md#project-layout-one-project-per-character)
for the full layout.

## Status

This is a work in progress. Humanoids are fully supported. Tails, wings and quadrupeds are planned (a body-plan
template plus spring bones or a keypoint fitter). Some steps still call scripts from
[anymove](external/anymove) (hair springs, GLB baking, rendering, the showcase video, Wan video generation); these are
being ported, and the submodule will then be removed. The plan and a dated progress log are in
[docs/rigging-plan.md](docs/rigging-plan.md).

A design rule: every method must work for **any** character, with no thresholds, colours or coordinates tuned to one
mesh. A setting in `character.json` only overrides a general default.

## Licences

asset-studio is licensed under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for adapted code. No GPL code
is copied in or linked. Blender runs our scripts but isn't distributed.

| Component | Licence | Use |
|---|---|---|
| TRELLIS.2, FlexGEMM | MIT | submodules |
| Qwen-Image-Edit-2511 | Apache 2.0 | reference views, not bundled |
| SAM 3D Body, SAM 3 | SAM License (commercial use allowed) | joints and regions, not bundled |
| Kimodo | Apache 2.0 code, NVIDIA Open Model License weights | text-to-motion, not bundled |
| Llama 3 8B Instruct | Llama 3 Community License | Kimodo's text encoder, not bundled |
| Wan | Apache 2.0 | video generation |
| MakeHuman base mesh, skeleton, weights | CC0 | data files only; MakeHuman's AGPL code is not used |
