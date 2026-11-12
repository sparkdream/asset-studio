# Rigging and motion capture: the plan to replace anymove

Status: started 2026-11-05. The project is Apache 2.0, and everything here stays Apache-compatible: no GPL code is
copied in or linked (no Rigify, no MPFB2). Blender scripts that use `bpy` stay Apache 2.0, which is GPL-compatible.

## Why

anymove is a general creature pipeline (paws, hooves, tails, two-view depth). Its motion solver only sees joint
**positions**, never rotations, so anything positions can't show was patched in afterwards:
- how far a bone rolls about its own axis (twist prior);
- which way the palm faces;
- shoulder collapse (helper bones);
- held props (grip joint, then re-grip).

Its joint limits are per-axis boxes set by hand, and it has no idea of hinge joints, coupled shoulder range,
collisions or balance.

SAM 3D Body already estimates a full human body model (Meta's Momentum Human Rig) from each video frame, with
**joint rotations** learned from real people. anymove keeps only its joint positions and throws the rotations away.

## Design: one skeleton system, several motion sources

| Creature | Skeleton | Skin weights | Motion |
|---|---|---|---|
| Human, or human-shaped (elf, orc) | humanoid template with twist bones | transferred from MakeHuman's CC0 base body | SAM 3D Body rotations, retargeted |
| Biped with a tail, wings, ears | humanoid template plus extra chains | as above, plus automatic weights for the extras | SAM for the body; tail and wings from spring physics or procedural motion |
| Quadruped or other body plan | its own template (spine, legs, tail, neck) | automatic weights plus anymove's useful fixes | keypoint tracking plus a fitter (ported from anymove) |

There is no commercially usable animal equivalent of SAM 3D Body that we know of. SMAL, the well-known animal body
model, is non-commercial. So the keypoint fitter stays for non-humanoids.

**Every improvement has to work for any character or creature**, not only the one it was tested on (so far, the
rogue). That rules out rules tuned to one character's look, such as finding hair by its texture colour or props by
hand-placed coordinates. Prefer what any mesh has: what a segmentation model sees in the rest renders, the fitted
body, the mesh's own shape. A per-character setting in `character.json` is only an optional override of a general
default.

## Project layout: one project per character

`outputs/<project>/` holds one character from picture to captures:

| Folder | What | Made by |
|---|---|---|
| `views/` | the reference views (front, 3/4s, sides, back, turntable frames), versioned, `views.json` | `reference/views.py` (studio: Views tab) |
| (`../items/<name>/`) | an item any character in the project can hold or wear: its own `views/` and TRELLIS runs | as a character's (studio: "+ New item…") |
| `<pipeline>_<set>/`, other run folders | TRELLIS.2 runs, each from one image or a set of views | `external/trellis2/example_xpu.py` (studio: Models tab) |
| `anymove/` | the character's mesh, scaled and standing (`prep/static.blend`), `creature.json`, move prompts, Wan's reference frame (`runs/ref/ref.png`), captures in `runs/clips/` | `rigging/promote.py` (studio: "Use for the rig"), `rigging/ref_frame.sh` |
| `rigging/` | `character.json`, the fit (`fit/`), the rig (`rigged/`), `promote.json` (which run it came from) | `rigging/promote.py`, `rigging/rig.sh` |
| `archive/<date-time>/` | the mesh, rig and captures a promotion replaced; never deleted, skipped by the studio | `rigging/promote.py` |

To try another TRELLIS run as the character's mesh, promote it: the current one goes to `archive/`, settings and move
prompts carry over, prop tips are found again on the new mesh. A capture belongs to the rig it was made with, so it is
archived with it.

## Steps

1. **Skeleton builder** (new package `rigging/`):
   - body-plan templates: humanoid, humanoid plus tail, quadruped;
   - twist bones in the upper arms, forearms and thighs;
   - fitting a template to a mesh. A humanoid's joints can be fitted automatically by running SAM 3D Body on renders
     of the static mesh.
2. **Human motion: SAM 3D Body retargeting.** Per frame: SAM's joint rotations → mapped onto the target skeleton,
   relative to SAM's reading of the character's own rest pose (so the scan's rest pose needn't be a T-pose) →
   smoothing → Blender bake → GLB. The first target is the current rogue rig, compared side by side with the anymove
   capture of `slash_windup`. Later, a light 2D touch-up for hands and a held prop's tip.
3. **Skin:** humanoid weights transferred from MakeHuman's CC0 base body (loaded directly, no MPFB2). Check each asset's
   licence file. Clothing that hangs away from the body (coat tails) gets cloth simulation or its own bones.
4. **Tails, wings, hair:** generalise the spring bones from anymove's `add_hair.py`.
5. **Quadrupeds:** port and clean up anymove's keypoint fitter, with this session's fixes (twist prior, rotation priors).
6. **Studio:** switch it to the new pipeline, add single-view video generation (SAM needs only one view, which also
   removes the "side half turns into a second front view" failure), then remove the `external/anymove` submodule.

## Progress

- 2026-11-05, step 2 first version (`rigging/`):
  - `sam_body.py` reads SAM 3D Body's joint rotations and positions.
  - `blender/render_rest.py` and `blender/retarget.py` drive the current rogue rig from them, with:
    - limbs aimed along SAM's joint positions;
    - feet grounded;
    - a 2D touch-up for held props;
  - `capture.sh` runs it all.
  - First clip: `slash_windup_sam`, next to anymove's `slash_windup`.
  - Still borrows anymove's renderer and showcase, and anymove's keypoint reader for the dagger tips.
- 2026-11-06, step 1 first version:
  - Template `templates/humanoid.json`: 30 bones including upper-arm, forearm and thigh twist bones, plus the retarget
    map.
  - `fit.py` triangulates MHR's joints from five rest renders.
  - `blender/build_skeleton.py`:
    - joint placement: limb joints centred in the mesh; hips and shoulders put on their limb's own axis (MHR's hip
      joints are 10 cm apart, the mesh's thighs 18);
    - dagger re-grip;
    - automatic weights plus the essential fixes ported from anymove;
    - twist-bone weights.
  - `rig.sh` builds a rig from `character.json`.
  - Clip: `slash_windup_rig2`.
  - Known issues for step 3 (skin): the inner thigh and coat panel between the legs stretch in a wide stance; a slight
    pinch at the sleeve edge.
- 2026-11-06, step 3 first version:
  - MakeHuman's CC0 base mesh, default skeleton and weights in `rigging/data/makehuman/`, read by `makehuman.py`.
  - `build_skeleton.py` poses MakeHuman's body onto the fitted skeleton:
    - two torso frames, turned with the hips and with the shoulders;
    - each limb bone turned the least from its parent, stretched to the bone's length;
    - the template's `makehuman` section maps the points, segments and bones.
  - Each vertex then takes the weights of the nearest point on that body. MakeHuman's skirt is used only for cloth that
    hangs well off the body and is nearer the skirt, never at boot level.
  - Cloth more than 4 cm off the body has its weights spread from what sits on it.
  - Between the legs, the nearest point flips from one leg to the other between neighbouring vertices, which tore the
    crotch open. Blending both legs over a wide band webbed the inner thighs together instead. Now the seam goes half to
    the pelvis and a quarter to each leg, fading back to the vertex's own weights 6 rings of vertices away.
  - That seam still ran down the inner thighs, which press together in the rest pose, and left a sheet between the
    legs. Now the builder finds the height where the mesh parts the legs (the highest cut below which the two knees
    are no longer joined: 0.84 m on the rogue). Below it each leg keeps only its own weights, fully 5 cm down.
  - The rest of the sheet came from the pivot. SAM put the hip joints 16 cm above this character's crotch, and a leg
    swinging about a pivot that high drags the crotch into a sheet between the thighs. Now the builder finds where the
    mesh parts the legs first, and puts the hip joints 5% of the height above it (template `crotch`). On the rogue they
    moved down 7 cm; crotch edges stretched more than 1.8x in the clip dropped from about 750 to about 270 a frame.
  - The coat's collar is nearer MakeHuman's chin than its chest and followed the head. Now nothing below a bone's parent
    joint follows the head or the neck (template `not_below`).
  - A vertex clearly nearer one leg's bones gives up the other leg's weights. Below the thigh, as in the automatic
    path, each vertex keeps to the leg it is connected to across the surface (a heel sits nearer the other foot's toes
    in this stance).
  - The spine's share is split along the spine bones.
  - `"weights": "heat"` in `character.json` brings back the automatic weights.
  - Check renders: `rigged/makehuman_front.png` and `makehuman_side.png` show the posed body over the character.
  - Stress pose (wide stance, arms raised): the trousers no longer balloon between the legs, and the coat flaps hang at
    the sides. The longest stretched edge went from 30 cm to 12 cm.
  - Still open:
    - the coat's back panel stretches between the legs (cloth simulation or coat bones, later);
    - a patch of the upper back, under the hair, follows the right upper arm: in the rest pose that arm hangs just
      behind the back.
  - Clip: `slash_windup_rig3`.
- 2026-11-07, Kimodo as a motion source:
  - `kimodo_gen.py` makes motion from text with NVIDIA's Kimodo on the SOMA skeleton (it needs the gated
    Llama 3 8B Instruct for its text encoder), on the Intel GPU.
  - `blender/retarget_kimodo.py` and `templates/humanoid_kimodo.json` retarget it the same way as SAM's.
  - Short, mocap-caption-like prompts move far better than long descriptions of a path ("A person swings a sword
    horizontally from right to left with their right hand.").
  - A capture made without a clip names its move and prompt in `source.json`, and the studio files it under that move.
  - A compound move is a sequence of prompts separated by `|`, each optionally with its length (`0.8s: …`), which
    Kimodo plays in order with its own transitions. In one long prompt it does the main action and drops the rest
    ("spins with arms crossed, then lands on one knee" gave spins only). As three segments it gave all of it.
  - Keyframes (`kimodo_capture.py --pose TIME=CAPTURE[:FRAME][@DEG][+arms=CAPTURE[:FRAME]]`, `--heading TIME=DEG`)
    pin the body to a frame of another Kimodo capture, or the hips' facing, at a time. The pose is moved onto the
    start's spot and turned to face DEG. `+arms=` and `+legs=` take both arms or both legs from another frame, so one
    capture's body can be combined with another's arms or legs. Workflow: make the key poses as moves of their own (a landing lunge), then pin them
    into the hard move.
  - The spinning jump, slash and lunge (`spin_slash_land_kimodo_s93_*`) needed four pins: arms crossed standing; the
    jump's apex with crossed arms, facing 180°; a quarter turn either side of it, to set the spin's direction; and
    the lunge with arms spread at landing and at the end. Text and segments alone gave a pirouette with no flight, or
    landed upright. Wan failed it outright (it dropped the split-screen and never jumped).
  - Legs straight and crossed through that spin (`s95_*`): the crossed legs from a move of their own ("A person steps
    one foot across in front of the other and stands with crossed legs."; "crossed at the ankles" got nothing). They
    were pinned with the crossed arms every 0.1 s across the spin, on the jump's frames, the facing stepping 45° each
    pin. With pins 0.2 s apart, the knees bent to 135° and the legs uncrossed between them.
  - Higher, with straight legs (`s96_*`): `+straight` sets the knees to SOMA's rest angle (straight), and `^Y` sets
    the hips' height at each pin along a ballistic arc (y = 1.75 − 4.9·dt² m, every 0.1 s). That gave feet 0.75 m off
    the floor, 0.8 s in the air and knees at 176°. Taken from the jump's apex frames as they were, straight legs had
    lowered the feet to 0.4 m: the hips rose no higher than the tucked jump's.
  - Upright and twice as fast (`s98_*`): `+upright` keeps only the hips' facing and sets the spine, neck and head to
    the rest pose. The jump's apex frame had leant the body 26° forward; now it leans 2°. Two turns in the 0.8 s
    flight: pins every 2 frames, each turned 61.5°. Pin times must fall on whole frames: at 0.05 s (1.5 frames) they
    rounded to gaps of 1 and 2 frames, and the spin pulsed between 46° and 23° per frame.
  - Recipes: a move's prompt and keyframes in `anymove/prompts/<move>.kimodo.json` (`kimodo_capture.py --recipe`; the
    studio's Kimodo card edits it under "Keyframes (recipe)"). Its `"jump"` block computes the pins: a ballistic arc
    from the top height and time, a pose held every few frames of the flight, an optional crouch before takeoff, and a
    spin of so many turns in so many seconds centred on the top. `spin_slash_land.kimodo.json` is the worked example:
    one fast turn (0.39 s, about 920°/s) at the top of a 0.8 m jump (`s99_*`).
  - Fixed in `retarget_kimodo.py`: the hips' height had kept the lowest foot on the ground in every frame, so on the
    rig no Kimodo jump left the floor. Now the rig's feet rise as far above the ground as SOMA's do (scaled to the
    rig), and only the ground-contact part is smoothed. The jump, spin-jump and spin-slash captures were re-exported.
    The run's loops couldn't be: their source captures had been deleted.
  - Pins must sit inside a single-prompt motion. With `|` segments, a pin on a segment's first frame fights Kimodo's
    own transition pins: the body snaps into the pose for a frame, then collapses.
  - `rigging/loop.py` turns any capture into a cycle (walk, run, crouching walk): the stretch whose end best matches
    its start in pose and motion, the seam spread over it, in place or with root motion.
  - It shares about 60 lines with `retarget.py`: make them one module if Kimodo stays.
- 2026-11-07/08, the mesh itself: reference views and multi-view TRELLIS.2:
  - The crotch still looked wrong in a wide stance because TRELLIS never saw the inner thighs. The source picture has
    the legs together and crossing, so the thighs came out fused and flat on the inside.
  - `reference/qwen.py` edits pictures with words, using Qwen-Image-Edit-2511 locally. The image model is the GGUF
    Q8 build, and it runs in two phases, so the text encoder and the image model never share the 32 GB GPU. Neither is
    ever copied whole into RAM: diffusers' own GGUF loader does that and froze WSL. An edit of the front picture into
    an A-pose, legs apart, gave legs that stay separate up to the crotch.
  - `reference/views.py` keeps a character's views, all made from its front picture. The sets: front, front + 3/4,
    front + side, front/left/back/right, front + both 3/4s, eight at 45°, and turntable frames from a Wan orbit
    video. Any view can be edited with words, replaced by an image of your own, approved, and kept in versions.
  - The view prompts are generic (no character's details), and say what the picture shows rather than where the
    camera went: Qwen confuses the subject's left and right. Even so, one seed of the left view faced the wrong way.
    Review views before a run.
  - `example_xpu.py` takes several views. At every sampling step each view's prediction is made and the mean followed,
    TRELLIS's own multidiffusion idea. It works because TRELLIS builds an object in its canonical frame whatever the
    camera. The latents are saved before decoding, so a decode that loses the device resumes without sampling again.
    The Intel GPU stack does that now and then.
  - On the rogue:
    - Front + 3/4 gave thin blades in place of front-to-back fans, and legs apart up to the crotch: 12 cm at 0.75 m,
      against 4 cm from the front alone.
    - Four views gave about the same body, with the daggers angled as the side views drew them, for three extra Qwen
      views.
    - Front + 3/4 is the default trade. More views pay off when the back matters: capes, wings, tails.
- 2026-11-08, rigging that works for any character (no per-character thresholds or coordinates):
  - Hair: `blender/region_views.py` renders the rigged mesh from 8 directions. `segment.py` has SAM 3 mark "hair" in
    each render (`character.json` `"hair": {"prompt": "fur"}` names it otherwise). Each vertex then takes the vote of
    the views that see it. `rig.sh` runs it before anymove's `add_hair.py`, which now reads `hair_region.npy` and keeps
    its colour rule only as a fallback. The rogue's brown hair had defeated that rule.
  - Prop tips are found from the mesh: of the surface joined to the hand that lies nearer the hand bone than any other
    bone, the point farthest from the wrist. Nothing past 1.5 hand lengths means an empty hand. A `"tip"` in
    `character.json` only overrides it. On both rogues it landed within 4 mm of the hand-placed tips.
  - A re-gripped prop takes only geometry joined to its blade or the fist, plus loose pieces centred on its axis just
    behind the fist (pommels). Loose scraps of hair or cloth that fell inside its cylinder used to ride with the hand.
  - Tethered limbs (the template's `"tethered"`: the arms): a limb bone's weight is kept to what lies near it along the
    surface, holding for 2 of its radii and gone by 4. The radius is measured on the mesh; the distance runs from the
    vertices within 1.25 radii, so both walls of TRELLIS's double-walled surfaces count. Weights copied from the
    nearest body surface had reached a coat's flank beside the hanging elbow, which the raised arm dragged into bands
    down to the belt. In the rogue's lunge, edges stretched more than 3x dropped by a quarter. Two straight-line
    cut-offs were worse: the limb's own skin tore off.
  - The legs are not tethered: coat panels over the thighs need the skirt's weights.
  - Loose cloth (`blender/add_cloth.py`, run by `rig.sh` after the hair): below the hip joints, what a leg moves but
    lies further from every leg bone than 1.6 times that leg's measured thickness. Each connected piece that borders
    vertices the torso carries is a panel. A loose boot top or trouser cuff borders only its leg, and stays on it.
    Each panel gets a chain of three spring bones from its top to its hem, parented to the root bone and simulated by
    `apply_motion.py --springs` like the hair. Its leg weight goes to the hips, the chain takes over towards the hem,
    and the weights are blended over 16 rings of vertices round the panel (about 11 cm). With 2 rings the stretch
    only moved to the seam between panel and coat.
  - On the rogue: two back panels. The rig check went from four leg bones dragging 1,681 vertices to one dragging 119.
    Coat edges stretched more than 1.8x, at the peak frame: 2,928 to 1,759 in the slash, 1,093 to 567 in the walk.
    From behind, the vent now opens in a lunge instead of the panel stretching after the leg.
  - There's no collision yet: a panel or a lock of hair can pass into the body (the capture check flags it).
  - Wan's reference frame (`rigging/ref_frame.sh PROJECT`, run by `promote.py` and the studio's rig rebuild): the
    rig's two-camera render with each limb painted its colour (`blender/paint_limbs.py`). The limbs are the template's
    chains on both sides, with the bones below them but no props, starting at the shoulder rather than the clavicle.
    `creature.json` `"paint"` gives the colours in order, and its bone lists are rewritten to the rig's names. A
    vertex keeps its paint only within 1.35 times the limb's measured thickness of its bones (`"paint_thickness"`
    changes it): skin weights also give a limb the cloth it carries, and the rogue's coat front, painted with the
    thighs, read as trousers. Promotion had left the project without a frame, so Wan clips couldn't start.
- 2026-11-08, studio:
  - The Views tab manages a project's views: make a set's missing views, remake one, edit with words, approve, use any
    image or an older version. It runs TRELLIS on a set and warns about views not yet approved.
  - "Use for the rig" on a model promotes it.
  - Qwen, TRELLIS and promotion jobs run in a memory cap (`systemd-run --user --scope -p MemoryMax=… -p
    MemorySwapMax=0`), so a runaway job fails on its own instead of freezing WSL. `STUDIO_DRY_RUN=1` shows job scripts
    without running them.
- 2026-11-09, multi-view TRELLIS.2 blended by what each view sees; items:
  - TRELLIS.2's frame, measured on the rogue's views: z up, a character faces -y, and a picture taken from angle a
    (round the subject towards its own left, as `reference/views.py` names views) comes from (sin a, -cos a, 0). Each
    side or 3/4 view's own reconstruction, projected that way, matched its silhouette (IoU 0.67-0.84; mirrored
    0.08-0.10), and a side view's reconstruction matched the other side's picture too: the views share one frame.
  - The plain mean lets views that can't see a part (a pouch on one hip, a quiver on the back) outvote those that do.
    `trellis2/pipelines/view_weights.py` weighs each view, per voxel, by its transmittance from that view's camera,
    and by how the surface there faces it (for the shape, a picture pins what it sees edge-on, not how deep what faces
    it lies; for the texture, the reverse). Stage 1 takes the occupancy decoded from the current estimate at each
    step, the later stages the sparse structure. `example_xpu.py` reads each view's angle from its file name; turntable
    frames have none and are averaged. `TRELLIS_VIEW_WEIGHTS=mean` (studio: "Blend views") keeps the mean.
  - Stage 1 on the rogue's four views, six seeds, silhouette IoU against the four pictures: mean 0.714 on average
    (worst 0.58), visibility alone 0.694 (worst 0.53), visibility with facing 0.726 (worst 0.53; five seeds of six at
    0.73 or more). Every failure, with any blend, was the daggers' depth: the front picture can't tell a blade pointing
    forward from one pointing sideways, and the result flips between the front's and the side views' readings by
    seed. The case the weighting is for, a part some views can't see, isn't in these views and is still to measure.
  - The daggers come off the character instead: `rogue_unarmed` is the rogue's front with empty loose fists
    (Qwen-Image-Edit), and the dagger is an item of its own, made from a Qwen picture of it alone, so it fills
    TRELLIS's whole grid. Fists rather than open hands: the template has no finger bones, so a held item goes
    through a closed fist.
  - First full runs with the new blend: `rogue_unarmed/trellis_four` (four views, 512) came out clean from every
    side, with empty fists, and its texture is free of the dark streaks the old four-view model had down the front
    (not a like-for-like comparison: the views differ). `items/dagger/trellis_1024c` (one picture, 1024 cascade, 50k
    faces): the guard, grip wrap and engraving are all there, where the rogue's own dagger was a few voxels wide.
  - Correction, from a closer look: the 512 run's back of the hair is blotched with the coat's green, and the hair's
    brown is smeared onto the left of the neck. That comes from the resolution, not the blend: at 512 the texture lives
    on a 32³ grid (953 voxels for this character), where the hair, the coat under it and the neck share voxels. Texture
    stage alone, same shape and seed: the plain mean has the same marks, and letting the first voxel of occupancy
    along each ray hide nothing (so a camera still colours the voxel just behind the hair) barely changed them, so that
    change was dropped. The same views at 1024 cascade (`rogue_unarmed/trellis_four_1024c`, 64³) are clean on the back
    of the hair and the neck. Characters with layers close together (long hair over a coat) need 1024 or more.
    Left: a dark cross-shaped mark on the upper chest, and green flecks on the lapels' grey lining.
  - Moves copied between characters (`rigging/copy_moves.py`; the studio's Rig page, "Copy moves from another
    character"). A motion.json can't be copied as it is: it turns bones from its own rig's rest pose, and rigs rest
    differently (rogue_apose's upper arm 49 degrees below horizontal, rogue_unarmed's 76: copied, the arms stuck out
    sideways). A Kimodo capture is retargeted again from its kimodo.npz; any other (loops, whose source captures were
    deleted, and captures from video) from the other rig's motion (`rigging/blender/retarget_rig.py`: each bone's world
    turn since its rest, each limb aimed along the other rig's, hips scaled by hip height). On the rig itself it gives
    back the same angles (within 0.001 degrees). rogue_apose's 10 kept captures copied onto rogue_unarmed: 3 from Kimodo,
    7 rig to rig, matching the originals frame for frame; the weapon bones are left out.
  - Sockets and grips, the first half of the items plan. `rigging/blender/add_sockets.py` (run by rig.sh last, or on
    an existing rig: weights untouched, captures still valid) adds `socket_hand_L/R`, bones that deform nothing, from
    the hand's own vertices: the head at the middle of the fist, +Y grip_deg (45, as build_skeleton.py re-grips a held
    prop) off the hand towards the thumb, +Z out of the back of the hand. They are nodes in every exported GLB.
    `rigging/item_grip.py` writes an item's item.json: its grip (origin, axes in the GLB's own coordinates: +Y to the
    tip, +Z out of its face) from its width along its length (a guard: the grip on its shorter side; a head at one
    end: the grip near the other end; anything else set by hand), the sockets it fits, and its real length (TRELLIS
    makes everything one unit long). The dagger: 0.61 m, from rogue_apose's weapon bones (knuckles to tip 0.51 m).
    Checked on rogue_unarmed's double slash with a dagger in each hand: in the fists, guard above the knuckles, the
    blades where the original's were. The handle passes through the closed fingers (no finger bones). rogue_unarmed's
    rig has its sockets and its 10 captures are exported again with them.
  - Items for the game, the equip preview and the export. Blender's glTF exporter keeps each bone's own axes on its
    node (measured: node world = the axis change times the bone's matrix, nothing else), so an item's light copy is
    baked already aligned to a socket: `rigging/item_game.py` ("Use for the item" on an item's model, and after an
    item's first TRELLIS run) estimates the grip, bakes a copy again from the run's voxels (5,000 faces, 1024 texture;
    the dagger 0.4 MB), moved so the grip is its origin and scaled to metres: in three.js, `socket.add(item)`.
    `studio/equip.js` (three.js 0.170) is both the studio's equip preview (Rig page) and the export's example: a
    character GLB, clips from other GLBs by bone name, items under sockets. `rigging/export_game.py` (Rig page, "Export
    the project for the game") writes `<project>/export/`: character.glb (rest pose, sockets;
    `rigging/blender/export_character.py`), each clip with no mesh (`apply_motion.py --anim-only --action NAME`, 70-120
    KB), items, manifest.json, equip.js and index.html. Default loadouts: `rigging/loadout.json`, saved from the
    preview. Checked in headless Chromium: the export's page and the studio's preview play the double slash with a
    dagger in each hand, as in the Blender test. The character is 12 MB (TRELLIS's 200k faces): the next step is the
    Models tab's Optimise.
  - Optimise, in the Models tab before rigging (the rig is fitted to one exact mesh, so a mesh optimised at export
    would never have been rigged or checked). `rigging/optimise.py` works from the run's sample_xpu.pt, not its model:
    o_voxel's `to_glb` is split into `remesh_snap` (the surface rebuilt from the voxels, snapped onto TRELLIS's mesh)
    and `bake_textured` (the texture baked from the voxels onto any triangles with UVs), with the same steps and
    results as before; between them `rigging/blender/optimise_mesh.py` decimates (tris) or lays quads with QuadriFlow
    (quads, from a 300k-face copy). The copy is a model of its own (`<run>_opt<k>k[_quads]/`, with optimise.json and
    layout.png, `rigging/blender/layout_sheet.py`: the faces over the body and the head; quads also as quads.obj for
    Blender, as glTF has no quads), to compare and to use for the rig. About 5 minutes for a 1024 run (5.2M faces
    rebuilt). rogue_unarmed at 30,000 triangles: 1.9 MB (the original 12 MB), the texture close to the original's
    (the face slightly softer); kept as `rogue_unarmed/trellis_four_1024c_opt30k`.
  - QuadriFlow's quads at the same budget (12,813 quads): even, flowing along the limbs, coat and legs, but the head
    torn and folded: hair strands lie over the cheeks closer together than a quad (about 1.5 cm), and one even grid
    can't keep the layers apart. Not the shrink-wrap back onto the detailed surface (without it, the same; it is now
    off, OPTIMISE_SHRINKWRAP to try it). Twice the budget (26,235 quads) helps only partly: the face is cleaner, but
    where the hair meets the face the quads still fold. So not density alone: one layout across layers that almost
    touch. Next to try: the hair laid out apart from the body (the segmentation the rig already makes,
    hair_region.npy), so no quad bridges the two, and density by region.
  - Tried (2026-11-09): the hair laid out apart. SAM 3 marks the hair in eight renders of the run's own model
    (region_views.py, segment.py), the marks carried by nearest point onto the decimated surface; faces labelled,
    the share smoothed over the surface first, small islands joined to their surroundings, corner-only contacts given
    to the body so each part is manifold; then QuadriFlow on each part (optimise_mesh.py, `region` in in.npz). The
    labels are clean (hair and face part exactly at the hairline), but the face is no better: with its boundary kept
    it gets a few large quads and spikes, without it the same. With the hair gone the face is a simple surface, and
    QuadriFlow still lays it badly, its corners standing off the surface. So the hair was not the main cause:
    QuadriFlow does poorly on small detailed features (nose, eyes, mouth) at a game budget. Left to try: another quad remesher (Instant Meshes, open source, with its own density
    and orientation fields), or quads only where they matter for bending (joints) and triangles elsewhere. For the
    game, triangles (tris mode) do the job: glTF is triangles anyway.
  - Decided (2026-11-10): triangles. The studio's Optimise makes triangles only; quads stay a command-line experiment
    (`rigging.optimise --mode quads`), the split code taken out (the attempt is described above).
  - Items: `outputs/<project>/items/<name>/`, shared by every character in the project and made like a character
    (views, TRELLIS runs); the studio lists them under "<project> · items" and makes them with "+ New item…".
  - For the game (open source, in the browser): Three.js, from separate files: the character (skinned mesh, skeleton,
    socket nodes such as `socket_hand_R`), each clip exported per character (each skeleton is fitted to its own mesh,
    so clips don't transfer by bone name), each item with its origin at its grip, and a manifest. Equipping is the
    game's data: an item is added under a socket at runtime. Not built yet.
- Still open:
  - the raised arm's shoulder stretches;
  - collision for spring bones (hair and cloth pass into the body);
  - the studio doesn't see runs started outside it (their GPU use, or a views.json they rewrite);
  - the Wan step of a turntable job isn't memory-capped yet.

## What to carry over from anymove (adapted, not imported)

- spring hair (`blender/add_hair.py`);
- GLB baking and rendering (`blender/apply_motion.py`): already bakes constraint-driven and spring bones correctly;
- the showcase video (`mocap/showcase.py`);
- Wan generation and prompt building (`mocap/wan.py`, `tools/make_prompt.py`);
- for quadrupeds: the keypoint reader and fitter (`mocap/detect.py`, `mocap/solve.py`).

## Licences

| Component | Licence | How we use it |
|---|---|---|
| This project, anymove | Apache 2.0 | our code |
| TRELLIS.2, FlexGEMM | MIT | submodules |
| Wan | Apache 2.0 | video generation |
| SAM 3D Body | SAM License (commercial use allowed) | called at capture time, not bundled |
| MakeHuman base mesh (hm08), default skeleton and weights | CC0 (stated in each file) | data files in `rigging/data/makehuman/`; MakeHuman's code (AGPL) is not used |
| Blender | GPL | runs our scripts; not distributed |
| Qwen-Image-Edit-2511 (and unsloth's GGUF build of it) | Apache 2.0 | reference views, run locally; not bundled |
| SAM 3 | SAM License of 2025-11-19 (commercial use allowed; gated on Hugging Face) | hair and other regions at rig time; not bundled |
| Kimodo | Apache 2.0 (code); SOMA weights under NVIDIA's Open Model License (commercial use allowed) | motion from text; not bundled |
| Meta Llama 3 8B Instruct | Llama 3 Community License (gated) | Kimodo's text encoder, at generation time only; not bundled |

**Stand, crouch, twirl, strike (`s111`).** The spin-slash recipe now starts from a standing frame (idle, frame 0). The crouch is shallower (`jump_kimodo_s31_0:15`), with the arms half spread: `"arms": [standing, "double_slash_kimodo_s61_1:74*0.5"]`. The new `*MIX` blends local rotations joint by joint from the arms so far toward the named frame's arms; it's in kimodo_gen as `+arms=SRC*0.5`. The landing slash used to be lazy, because Kimodo spread the arms over the whole descent. A new jump `"release": {"arms", "seconds": 0.2}` pins the air pose's arms sweeping into the slash over the flight's last 0.2 s, speeding up (k²), so the hands go from crossed to fully spread in about 4 frames, ending at touchdown. The lunge is pinned 0.2 s after landing. The old recipe is kept as `spin_slash_land.kimodo.v1.json`.

**Spinning into the jump and travelling forward (`s112`, `s113`).** The jump's `"spin"` takes `"start"`, in seconds from takeoff. With a negative value (−0.07), the turn begins while the feet are still pushing off: `--heading` pins cover the frames before the flight. In `s112` she's 60–90° round at takeoff, and the 360° ends just before the apex. `"forward"` (1.5 m) moves the hips' pins forward evenly over the flight, and poses after the landing sit at the landing spot. This works through kimodo_gen's `>Z` on `--pose` and `--heading`, which sets the root's z (the start's facing). `s113_1` travels 1.45 m with 4 cm of sideways drift.

**Studio job clean-up.** The Jobs tab has a checkbox on every queued or running job, and buttons for Cancel selected, Cancel automatic and Cancel all (`/api/jobs/cancel_many`; queued jobs are cancelled before running ones). Generating a Kimodo move again first offers to cancel the queued automatic checks of that move's earlier captures (`/api/jobs/move_checks`, matched by each capture's source.json `move`). Cancelled checks aren't queued again until their capture changes or the server restarts.

**Shorter leap, soft landing (`s115`).** `"forward"` is now 1.0 m. The legs used to stay straight and crossed until the last frame in the air, so she touched down on straight knees. The jump's `"release"` now takes `"legs": "land_lunge_kimodo_s81_2:25*0.45"` over `"legs_seconds": 0.25`: the air pose's straightened legs ease (smoothstep) 45% of the way toward the lunge's legs before touchdown. Then the lunge pin finishes the bend and the hips sink about 0.2 m over 4 frames. Going all the way (`s114`, `*1`) arrived with the lunge already formed and nothing left to absorb. This needed kimodo_gen to apply a pose's `+` steps in order (`+legs=A+straight+legs=B*0.3`), and recipes carry the release's leg blend after the flags.

**Less hang, a smaller dip (`s115b`).** The flight follows the arc's physics, so hips 0.75 m above standing meant 0.78 s in the air, which looks floaty. The top is now 1.6 m, for 0.67 s. The landing used to pin the lunge clip's deepest frame (25: hips 0.61 m, torso leaned 47°), and the end repeated it, so she stood back up and dipped twice. Now frame 20 is pinned with the hips at 0.76 m at 1.9 s, 2.3 s and the end (2.7 s): a single dip of about 30°. The same seed with new keyframes keeps each sample's character (`s115b_3` is `s115_3` with these changes).
**kimodo_capture no longer replaces captures.** A seed used again writes `<move>_kimodo_s<seed>b_*`, then `c`, and so on. It used to delete the earlier captures of that seed.
