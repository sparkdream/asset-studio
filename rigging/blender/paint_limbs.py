"""Paint each limb its colour on a copy of the rig, for the video model's reference frame (anymove's render_ref.py).

blender -b -P rigging/blender/paint_limbs.py -- rigged.blend creature.json painted.blend

The limbs are the template's (rig_info.json names it), each chain on both sides with every bone hanging below it (twist
bones, toes) but not held props, and without a first bone that hangs off the torso rather than the root bone (a
clavicle, which would paint the shoulders). Their colours and thresholds come from creature.json "paint" in its order
(left before right, arms before legs, as the template lists the chains); creature.json's bone lists are rewritten to
this rig's names so anymove's --clean-paint and checks read the same limbs. Nothing about one character.
"""
import json, pathlib, sys

import bpy
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "external/anymove/blender"))
import common  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
blend, creature_path, out = pathlib.Path(args[0]), pathlib.Path(args[1]), pathlib.Path(args[2])
bpy.ops.wm.open_mainfile(filepath=str(blend))
rig, body = bpy.data.objects["rig"], bpy.data.objects["body"]
info = json.load(open(blend.parent / "rig_info.json"))
tmpl = json.load(open(pathlib.Path(__file__).resolve().parents[1] / "templates" / f"{info.get('template', 'humanoid')}.json"))
char = json.load(open(blend.parent.parent / "character.json")) if (blend.parent.parent / "character.json").exists() else {}
props = set(char.get("props", {}))
bones = rig.data.bones
root = next(b for b in bones if b.parent is None).name

limbs = []  # (side, bone names) in the template's order, left then right
for chain in tmpl["limbs"]:
    for s in ("L", "R"):
        names = [n[:-2] + "_" + s if n.endswith(("_L", "_R")) else n for n in chain]
        names = [n for n in names if n in bones]
        if names and bones[names[0]].parent and bones[names[0]].parent.name != root and len(names) > 1:
            names = names[1:]  # a clavicle: the limb starts at the shoulder joint
        todo, got = [bones[names[0]]], []
        while todo:
            b = todo.pop()
            if b.name not in props:
                got.append(b.name)
                todo += list(b.children)
        limbs.append(got)

creature = json.load(open(creature_path))
old = list((creature.get("paint") or {}).items())
paint = {}
for i, got in enumerate(limbs):
    key, spec = old[i] if i < len(old) else (f"limb{i}", {})
    color = spec.get("color") or list(common.LEG_COLORS.values())[i % len(common.LEG_COLORS)]
    paint[key] = {"color": color, "bones": got, **({"threshold": spec["threshold"]} if "threshold" in spec else {})}
creature["paint"] = paint
creature_path.write_text(json.dumps(creature, indent=2))
common.LEG_COLORS = {k: v["color"] for k, v in paint.items()}
common.LEG_BONES = {k: v["bones"] for k, v in paint.items()}
common.PAINT_THRESHOLD = {k: v["threshold"] for k, v in paint.items() if "threshold" in v}
common.paint_legs(body)  # the material that shows the paint
# Only what the limb itself is: skin weights also give a limb the cloth it carries (a coat's front panels on the thighs),
# which painted would read as trousers. A vertex keeps its paint within `thickness` times the limb's own thickness (the
# median distance of the vertices it surely moves) of the limb's bones; loose pieces (floating bits) stay unpainted.
thickness = float(creature.get("paint_thickness", 1.35))
co = np.array([body.matrix_world @ v.co for v in body.data.vertices])
nv = len(co)
gn = [g.name for g in body.vertex_groups]
W = np.zeros((nv, len(gn)))
for v in body.data.vertices:
    for g in v.groups:
        W[v.index, g.group] = g.weight
R = rig.matrix_world


def seg_dist(n):
    a, b = np.array(R @ bones[n].head_local), np.array(R @ bones[n].tail_local)
    t = np.clip(((co - a) @ (b - a)) / max((b - a) @ (b - a), 1e-12), 0, 1)
    return np.linalg.norm(a + (b - a) * t[:, None] - co, axis=1)


rgba = common.limb_paint(body, skip_loose=True)
for k, v in paint.items():
    cols = [gn.index(n) for n in v["bones"] if n in gn]
    w = W[:, cols].sum(1)
    d = np.min(np.stack([seg_dist(n) for n in v["bones"]]), 0)
    core = w > 0.9
    if core.sum() < 20:
        continue
    far = (d > thickness * np.median(d[core])) & (rgba[:, 3] > 0) & np.all(np.isclose(rgba[:, :3], common.srgb_to_linear(v["color"])), 1)
    rgba[far] = 0
    print(f"limb paint: {k} {int(far.sum())} vertices of carried cloth left unpainted")
body.data.color_attributes["legpaint"].data.foreach_set("color", rgba.ravel())
out.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(out), copy=True)
print("limb paint:", {k: v["bones"] for k, v in paint.items()}, "->", out)
