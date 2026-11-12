"""Print each hand's and hand socket's rest axes as JSON (rigging/kimodo_posture.py --blade-up aims what a hand holds).

blender -b rigged.blend -P rigging/blender/socket_axes.py
    -> one line: SOCKET_AXES {"R": {"hand_y": [...], "socket_y": [...], "socket_z": [...]}, "L": {...}}  (Blender's axes, at rest)
"""
import json

import bpy

rig = bpy.data.objects["rig"]
out = {}
for s in ("L", "R"):
    h, k = rig.data.bones.get(f"hand_{s}"), rig.data.bones.get(f"socket_hand_{s}")
    if h and k:
        m = k.matrix_local.to_3x3()
        out[s] = {"hand_y": list(h.matrix_local.to_3x3().col[1]), "socket_y": list(m.col[1]), "socket_z": list(m.col[2])}
print("SOCKET_AXES", json.dumps(out))
