# -*- coding: utf-8 -*-
"""
ai_mocap_lib — Text-to-Motion -> GLB / FBX(UE5) / BVH exporters.
Only dependency: numpy.
"""
import json
import math
import re
import struct
import numpy as np

# =====================================================================
# 1. MATH
# =====================================================================
def _aa_to_mat_single(aa):
    aa = np.asarray(aa, dtype=np.float64)
    n = np.linalg.norm(aa)
    if n < 1e-12:
        return np.eye(3)
    a = aa / n
    t = n
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)

def aa_to_mat(aa):
    aa = np.asarray(aa, dtype=np.float64)
    orig = aa.shape
    single = orig == (3,)
    aa = aa.reshape(-1, 3)
    out = np.stack([_aa_to_mat_single(a) for a in aa])
    return out[0] if single else out.reshape(orig[:-1] + (3, 3))

def _mat_to_quat_single(R):
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)

def mat_to_quat(R):
    R = np.asarray(R, dtype=np.float64)
    orig = R.shape
    single = orig == (3, 3)
    R = R.reshape(-1, 3, 3)
    out = np.stack([_mat_to_quat_single(m) for m in R])
    return out[0] if single else out.reshape(orig[:-2] + (4,))

def quat_to_mat(q):
    q = np.asarray(q, dtype=np.float64)
    single = q.shape == (4,)
    q = q.reshape(-1, 4)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n[n == 0] = 1.0
    q = q / n
    x, y, z, w = q.T
    out = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)
    return out[0] if single else out.reshape(q.shape[:-1] + (3, 3))

def normalize_quats(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return q / n

# =====================================================================
# 2. SKELETONS
# =====================================================================
SMPL22_NAMES = ["pelvis", "left_hip", "right_hip", "spine_1", "left_knee",
                "right_knee", "spine_2", "left_ankle", "right_ankle",
                "spine_3", "left_foot", "right_foot", "neck_1", "head",
                "left_collar", "right_collar", "left_shoulder",
                "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
                "right_wrist"]
SMPL22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SMPL22_OFFSETS_M = [
    [0.0, 0.0, 0.0],
    [0.069520, -0.091406, -0.006815],
    [-0.067670, -0.090522, -0.004320],
    [-0.002533, 0.108963, -0.026696],
    [0.034277, -0.375199, -0.004496],
    [-0.038290, -0.382569, -0.008850],
    [0.005487, 0.135180, 0.001092],
    [-0.013596, -0.397960, -0.043693],
    [0.015774, -0.398415, -0.042312],
    [0.001457, 0.052922, 0.025425],
    [0.026358, -0.055791, 0.119288],
    [-0.025372, -0.048144, 0.123348],
    [-0.002778, 0.213870, -0.042857],
    [0.078845, 0.121749, -0.034090],
    [-0.081759, 0.118833, -0.038615],
    [0.005152, 0.064970, 0.051349],
    [0.090977, 0.030469, -0.008868],
    [-0.096012, 0.032551, -0.009143],
    [0.259612, -0.012772, -0.027456],
    [-0.253742, -0.013329, -0.021401],
    [0.249234, 0.008986, -0.001171],
    [-0.255298, 0.007772, -0.005559],
]
SMPL22_ENDS = [("L_wrist_end", 20, [0.084042, -0.008162, -0.014945]),
               ("R_wrist_end", 21, [-0.084622, -0.006117, -0.010315])]

UE5_MAP_SMPL22 = {
    "pelvis": "pelvis",
    "left_hip": "thigh_l", "right_hip": "thigh_r",
    "left_knee": "calf_l", "right_knee": "calf_r",
    "left_ankle": "foot_l", "right_ankle": "foot_r",
    "left_foot": "ball_l", "right_foot": "ball_r",
    "spine_1": "spine_01", "spine_2": "spine_02", "spine_3": "spine_03",
    "neck_1": "neck_01", "head": "head",
    "left_collar": "clavicle_l", "right_collar": "clavicle_r",
    "left_shoulder": "upperarm_l", "right_shoulder": "upperarm_r",
    "left_elbow": "lowerarm_l", "right_elbow": "lowerarm_r",
    "left_wrist": "hand_l", "right_wrist": "hand_r",
}
UE5_MAP_SOMA77 = {
    "Hips": "pelvis",
    "Spine1": "spine_01", "Spine2": "spine_02", "Chest": "spine_03",
    "Neck1": "neck_01", "Neck2": "neck_02", "Head": "head",
    "LeftShoulder": "clavicle_l", "RightShoulder": "clavicle_r",
    "LeftArm": "upperarm_l", "RightArm": "upperarm_r",
    "LeftForeArm": "lowerarm_l", "RightForeArm": "lowerarm_r",
    "LeftHand": "hand_l", "RightHand": "hand_r",
    "LeftLeg": "thigh_l", "RightLeg": "thigh_r",
    "LeftShin": "calf_l", "RightShin": "calf_r",
    "LeftFoot": "foot_l", "RightFoot": "foot_r",
    "LeftToeBase": "ball_l", "RightToeBase": "ball_r",
    "LeftHandThumb1": "thumb_01_l", "LeftHandThumb2": "thumb_02_l", "LeftHandThumb3": "thumb_03_l",
    "RightHandThumb1": "thumb_01_r", "RightHandThumb2": "thumb_02_r", "RightHandThumb3": "thumb_03_r",
    "LeftHandIndex1": "index_01_l", "LeftHandIndex2": "index_02_l", "LeftHandIndex3": "index_03_l", "LeftHandIndex4": "index_04_l",
    "RightHandIndex1": "index_01_r", "RightHandIndex2": "index_02_r", "RightHandIndex3": "index_03_r", "RightHandIndex4": "index_04_r",
    "LeftHandMiddle1": "middle_01_l", "LeftHandMiddle2": "middle_02_l", "LeftHandMiddle3": "middle_03_l", "LeftHandMiddle4": "middle_04_l",
    "RightHandMiddle1": "middle_01_r", "RightHandMiddle2": "middle_02_r", "RightHandMiddle3": "middle_03_r", "RightHandMiddle4": "middle_04_r",
    "LeftHandRing1": "ring_01_l", "LeftHandRing2": "ring_02_l", "LeftHandRing3": "ring_03_l", "LeftHandRing4": "ring_04_l",
    "RightHandRing1": "ring_01_r", "RightHandRing2": "ring_02_r", "RightHandRing3": "ring_03_r", "RightHandRing4": "ring_04_r",
    "LeftHandPinky1": "pinky_01_l", "LeftHandPinky2": "pinky_02_l", "LeftHandPinky3": "pinky_03_l", "LeftHandPinky4": "pinky_04_l",
    "RightHandPinky1": "pinky_01_r", "RightHandPinky2": "pinky_02_r", "RightHandPinky3": "pinky_03_r", "RightHandPinky4": "pinky_04_r",
}

def smpl22_skeleton(with_ends=False):
    names = list(SMPL22_NAMES)
    parents = list(SMPL22_PARENTS)
    offsets = [list(o) for o in SMPL22_OFFSETS_M]
    if with_ends:
        for nm, p, off in SMPL22_ENDS:
            names.append(nm)
            parents.append(p)
            offsets.append(off)
    return names, parents, offsets

# =====================================================================
# 3. BVH READER / WRITER
# =====================================================================
class BvhData:
    """Motion on an arbitrary BVH skeleton.
    local_pos (T,J,3): animated local position (== rest offset for static joints)
    local_quats (T,J,4): local rotations xyzw
    has_pos: which joints carry position channels
    """
    def __init__(self, names, parents, offsets_m, local_quats, local_pos, has_pos, fps):
        self.names = names
        self.parents = parents
        self.offsets_m = offsets_m
        self.local_quats = local_quats
        self.local_pos = local_pos
        self.has_pos = has_pos
        self.fps = fps

    @property
    def n_frames(self):
        return self.local_quats.shape[0]

    @property
    def root_index(self):
        return self.parents.index(-1)

    def world_fk(self):
        """Returns (world_pos (T,J,3), world_rot (T,J,3,3))."""
        T, J = self.local_quats.shape[:2]
        world_pos = np.zeros((T, J, 3), dtype=np.float64)
        world_rot = np.zeros((T, J, 3, 3), dtype=np.float64)
        ridx = self.root_index
        for t in range(T):
            world_pos[t, ridx] = self.local_pos[t, ridx]
            world_rot[t, ridx] = quat_to_mat(self.local_quats[t, ridx])
            for j in range(1, J):
                p = self.parents[j]
                if p < 0:
                    continue
                world_rot[t, j] = world_rot[t, p] @ quat_to_mat(self.local_quats[t, j])
                world_pos[t, j] = world_pos[t, p] + (world_rot[t, p] @ self.local_pos[t, j])
        return world_pos, world_rot

def flatten_motion(b: BvhData):
    """Normalize the root: if a static wrapper root has a child with position
    channels (e.g. Kimodo 'Root' wrapper -> 'Hips'), make that child the root.
    Returns a (possibly new) BvhData whose root carries the world motion."""
    ridx = b.root_index
    moved = ridx
    kids = [c for c in range(len(b.names)) if b.parents[c] == ridx]
    if b.has_pos[ridx] and b.n_frames > 1:
        rstd = float(np.std(b.local_pos[:, ridx, :]))
        for c in kids:
            if b.has_pos[c] and rstd <= 1e-6 and float(np.std(b.local_pos[:, c, :])) > 1e-6:
                moved = c
                break
    elif not b.has_pos[ridx]:
        for c in kids:
            if b.has_pos[c]:
                moved = c
                break
    if moved == ridx:
        return b
    keep = [moved]
    def add(k):
        for c in range(len(b.names)):
            if b.parents[c] == k:
                keep.append(c)
                add(c)
    add(moved)
    old2new = {old_j: new_j for new_j, old_j in enumerate(keep)}
    names = [b.names[j] for j in keep]
    offsets = [b.offsets_m[j] for j in keep]
    has_pos = [b.has_pos[j] for j in keep]
    parents = [old2new.get(b.parents[j], -1) for j in keep]
    local_quats = b.local_quats[:, keep, :]
    local_pos = b.local_pos[:, keep, :]
    return BvhData(names, parents, offsets, local_quats, local_pos, has_pos, b.fps)

def _euler_zyx_mat(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def _quat_to_euler_zyx(q):
    R = quat_to_mat(q)
    sy = float(np.clip(-R[2, 0], -1.0, 1.0))
    ry = math.asin(sy)
    if abs(sy) < 0.99999:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return rx, ry, rz

def read_bvh(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.strip() for ln in f.read().splitlines()]
    names, parents, offsets, channels, has_pos = [], [], [], [], []
    stack = []
    in_motion = False
    i = 0
    fps, n_frames, data_rows = 30.0, 0, []
    n = len(lines)
    while i < n:
        s = lines[i]
        if s == "HIERARCHY":
            i += 1
            continue
        if s == "MOVTION":
            in_motion = True
            i += 1
            continue
        if not in_motion:
            if s.startswith("ROOT") or s.startswith("JOINT"):
                jname = s.split()[-1]
                idx = len(names)
                names.append(jname)
                parents.append(stack[-1] if stack else -1)
                offsets.append(None)
                channels.append(None)
                has_pos.append(False)
                stack.append(idx)
            elif s.startswith("OFFSET") and stack:
                if offsets[stack[-1]] is None:
                    offsets[stack[-1]] = [float(v) for v in s.split()[1:4]]
            elif s.startswith("CHANNELS") and stack:
                parts = s.split()
                chn = int(parts[1])
                ch = parts[2:2 + chn]
                channels[stack[-1]] = ch
                has_pos[stack[-1]] = "Xposition" in ch
            elif s == "End Site":
                # skip the end site block
                depth = 0
                started = False
                while i < n:
                    t = lines[i]
                    if t == "{":
                        depth += 1
                        started = True
                    elif t == "}":
                        depth -= 1
                        if started and depth == 0:
                            break
                    i += 1
            elif s == "}":
                if stack:
                    stack.pop()
        else:
            if s.startswith("Frames Per Second"):
                fps = float(s.split(":")[1])
            elif s.startswith("Number of Frames"):
                n_frames = int(s.split(":")[1])
            elif s and re.match(r"^[\d.\-eE+]+( [\d.\-eE+]+)+$", s):
                data_rows.append(s)
        i += 1
    for j in range(len(names)):
        if offsets[j] is None:
            offsets[j] = [0.0, 0.0, 0.0]
        if channels[j] is None:
            channels[j] = []
    data = np.array([[float(v) for v in r.split()] for r in data_rows[:n_frames]], dtype=np.float64)
    T = data.shape[0]
    J = len(names)
    local_quats = np.zeros((T, J, 4), dtype=np.float64)
    local_pos = np.array(offsets, dtype=np.float64).reshape(1, J, 3).repeat(T, axis=0)
    col_of = []
    col = 0
    for ch in channels:
        col_of.append(col)
        col += len(ch)
    for t in range(T):
        for jn in range(J):
            ch = channels[jn]
            if not ch:
                continue
            if "Xposition" in ch:
                c0 = col_of[jn]
                local_pos[t, jn] = (data[t, c0 + ch.index("Xposition")],
                                    data[t, c0 + ch.index("Yposition")],
                                    data[t, c0 + ch.index("Zposition")])
            if "Zrotation" in ch:
                c0 = col_of[jn]
                rz = math.radians(data[t, c0 + ch.index("Zrotation")])
                ry = math.radians(data[t, c0 + ch.index("Yrotation")])
                rx = math.radians(data[t, c0 + ch.index("Xrotation")])
                local_quats[t, jn] = mat_to_quat(_euler_zyx_mat(rx, ry, rz))
    return BvhData(names, parents, offsets, local_quats, local_pos, has_pos, fps)

def write_bvh(b: BvhData, path, unit_scale=1.0):
    out = ["HIERARCHY"]
    ridx = b.root_index
    def emit(j, depth):
        pad = "\t" * (depth + 1)
        kw = "ROOT" if j == ridx else "JOINT"
        out.append(f"{pad}{kw} {b.names[j]}")
        out.append(pad + "{")
        off = b.offsets_m[j]
        out.append(f"{pad}\tOFFSET {off[0]*unit_scale:.6f} {off[1]*unit_scale:.6f} {off[2]*unit_scale:.6f}")
        if b.has_pos[j]:
            out.append(f"{pad}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            out.append(f"{pad}\tCHANNELS 3 Zrotation Yrotation Xrotation")
        kids = [k for k in range(len(b.names)) if b.parents[k] == j]
        for k in kids:
            emit(k, depth + 1)
        if not kids:
            out.append(f"{pad}\tEnd Site")
            out.append(f"{pad}\t\t{{")
            out.append(f"{pad}\t\t\tOFFSET 0.000000 0.000000 0.000000")
            out.append(f"{pad}\t\t}}")
        out.append(pad + "}")
    emit(ridx, 0)
    out.append("}  # HIERARCHY")
    out.append("")
    out.append("MOVTION")
    out.append(f"Frames Per Second: {b.fps:.4f}")
    out.append(f"Number of Frames: {b.n_frames}")
    for t in range(b.n_frames):
        vals = []
        for jn in range(len(b.names)):
            if b.has_pos[jn]:
                vals.extend(float(v) * unit_scale for v in b.local_pos[t, jn])
            rx, ry, rz = _quat_to_euler_zyx(b.local_quats[t, jn])
            vals.extend((math.degrees(rz), math.degrees(ry), math.degrees(rx)))
        out.append(" ".join(f"{v:.6f}" for v in vals))
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")

# =====================================================================
# 4. GLB (skeleton-only glTF 2.0 binary)
# =====================================================================
def _offsets_from_localpos(local_pos, J, T):
    """Static rest offsets: frame-0 local positions (root -> 0)."""
    out = []
    for j in range(J):
        out.append(np.asarray(local_pos[0, j, :], dtype=np.float64))
    return out

def write_glb_skeleton(names, parents, local_pos, local_quats, fps, path,
                       node_names=None, anim_name="motion"):
    J = len(names)
    T = int(local_quats.shape[0])
    if node_names is None:
        node_names = list(names)
    assert len(node_names) == J, "node_names must match skeleton"
    children = [[] for _ in range(J)]
    for j in range(J):
        if parents[j] >= 0:
            children[parents[j]].append(j)
    ridx = parents.index(-1)
    offsets_m = [np.zeros(3) if p < 0 else _off for p, _off in zip(parents, _offsets_from_localpos(local_pos, J, T))]
    dt = 1.0 / max(fps, 1e-6)

    buf = b""
    views, accs = [], []

    def f32view(vals):
        nonlocal buf
        vals = [float(v) for v in vals]
        assert len(buf) % 4 == 0
        data = struct.pack("<%df" % len(vals), *vals)
        views.append({"buffer": 0, "byteOffset": len(buf), "byteLength": len(data)})
        buf += data
        return len(views) - 1

    def accessor(view_idx, type_, count, mn, mx):
        accs.append({"bufferView": view_idx, "componentType": 5126, "count": count,
                     "type": type_, "min": [float(v) for v in mn], "max": [float(v) for v in mx]})
        return len(accs) - 1

    times = [t * dt for t in range(T)]
    a_time = accessor(f32view(times), "SCALAR", T, [times[0]], [times[-1]])

    rtx = np.asarray(local_pos[:, ridx, :], dtype=np.float32)
    a_rtrans = accessor(f32view(rtx.reshape(-1).tolist()), "VEC3", T, rtx.min(0).tolist(), rtx.max(0).tolist())

    qall = normalize_quats(np.asarray(local_quats, dtype=np.float64))
    rot_accs = []
    for j in range(J):
        q = qall[:, j, :]
        rot_accs.append(accessor(f32view(q.reshape(-1).tolist()), "VEC4", T,
                                 q.min(0).tolist(), q.max(0).tolist()))

    offs = np.asarray(local_pos[0, :, :], dtype=np.float32)
    a_offs = accessor(f32view(offs.reshape(-1).tolist()), "VEC3", J, offs.min(0).tolist(), offs.max(0).tolist())

    samplers = [{"input": a_time, "output": a_rtrans, "interpolation": "LINEAR"}]
    for j in range(J):
        samplers.append({"input": a_time, "output": rot_accs[j], "interpolation": "LINEAR"})
    channels = [{"sampler": 0, "target": {"node": ridx, "path": "translation"}}]
    for j in range(J):
        channels.append({"sampler": j + 1, "target": {"node": j, "path": "rotation"}})

    nodes = []
    for j in range(J):
        n = {"name": node_names[j]}
        n["translation"] = [0.0, 0.0, 0.0] if parents[j] < 0 else [float(v) for v in offsets_m[j]]
        n["rotation"] = [0.0, 0.0, 0.0, 1.0]
        if children[j]:
            n["children"] = children[j]
        nodes.append(n)

    gltf = {
        "asset": {"version": "2.0", "generator": "ai-mocap-colab"},
        "scene": 0,
        "scenes": [{"nodes": [ridx]}],
        "nodes": nodes,
        "animations": [{"name": anim_name, "samplers": samplers, "channels": channels}],
        "accessors": accs,
        "bufferViews": views,
        "buffers": [{"byteLength": len(buf)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(buf)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", len(buf), 0x004E4942))
        f.write(buf)
    return path

# =====================================================================
# 5. ASCII FBX (skeleton + animation, UE5 friendly)
# =====================================================================
def write_fbx_skeleton(names, parents, local_pos, local_quats, fps, path,
                       node_names=None, anim_name="motion"):
    J = len(names)
    T = int(local_quats.shape[0])
    if node_names is None:
        node_names = list(names)
    ridx = parents.index(-1)
    fps_ms = 1000.0 / max(fps, 1e-6)
    n_curves = 3 + 4 * J
    offsets_m = [np.zeros(3) if p < 0 else _off for p, _off in zip(parents, _offsets_from_localpos(local_pos, J, T))]
    L = []
    A = L.append
    A("; FBX 7.4.0 project file")
    A("; ai-mocap-colab skeleton+animation export")
    A("")
    A("FBXHeaderExtension:  {")
    A("    FBXHeaderVersion: 1003")
    A("    FBXVersion: 7400")
    A('    Creator: "ai-mocap-colab"')
    A("    SceneInfo:  {")
    A('        Type: "Data"')
    A('        Version: 1000')
    A('        Properties70:  {')
    A('            P: "DocumentUrl", "string", "", "", ""')
    A(f'            P: "FPS", "double", "Number", "", "{fps:.4f}"')
    A('            P: "GlobalStart", "KTime", "Time", "", "0"')
    A('            P: "GlobalEnd", "KTime", "Time", "", "0"')
    A("        }")
    A("    }")
    A("}  # FBXHeaderExtension")
    A("")
    A("GlobalSettings:  {")
    A("    Version: 1000")
    A("    Properties70:  {")
    A('        P: "UpAxis", "int", "Integer", "", "1"')
    A('        P: "UpAxisSign", "int", "Integer", "", "1"')
    A('        P: "FrontAxis", "int", "Integer", "", "2"')
    A('        P: "FrontAxisSign", "int", "Integer", "", "1"')
    A('        P: "CoordAxis", "int", "Integer", "", "0"')
    A('        P: "CoordAxisSign", "int", "Integer", "", "1"')
    A('        P: "OriginalUpAxis", "int", "Integer", "", "1"')
    A('        P: "UnitScaleFactor", "double", "Number", "", "1"')
    A('        P: "OriginalUnitScaleFactor", "double", "Number", "", "1"')
    A("    }")
    A("}  # GlobalSettings")
    A("")
    A("Definitions:  {")
    A("    Version: 100")
    A(f"    Count: {2 + J + n_curves}")
    A('    ObjectType: "GlobalSettings"')
    A('    Object: "Model", "NULL", "Null"')
    for _ in range(J):
        A('    Object: "Model", "NULL", "Null"')
    A('    Object: "AnimationStack", "AnimStack", "AnimationStack"')
    A('    Object: "AnimationLayer", "AnimLayer", "AnimationLayer"')
    for _ in range(n_curves):
        A('    Object: "Curve", "AnimCurve", "Curve"')
    A("}  # Definitions")
    A("")
    A("Objects:  {")
    A(f"    Count: {1 + J + 1 + 1 + n_curves}")
    A('    Model: "Model::RootWrapper", "Null", "" {')
    A("        Version: 232")
    A("        Properties70:  {")
    A('            P: "Lcl Translation", "Lcl Translation", "", "A", 0, 0, 0')
    A('            P: "Lcl Rotation", "Lcl Rotation", "", "A", 0, 0, 0')
    A('            P: "Lcl Scaling", "Lcl Scaling", "", "A", 1, 1, 1')
    A('            P: "DefaultAttributeIndex", "int", "Integer", "", "0"')
    A("        }")
    A("        MultiLayer: 0")
    A("        MultiTake: 0")
    A('        TType: "Null"')
    A("    }  # Model")
    for j in range(J):
        off = offsets_m[j]
        A(f'    Model: "Model::{node_names[j]}", "Null", "" {{')
        A("        Version: 232")
        A("        Properties70:  {")
        A(f'            P: "Lcl Translation", "Lcl Translation", "", "A", {off[0]:.6f}, {off[1]:.6f}, {off[2]:.6f}')
        A('            P: "Lcl Rotation", "Lcl Rotation", "", "A", 0, 0, 0')
        A('            P: "Lcl Scaling", "Lcl Scaling", "", "A", 1, 1, 1')
        A('            P: "DefaultAttributeIndex", "int", "Integer", "", "0"')
        A("        }")
        A("        MultiLayer: 0")
        A("        MultiTake: 0")
        A('        TType: "Null"')
        A("    }  # Model")
    A(f'    AnimationStack: "AnimStack::{anim_name}", "AnimStack", "" {{')
    A('        Attributes: ""')
    A("    }  # AnimationStack")
    A(f'    AnimationLayer: "AnimLayer::{anim_name}", "AnimLayer", "" {{')
    A("    }  # AnimationLayer")

    def emit_curve(label, values_by_frame):
        n = len(values_by_frame)
        times = " ".join(str(int(round(t * fps_ms))) for t in range(n))
        vals = " ".join(f"{v:.8f}" for v in values_by_frame)
        A(f'    Curve: "AnimCurve::{label}", "AnimCurve", "" {{')
        A("        Dimensions: 1")
        A(f"        KeyTime: *{n} {{ {times} }}")
        A(f"        KeyValueFloat: *{n} {{ {vals} }}")
        A("    }  # Curve")

    for c in range(3):
        emit_curve(f"tr_{c}", [float(local_pos[t, ridx, c]) for t in range(T)])
    for j in range(J):
        for c in range(4):
            emit_curve(f"node{j}_rot_{c}", [float(local_quats[t, j, c]) for t in range(T)])
    A("}  # Objects")
    A("")
    A("Connections:  {")
    for j in range(J):
        if parents[j] >= 0:
            A(f"    C: \"OO\", {parents[j] + 1}, {j + 1}, \"\"")
        else:
            A(f"    C: \"OO\", 0, {j + 1}, \"\"")
    stack_id = 1 + J
    layer_id = stack_id + 1
    first_curve = layer_id + 1
    A(f"    C: \"OO\", {stack_id}, {layer_id}, \"\"")
    for j in range(J):
        A(f"    C: \"OO\", {layer_id}, {j + 1}, \"\"")
    ci = 0
    for j in range(J):
        model_id = j + 1
        if j == ridx:
            for c in range(3):
                A(f"    C: \"OO\", {model_id}, {first_curve + ci}, \"\"")
                ci += 1
        for c in range(4):
            A(f"    C: \"OO\", {model_id}, {first_curve + ci}, \"\"")
            ci += 1
    A("}  # Connections")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path

# =====================================================================
# 6. FK + MODEL-OUTPUT CONVERSIONS
# =====================================================================
def fk_world_positions(local_quats, local_pos, parents):
    T, J, _ = local_quats.shape
    world_pos = np.zeros((T, J, 3), dtype=np.float64)
    world_rot = np.zeros((T, J, 3, 3), dtype=np.float64)
    ridx = parents.index(-1)
    for t in range(T):
        world_pos[t, ridx] = local_pos[t, ridx]
        world_rot[t, ridx] = quat_to_mat(local_quats[t, ridx])
        for j in range(1, J):
            p = parents[j]
            if p < 0:
                continue
            world_rot[t, j] = world_rot[t, p] @ quat_to_mat(local_quats[t, j])
            world_pos[t, j] = world_pos[t, p] + (world_rot[t, p] @ local_pos[t, j])
    return world_pos

def gts_to_motion(gts, root_mode="floor", target_pelvis_height=0.95, fps=20.0):
    """MoMask/MLD gts (T,263) -> BvhData on the SMPL-22 skeleton.
    Layout: [yaw_vel1 | vel_xz2 | root_y1 | ric_pos63 | rot6d(21 joints)126 | local_vel66 | foot4]
    Body rotations are cont6d (first two columns of each local rotation matrix).
    """
    gts = np.asarray(gts, dtype=np.float64)
    T = gts.shape[0]
    # root: pure Y-axis rotation from cumulative yaw velocity (MLD stores half-angles)
    half_ang = np.zeros(T)
    if T > 1:
        half_ang[1:] = np.cumsum(gts[:T - 1, 0])
    # recover_root_rot_pos: r_pos[1:, (0,2)] = qrot(qinv(rq), [vx, 0, vz]); cumsum; y = root_y
    r_q = np.zeros((T, 4))  # library convention (x, y, z, w)
    r_q[:, 1] = np.sin(half_ang)
    r_q[:, 3] = np.cos(half_ang)
    vel = np.zeros((T, 3))
    if T > 1:
        vel[1:, 0] = gts[:-1, 1]
        vel[1:, 2] = gts[:-1, 2]
    r_pos = np.zeros((T, 3))
    if T > 1:
        rv = quat_to_mat(r_q)
        rvel_world = np.einsum("tij,tj->ti", np.linalg.inv(rv), vel)
        r_pos[1:] = np.cumsum(rvel_world[1:], axis=0)
    r_pos[:, 1] = gts[:, 3]
    # body: cont6d -> local rotation matrices -> quats
    local_quats = np.zeros((T, 22, 4), dtype=np.float64)
    local_quats[:, 0] = r_q
    rot6 = gts[:, 67:193]
    for j in range(21):
        c1 = rot6[:, j * 6:j * 6 + 3]
        c2 = rot6[:, j * 6 + 3:j * 6 + 6]
        c3 = np.cross(c1, c2)
        c1n = c1 / np.maximum(np.linalg.norm(c1, axis=1, keepdims=True), 1e-12)
        c2n = c2 - np.einsum("ij,ij->i", c2, c1n)[:, None] * c1n
        c2n = c2n / np.maximum(np.linalg.norm(c2n, axis=1, keepdims=True), 1e-12)
        R = np.stack([c1n, c2n, np.cross(c1n, c2n)], axis=2)
        local_quats[:, j + 1] = mat_to_quat(R)
    local_quats = normalize_quats(local_quats)
    names, parents, offsets = smpl22_skeleton(with_ends=False)
    local_pos = np.array(offsets, dtype=np.float64).reshape(1, 22, 3).repeat(T, axis=0)
    if root_mode == "floor":
        ry = r_pos[:, 1]
        local_pos[:, 0, 1] = target_pelvis_height + (ry - ry[0])
        local_pos[:, 0, 0] = r_pos[:, 0]
        local_pos[:, 0, 2] = r_pos[:, 2]
    elif root_mode == "inplace":
        local_pos[:, 0, :] = np.stack([np.zeros(T), np.full(T, target_pelvis_height), np.zeros(T)], axis=1)
    else:  # origin: frame 0 como referência
        local_pos[:, 0, :] = r_pos - r_pos[0]
    has_pos = [False] * 22
    has_pos[0] = True
    return BvhData(names, parents, offsets, local_quats, local_pos, has_pos, fps)

def poses_to_motion(Rh, poses, trans, target_pelvis_height=0.95, fps=30.0, root_mode="floor"):
    """HY-Motion smpl_data: Rh (T,3,3)|(T,9), poses (T, >=63*3) aa, trans (T,3).
    Body = first 22 joints (root + 21)."""
    Rh = np.asarray(Rh, dtype=np.float64)
    root_quat = mat_to_quat(Rh.reshape(-1, 3, 3))
    T = root_quat.shape[0]
    poses = np.asarray(poses, dtype=np.float64)
    body_aa = poses.reshape(T, -1, 3)[:, :21, :]
    trans = np.asarray(trans, dtype=np.float64).reshape(T, 3)
    local_quats = np.zeros((T, 22, 4), dtype=np.float64)
    local_quats[:, 0] = root_quat
    local_quats[:, 1:] = mat_to_quat(aa_to_mat(body_aa))
    local_quats = normalize_quats(local_quats)
    names, parents, offsets = smpl22_skeleton(with_ends=False)
    local_pos = np.array(offsets, dtype=np.float64).reshape(1, 22, 3).repeat(T, axis=0)
    if root_mode == "floor":
        local_pos[:, 0] = np.stack([trans[:, 0], target_pelvis_height + (trans[:, 1] - trans[0, 1]), trans[:, 2]], axis=1)
    elif root_mode == "inplace":
        local_pos[:, 0] = np.stack([np.zeros(T), np.full(T, target_pelvis_height), np.zeros(T)], axis=1)
    else:  # origin: frame 0 como referência
        local_pos[:, 0] = trans - trans[0]
    has_pos = [False] * 22
    has_pos[0] = True
    return BvhData(names, parents, offsets, local_quats, local_pos, has_pos, fps)

def with_ue_names(b: BvhData, mapping):
    return [mapping.get(n, n) for n in b.names]

# =====================================================================
# 7. PREVIEW HTML (three.js stick figure)
# =====================================================================
def build_preview_html(samples, title="AI Motion Preview"):
    data = []
    for s in samples:
        P = np.asarray(s["positions"]).reshape(-1, 3)
        data.append({
            "label": s["label"],
            "parents": s["parents"],
            "T": len(P),
            "fps": s.get("fps", 30.0),
            "pos": [[round(float(v), 4) for v in row] for row in P],
        })
    payload = json.dumps(data)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{margin:0;background:#101018;font-family:sans-serif;color:#eee}}
#hud{{position:fixed;top:10px;left:10px;z-index:10;background:#1c1c28cc;padding:10px 14px;border-radius:10px;font-size:13px;line-height:1.6}}
#ctrl{{position:fixed;bottom:10px;left:50%;transform:translateX(-50%);z-index:10;background:#1c1c28cc;padding:10px 14px;border-radius:10px;display:flex;gap:10px;align-items:center}}
button{{background:#3a3a55;color:#eee;border:0;border-radius:6px;padding:6px 12px;cursor:pointer}}
input[type=range]{{width:260px}}</style></head><body>
<div id="hud"><b>{title}</b><br><span id="lbl"></span></div>
<div id="ctrl">
 <button id="play">&#9208; Pause</button>
 <input id="scrub" type="range" min="0" max="100" value="0">
 <span id="tinfo"></span>
 <select id="sample"></select>
</div>
<script type="importmap">{{"imports":{{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}}}</script>
<script type="module">
import * as THREE from 'three';
import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';
const DATA = {payload};
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101018);
const cam = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 100);
cam.position.set(1.6, 1.4, 2.4);
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(cam, renderer.domElement);
controls.target.set(0, 0.9, 0);
scene.add(new THREE.GridHelper(6, 12, 0x444466, 0x2a2a3a));
scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const dir = new THREE.DirectionalLight(0xffffff, 1.2); dir.position.set(2,4,3); scene.add(dir);
let cur = 0, frame = 0, playing = true;
const sel = document.getElementById('sample');
DATA.forEach((d,i)=>{{ const o=document.createElement('option'); o.value=i; o.textContent=d.label; sel.appendChild(o); }});
sel.onchange = ()=>{{ cur=+sel.value; frame=0; rebuild(); syncUI(); }};
function rebuild(){{
  const old = scene.getObjectByName('fig'); if (old) scene.remove(old);
  const d = DATA[cur];
  const g = new THREE.Group(); g.name='fig';
  const mat = new THREE.LineBasicMaterial({{color:0x66ccff}});
  const segs = [];
  for (let j=1;j<d.parents.length;j++) if (d.parents[j]>=0) segs.push([d.parents[j], j]);
  const lines = [];
  for (const [a,b] of segs){{
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6),3));
    const line = new THREE.Line(geo, mat); g.add(line); lines.push(line);
  }}
  const sphereGeo = new THREE.SphereGeometry(0.032, 12, 12);
  const joints = [];
  for (let j=0;j<d.parents.length;j++){{
    const m = new THREE.MeshStandardMaterial({{color: j===0?0xffcc44:0x66ff99}});
    const s = new THREE.Mesh(sphereGeo, m); g.add(s); joints.push(s);
  }}
  g._lines = lines; g._joints = joints; g._segs = segs;
  scene.add(g);
}}
function setFrame(f){{
  const d = DATA[cur];
  frame = Math.max(0, Math.min(d.T-1, f|0));
  const g = scene.getObjectByName('fig'); if(!g) return;
  const off = frame*d.pos.length;
  for (let j=0;j<d.parents.length;j++){{
    const p = d.pos[off + j*3];
    g._joints[j].position.set(p[0], p[1], p[2]);
  }}
  g._lines.forEach((ln,i)=>{{
    const a = d.pos[off + g._segs[i][0]*3], b = d.pos[off + g._segs[i][1]*3];
    const arr = ln.geometry.attributes.position;
    arr.setXYZ(0, a[0],a[1],a[2]); arr.setXYZ(1, b[0],b[1],b[2]);
    arr.needsUpdate = true;
  }});
  document.getElementById('scrub').value = frame;
  document.getElementById('tinfo').textContent = (frame+1)+' / '+d.T+'  ('+(frame/d.fps).toFixed(1)+'s)';
  document.getElementById('lbl').textContent = d.label;
}}
rebuild(); syncUI();
function syncUI(){{ setFrame(frame); }}
let last = performance.now();
function tick(now){{
  requestAnimationFrame(tick);
  const d = DATA[cur];
  if (playing){{
    frame += (now-last)/1000*d.fps;
    if (frame >= d.T) frame = 0;
    setFrame(frame);
  }}
  last = now;
  renderer.render(scene, cam);
}}
requestAnimationFrame(tick);
document.getElementById('play').onclick = ()=>{{ playing=!playing; document.getElementById('play').textContent = playing?'\\u23F8 Pause':'\\u25B6 Play'; }};
document.getElementById('scrub').oninput = (e)=>{{ playing=false; document.getElementById('play').textContent='\\u25B6 Play'; setFrame(+e.target.value); }};
addEventListener('resize', ()=>{{ cam.aspect=innerWidth/innerHeight; cam.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); }});
</script></body></html>"""
