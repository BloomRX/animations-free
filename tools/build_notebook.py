#!/usr/bin/env python3
"""Builds notebooks/ai-text-to-motion-colab.ipynb from cell sources."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = open(os.path.join(HERE, "ai_mocap_lib.py"), encoding="utf-8").read()

CELLS = []

def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src})

# ═══════════════════════════════════════════════════════════════════
md(r"""# 🎭 Texto → Animação 3D → UE5  (Maid Cat Cafe)

Gere animações de personagem a partir de texto (ex.: *"a maid serves tea gracefully"*) e baixe
arquivos prontos para **Unreal Engine 5** (IK Retargeter → Manny). Tudo roda no **Colab gratuito (T4)**.

## Os 4 passos
1. **CONFIG** (célula 1): escolha o modelo desta sessão + escreva o prompt.
2. **SETUP** (célula 3): instala dependências + baixa os pesos — **com cache** (só na 1ª vez).
3. **GENERATE** (célula 4): gera as variações da animação.
4. **EXPORT** (célula 5): pré-visualização 3D + download de **GLB / FBX(UE5) / BVH / NPZ**.

## Modelos (todos gratuitos)
| Modelo | Qualidades | VRAM (T4) | Licença |
|---|---|---|---|
| ⭐ **Kimodo SOMA-RP v1.1** (NVIDIA) | Melhor qualidade; **77 joints (dedos!)**; realista | <3 GB (encoder de texto no CPU) | NVIDIA Open + Apache 2.0 ✅ comercial |
| **HY-Motion 1.0 Lite** (Tencent) | Muito rápido; bom p/ dança e movimentos casuais (200+ categorias) | ~16 GB (cabe no T4) | Tencent Community License ✅ comercial (<1M MAU; Brasil ok) |
| **MoMask** (CVPR 2024) | Segue bem instruções precisas (contar passos, direções) | ~8 GB | MIT ✅ comercial |

## Como funciona a sessão
- **1 modelo por sessão** (de propósito: os pesos ficam na memória). Para trocar de modelo:
  `Runtime → Restart runtime` (e o SETUP roda de novo, usando o cache local/Drive).
- Rodar **novamente o mesmo modelo NÃO re-baixa nada** — o CONFIG (célula 1) avisa
  "✅ já carregado nesta sessão".
- Marque **"Google Drive"** no CONFIG para o cache de pesos sobreviver a resets do Colab grátis.

## Saída para o UE
- `*_ue.fbx` — ossos **já batizados com os nomes do Manny** (pelvis, spine_01, thigh_l, thumb_01_l…) →
  IK Retargeter mapeia quase tudo automaticamente. Veja o passo a passo na última célula.
- `.glb` — mesma animação para Blender/preview. `.bvh` — para Blender/Mixamo. `.npy` — dados brutos.
""")

# ═══════════════════════════════════════════════════════════════════
code(r"""# ════════════════════════ CONFIG ════════════════════════
# 1) Escolha o MODELO desta sessão (para trocar: Runtime -> Restart)
# 2) Escreva o PROMPT (em inglês, descrevendo o movimento)
# 3) Rode esta célula -> depois rode a célula SETUP (1a vez) e GENERATE

import ipywidgets as widgets
import datetime

MODELS = {
    "kimodo":   "Kimodo SOMA-RP v1.1 (NVIDIA) - MELHOR QUALIDADE, tem dedos, <3GB VRAM",
    "hymotion": "HY-Motion 1.0 Lite (Tencent) - rapido, bom p/ danca e movimentos casuais",
    "momask":   "MoMask (CVPR 2024) - segue bem instrucoes precisas",
}

STATE = {"model": None, "loaded": False, "weights_ready": False, "motions": []}

_model_dd   = widgets.Dropdown(options=list(MODELS), value="kimodo",
                               description="MODELO (sessao)", layout=widgets.Layout(width="700px"))
_prompt_tb  = widgets.Textarea(
    value="A cute maid bows politely, then picks up a teacup and serves it with both hands",
    description="PROMPT (em ingles - descreva o movimento)", rows=3,
    layout=widgets.Layout(width="700px"))
_row1 = widgets.HBox([
    widgets.IntSlider(value=5, min=2, max=10, step=1, description="Duracao (s)"),
    widgets.IntSlider(value=2, min=1, max=4, description="Variacoes"),
    widgets.IntText(value=42, description="Seed"),
])
_row2 = widgets.HBox([
    widgets.Dropdown(options=["ue5_manny", "original"], value="ue5_manny",
                     description="Ossos no arquivo", layout=widgets.Layout(width="320px")),
    widgets.Dropdown(options=["floor", "origin", "inplace"], value="floor",
                     description="Modo do root (UE)", layout=widgets.Layout(width="300px")),
    widgets.FloatText(value=0.95, description="Altura do pelvis (m)",
                      layout=widgets.Layout(width="260px")),
])
_drive_cb = widgets.Checkbox(value=False,
                             description="Manter cache de pesos no Google Drive (recomendado - nao precisa remountar)")

CFG = {}

def _run_config():
    CFG.update({
        "model": _model_dd.value,
        "prompt": _prompt_tb.value.strip(),
        "duration": int(_row1.children[0].value),
        "samples": int(_row1.children[1].value),
        "seed": int(_row1.children[2].value),
        "skeleton": _row2.children[0].value,
        "root_mode": _row2.children[1].value,
        "pelvis_h": float(_row2.children[2].value),
        "use_drive": _drive_cb.value,
        "_ok": True,
    })
    print("=" * 74)
    print(f"MODELO (esta sessao): {CFG['model']}")
    print(f"                     {MODELS[CFG['model']]}")
    print(f"PROMPT:      {CFG['prompt']}")
    print(f"DURACAO:     {CFG['duration']}s | VARIACOES: {CFG['samples']} | SEED: {CFG['seed']}")
    print(f"OSSOS:       {CFG['skeleton']}  (ue5_manny = ossos batizados p/ Manny)")
    print(f"ROOT (UE):   {CFG['root_mode']}  (floor = pelvis a {CFG['pelvis_h']}m do chao)")
    print("-" * 74)
    if STATE["loaded"] and STATE["model"] == CFG["model"]:
        print(f"  {CFG['model']} JA CARREGADO nesta sessao.")
        print("  -> NAO vai baixar nada de novo. Rode direto a celula GENERATE (celula 4).")
    elif STATE["model"] is not None and STATE["model"] != CFG["model"]:
        print(f"  ATT: a sessao esta presa ao modelo '{STATE['model']}'.")
        print("  Para trocar: Runtime -> Restart runtime (os pesos ficam em cache/Drive).")
    else:
        print("  1o uso deste modelo na sessao: rode a celula SETUP (celula 3)")
        print("  (instala dependencias + baixa pesos - com cache, uma unica vez).")
    print("=" * 74)

_run_config()
display(widgets.VBox([_model_dd, _prompt_tb, _row1, _row2, _drive_cb]))""")

# ═══════════════════════════════════════════════════════════════════
code("# ════════════════════════ BIBLIOTECA ════════════════════════\n"
     "# Converters SMPL/SOMA -> GLB / FBX(UE5) / BVH + FK + preview.\n"
     "# (codigo fixo - rode como esta)\n\n" + LIB)

# ═══════════════════════════════════════════════════════════════════
code(r"""# ════════════════════════ SETUP (env + pesos) ════════════════════════
# Rode uma vez por modelo/sessao. Usa cache local (/content) e opcional (Drive):
#   2a vez no mesmo Colab        -> nada baixa (cache local)
#   apos restart do runtime      -> nada baixa (cache local sobrevive em /content? NAO:
#                                     /content limpa no restart) -> baixa do Drive
#   outro notebook (mesma conta) -> baixa do Drive
import os, sys, shutil, subprocess, glob, time

if not CFG.get("_ok"):
    raise RuntimeError("Rode a celula CONFIG (celula 1) primeiro.")
MODEL = CFG["model"]

CACHE = "/content/ai_mocap_cache"
REPO  = "/content/ai_mocap"
os.makedirs(CACHE, exist_ok=True); os.makedirs(REPO, exist_ok=True)
DRIVE = os.path.expanduser("~/drive/MyDrive/ai_mocap_cache")
USE_DRIVE = CFG.get("use_drive", False) and os.path.isdir(os.path.expanduser("~/drive/MyDrive"))
if CFG.get("use_drive") and not USE_DRIVE:
    print("ATT: Google Drive nao montado -> cache so local. (Runtime -> Mount Drive, se quiser)")

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (vai demorar!)")

def run(cmd):
    print("$ " + cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise RuntimeError(f"Falhou o comando: {cmd}")

def get_repo(name, git_url):
    local = f"{REPO}/{name}"
    if os.path.isdir(local) and os.listdir(local):
        print(f"  OK repositorio em cache: {local}")
        return local
    d = f"{DRIVE}/{name}" if USE_DRIVE else None
    if d and os.path.isdir(d) and os.listdir(d):
        print(f"  <- Do Drive: {d}"); shutil.copytree(d, local)
    else:
        print(f"  <- Baixando {name}...")
        run(f"git clone -q --depth 1 {git_url} {local}")
        if USE_DRIVE:
            os.makedirs(DRIVE, exist_ok=True); shutil.copytree(local, d, dirs_exist_ok=True)
    return local

def get_weights(name, test, download_cmd, drive_sub=None):
    local = f"{CACHE}/{name}"
    if os.path.isdir(local) and test(local):
        print(f"  OK pesos em cache: {local}")
        return local
    d = f"{DRIVE}/{name}" if USE_DRIVE else None
    if d and os.path.isdir(d) and test(d):
        print(f"  <- Pesos do Drive: {d}"); shutil.copytree(d, local)
    else:
        print(f"  <- Baixando pesos de {name} (pode demorar)...")
        run(download_cmd.format(CACHE=CACHE, name=name))
        if USE_DRIVE:
            os.makedirs(DRIVE, exist_ok=True); shutil.copytree(local, f"{DRIVE}/{name}", dirs_exist_ok=True)
    return local

t0 = time.time()
if MODEL == "kimodo":
    repo = get_repo("kimodo", "https://github.com/nv-tlabs/kimodo")
    print("  Instalando pacotes do Kimodo...")
    run(f"pip install -q {repo}")
    os.makedirs(f"{CACHE}/hf", exist_ok=True)   # cache dos pesos NVIDIA (auto-download na 1a generacao)
    STATE.update(model=MODEL, weights_ready=True)

elif MODEL == "hymotion":
    repo = get_repo("hymotion", "https://github.com/Tencent-Hunyuan/HY-Motion-1.0")
    print("  Instalando dependencias do HY-Motion...")
    run(f"pip install -q -r {repo}/requirements.txt || echo 'alguns pacotes falharam (ok se torch ja existe)'")
    get_weights(
        "hymotion_ckpt",
        test=lambda p: os.path.exists(f"{p}/HY-Motion-1.0-Lite/config.yml"),
        download_cmd=(
            "huggingface-cli download tencent/HY-Motion-1.0 --include 'HY-Motion-1.0-Lite/*' "
            "--local-dir {CACHE}/{name}"
        ),
    )
    STATE.update(model=MODEL, weights_ready=True)

elif MODEL == "momask":
    repo = get_repo("momask", "https://github.com/EricGuo5513/momask-codes")
    print("  Instalando CLIP (encoder de texto do MoMask)...")
    run("pip install -q git+https://github.com/openai/CLIP.git")
    get_weights(
        "momask_ckpt",
        test=lambda p: len(glob.glob(f"{p}/**/opt.txt", recursive=True)) > 0,
        download_cmd=(
            "gdown -q --fuzzy 'https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view' "
            "-O {CACHE}/momask_dl.zip && mkdir -p {CACHE}/{name} && "
            "unzip -q {CACHE}/momask_dl.zip -d {CACHE}/{name} && rm {CACHE}/momask_dl.zip"
        ),
    )
    # o MoMask espera: <repo>/checkpoints/t2m/<dataset_name>/... -> faz o link
    ck = f"{CACHE}/momask_ckpt"
    opts = glob.glob(f"{ck}/**/opt.txt", recursive=True)
    if opts:
        sys.path.insert(0, repo)
        from options.eval_option import EvalT2MOptions
        _opt = EvalT2MOptions().parse()
        vq_dir = os.path.dirname(opts[0])              # .../<dataset>/<model>
        ds_dir = os.path.dirname(vq_dir)               # .../<dataset>
        link = f"{repo}/checkpoints/t2m/{_opt.dataset_name}"
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.islink(link):
            os.remove(link)
        elif os.path.exists(link):
            raise RuntimeError(f"{link} ja existe - remova manualmente")
        os.symlink(ds_dir, link)
        print(f"  OK checkpoints linkados: {link} -> {ds_dir}")
    STATE.update(model=MODEL, weights_ready=True)

print(f"SETUP concluido em {time.time()-t0:.0f}s. Modelo: {MODEL}")
print("Agora rode a celula GENERATE (celula 4).")""")

# ═══════════════════════════════════════════════════════════════════
code(r"""# ════════════════════════ GENERATE ════════════════════════
# Gera as variacoes da animacao com o modelo da sessao.
# Se o modelo ja esta carregado: gera direto, SEM re-download.
import os, sys, glob, subprocess, datetime

if not CFG.get("_ok"):
    raise RuntimeError("Rode a celula CONFIG (celula 1) primeiro.")
if STATE["model"] != CFG["model"]:
    if STATE["loaded"]:
        raise RuntimeError(
            f"Esta sessao esta usando '{STATE['model']}'. Para trocar de modelo: "
            "Runtime -> Restart runtime (1 modelo por sessao, de proposito).")
    raise RuntimeError("Rode a celula SETUP (celula 3) primeiro.")

if STATE["loaded"]:
    print(f"  {CFG['model']} ja carregado nesta sessao -> gerando direto (sem re-download)")
else:
    if not STATE["weights_ready"]:
        raise RuntimeError("Rode a celula SETUP (celula 3) primeiro.")
    print(f"  {CFG['model']} carregando pela 1a vez nesta sessao...")

ts = datetime.datetime.now().strftime("%H%M%S")
OUT = f"/content/ai_mocap_out/{ts}"
os.makedirs(OUT, exist_ok=True)
STATE["motions"] = []
MODEL = CFG["model"]

if MODEL == "kimodo":
    env = os.environ.copy()
    env["TEXT_ENCODER_DEVICE"] = "cpu"      # encoder de texto no CPU -> usa <3GB de VRAM na T4
    env["HF_HOME"] = f"{CACHE}/hf"          # cache dos pesos NVIDIA
    stem = f"{OUT}/gen"
    cmd = [sys.executable, "-m", "kimodo.scripts.generate", CFG["prompt"],
           "--model", "Kimodo-SOMA-RP-v1.1",
           "--duration", str(CFG["duration"]),
           "--num_samples", str(CFG["samples"]),
           "--seed", str(CFG["seed"]),
           "--bvh", "--bvh_standard_tpose",
           "--output", stem]
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd, env=env, cwd=f"{REPO}/kimodo")
    if r.returncode != 0:
        raise RuntimeError("Falha na geracao do Kimodo (veja o log acima)")
    STATE["motions"] = [{"model": "kimodo", "bvh": p}
                        for p in sorted(glob.glob(stem + "*.bvh") + glob.glob(stem + "/*.bvh"))]
    print(f"  {len(STATE['motions'])} animacao(oes) gerada(s) -> {OUT}")

elif MODEL == "hymotion":
    sys.path.insert(0, f"{REPO}/hymotion")
    if STATE.get("hymotion_rt") is None:
        from hymotion.utils.t2m_runtime import T2MRuntime
        ck = f"{CACHE}/hymotion_ckpt/HY-Motion-1.0-Lite"
        STATE["hymotion_rt"] = T2MRuntime(
            config_path=f"{ck}/config.yml",
            ckpt_name=f"{ck}/latest.ckpt",
            disable_prompt_engineering=True,   # sem LLM de reescrita/duracao (nao precisa)
        )
        print("  HY-Motion carregado (fica em memoria pelo resto da sessao)")
    seeds = ",".join(str(CFG["seed"] + i) for i in range(CFG["samples"]))
    base = f"{OUT}/gen"
    html, _, _ = STATE["hymotion_rt"].generate_motion(
        text=CFG["prompt"],
        seeds_csv=seeds,
        duration=float(CFG["duration"]),
        cfg_scale=5.0,
        output_format="dict",
        output_dir=OUT,
        output_filename=base,
    )
    STATE["hymotion_html"] = html
    STATE["motions"] = [{"model": "hymotion", "npz": p} for p in sorted(glob.glob(f"{base}_*.npz"))]
    print(f"  {len(STATE['motions'])} animacao(oes) gerada(s) -> {OUT}")

elif MODEL == "momask":
    os.chdir(f"{REPO}/momask")
    sys.path.insert(0, f"{REPO}/momask")
    if STATE.get("momask") is None:
        import numpy as np, torch, torch.nn.functional as F
        from utils.fixseed import fixseed
        from options.eval_option import EvalT2MOptions
        from utils.get_opt import get_opt
        from gen_t2m import load_vq_model, load_res_model, load_trans_model, load_len_estimator

        parser = EvalT2MOptions()
        opt = parser.parse()
        fixseed(CFG["seed"])
        opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dim_pose = 263
        root_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name)
        model_opt = get_opt(os.path.join(root_dir, opt.name, "opt.txt"), device=opt.device)
        vq_opt = get_opt(os.path.join(root_dir, model_opt.vq_name, "opt.txt"), device=opt.device)
        vq_opt.dim_pose = dim_pose
        vq_model, vq_opt = load_vq_model(vq_opt)
        model_opt.num_tokens = vq_opt.nb_code
        model_opt.num_quantizers = vq_opt.num_quantizers
        model_opt.code_dim = vq_opt.code_dim
        res_opt = get_opt(os.path.join(root_dir, model_opt.res_name, "opt.txt"), device=opt.device)
        res_model = load_res_model(res_opt, vq_opt, opt)
        t2m = load_trans_model(model_opt, opt, "latest.tar")
        length_estimator = load_len_estimator(opt)
        for m_ in (t2m, vq_model, res_model, length_estimator):
            m_.eval().to(opt.device)
        mean = np.load(os.path.join(root_dir, model_opt.vq_name, "meta", "mean.npy"))
        std = np.load(os.path.join(root_dir, model_opt.vq_name, "meta", "std.npy"))
        frames = max(8, (int(CFG["duration"]) * 20) // 4 * 4)   # 20 fps, multiplo de 4
        STATE["momask"] = dict(opt=opt, t2m=t2m, vq=vq_model, res=res_model,
                               mean=mean, std=std, frames=frames)
        print(f"  MoMask carregado ({frames} frames @20fps)")
    st = STATE["momask"]
    import numpy as np, torch
    token_lens = torch.tensor([st["frames"] // 4]).to(st["opt"].device)
    prompt_list = [CFG["prompt"]]
    for i in range(CFG["samples"]):
        from utils.fixseed import fixseed
        fixseed(CFG["seed"] + i)
        mids = st["t2m"].generate(prompt_list, token_lens, timesteps=st["opt"].time_steps,
                                  cond_scale=st["opt"].cond_scale,
                                  temperature=st["opt"].temperature,
                                  topk_filter_thres=st["opt"].topkr,
                                  gsample=st["opt"].gumbel_sample)
        mids = st["res"].generate(mids, prompt_list, token_lens, temperature=1, cond_scale=5)
        pred = st["vq"].forward_decoder(mids).detach().cpu().numpy()
        gts = pred * st["std"] + st["mean"]
        STATE["motions"].append({"model": "momask", "gts": gts[0], "fps": 20.0})
    print(f"  {len(STATE['motions'])} animacao(oes) gerada(s)")

STATE["loaded"] = True
if not STATE["motions"]:
    raise RuntimeError("Nenhum arquivo de animacao encontrado - veja o log acima")
print(f"PROMPT: {CFG['prompt']}")
print(f"{len(STATE['motions'])} animacao(oes) prontas -> rode a celula EXPORT (celula 5).")""")

# ═══════════════════════════════════════════════════════════════════
code(r"""# ════════════════════════ EXPORT + PREVIEW + DOWNLOAD ════════════════════════
# Gera GLB / FBX (ossos do Manny) / BVH / NPZ + preview 3D + download
import os, zipfile
import numpy as np
import IPython.display as IP
from google.colab import files

if not STATE.get("loaded") or not STATE["motions"]:
    raise RuntimeError("Rode a celula GENERATE (celula 4) primeiro.")

OUTD = "/content/motion_out"
os.makedirs(OUTD, exist_ok=True)
preview, files_out = [], []

for i, mo in enumerate(STATE["motions"]):
    if mo["model"] == "kimodo":
        b = flatten_motion(read_bvh(mo["bvh"]))
        # BVH do Kimodo (SOMA) vem em centimetros -> detectar e converter p/ metros
        span = float(np.abs(b.local_pos).max())
        scale = 1.0 / 100.0 if span > 5 else 1.0
        if scale != 1.0:
            b.offsets_m = [list(np.array(o, dtype=float) * scale) for o in b.offsets_m]
            b.local_pos = b.local_pos * scale
        # normaliza o root conforme o modo escolhido no CONFIG
        ri = b.root_index
        if CFG["root_mode"] == "floor":
            y0 = b.local_pos[0, ri, 1]
            b.local_pos[:, ri, 1] = CFG["pelvis_h"] + (b.local_pos[:, ri, 1] - y0)
        elif CFG["root_mode"] == "origin":
            b.local_pos[:, ri] = b.local_pos[:, ri] - b.local_pos[0, ri]
        elif CFG["root_mode"] == "inplace":
            b.local_pos[:, ri, 0] = b.local_pos[0, ri, 0]
            b.local_pos[:, ri, 2] = b.local_pos[0, ri, 2]
        mapping = UE5_MAP_SOMA77
    elif mo["model"] == "momask":
        b = gts_to_motion(mo["gts"], root_mode=CFG["root_mode"],
                          target_pelvis_height=CFG["pelvis_h"], fps=mo.get("fps", 20.0))
        mapping = UE5_MAP_SMPL22
    else:  # hymotion
        d = np.load(mo["npz"])
        b = poses_to_motion(d["Rh"], d["poses"], d["trans"],
                            target_pelvis_height=CFG["pelvis_h"], fps=30.0,
                            root_mode=CFG["root_mode"])
        mapping = UE5_MAP_SMPL22

    names = with_ue_names(b, mapping) if CFG["skeleton"] == "ue5_manny" else b.names
    stem = f"{OUTD}/motion_{mo['model']}_{i:02d}"
    write_glb_skeleton(b.names, b.parents, b.local_pos, b.local_quats, b.fps,
                       stem + ".glb", node_names=names, anim_name="motion")
    write_fbx_skeleton(b.names, b.parents, b.local_pos, b.local_quats, b.fps,
                       stem + "_ue.fbx", node_names=names, anim_name="motion")
    write_bvh(b, stem + ".bvh")
    if mo["model"] == "momask":
        np.save(stem + "_raw.npy", mo["gts"])
        files_out.append(stem + "_raw.npy")
    elif mo["model"] == "hymotion":
        d2 = np.load(mo["npz"])
        np.savez(stem + "_raw.npz", **{k: d2[k] for k in d2.files})
        files_out.append(stem + "_raw.npz")
    wp = fk_world_positions(b.local_quats, b.local_pos, b.parents)
    preview.append({"label": f"[{i}] {CFG['prompt'][:52]}",
                    "parents": b.parents, "positions": wp, "fps": b.fps})
    files_out += [stem + ".glb", stem + "_ue.fbx", stem + ".bvh"]

print(f"OSSOS: {CFG['skeleton']} | ROOT: {CFG['root_mode']} | PELVIS: {CFG['pelvis_h']}m")
print("Arquivos:")
for f_ in files_out:
    print("  -", os.path.basename(f_))

# --- Preview 3D (stick figure, arraste p/ girar, barra p/ scrub) ---
IP.display(IP.HTML(build_preview_html(preview, title="AI Motion Preview - Maid Cat Cafe")))

# --- Preview do HY-Motion (HTML oficial, se aplicavel) ---
if STATE.get("hymotion_html"):
    IP.display(IP.HTML(f"<h4>Preview oficial HY-Motion:</h4>{STATE['hymotion_html']}"))

# --- Download (zip com tudo) ---
zpath = "/content/motion_pack.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f_ in files_out:
        z.write(f_)
print("Baixando motion_pack.zip ...")
files.download(zpath)
print("Depois: veja o passo a passo de importacao no UE5 (celula 7).")""")

# ═══════════════════════════════════════════════════════════════════
md(r"""## 🎮 Importando no Unreal Engine 5 (passo a passo)

Os arquivos `*_ue.fbx` já vêm com **ossos batizados com os nomes do Manny**
(`pelvis`, `spine_01..03`, `thigh_l`, `calf_l`, `clavicle_l`, `thumb_01_l` …),
o que deixa o retarget quase automático.

1. **Importar o FBX**
   - Arraste o `motion_xxx_ue.fbx` para o Content Browser.
   - No diálogo de import: deixe **Skeletal Mesh** → **Import** (também marca
     *Import Animation* automaticamente).
   - Isso cria um `SK_*` + uma `AnimSequence` com o esqueleto importado.

2. **Retargetar para o seu Manny (IK Retargeter)**
   - `Tools → Animation → IK Retargeter` (ou *Window → IK Retargeting Tool* em versões recentes).
   - **Source Asset** = a AnimSequence importada · **Target Asset** = seu `SK_Mannequin`
     (ou o esqueleto do seu personagem).
   - Como os nomes já casam, o auto-mapeamento resolve quase tudo; confira a lista
     (pelvis↔pelvis, spine_01↔spine_01, thigh_l↔thigh_l…).
   - **Generate Retargeted Animation** → salve no seu pasta.

3. **Se o personagem flutua / afunda**
   - No CONFIG usei *Root = floor* com pelvis a 0.95 m (altura do Manny). Para um
     personagem mais alto/curto, ajuste *Altura do pelvis (m)* e gere de novo.
   - Alternativas: `origin` (pelvis no Y=0, movimento de raiz fiel) ou
     `inplace` (sem deslocamento — bom para ações fixas, como servir café).

4. **Dedos**
   - Só o **Kimodo (SOMA-77)** anima os dedos (5 dedos × 3-4 ossos, já mapeados
     para `thumb_01_l`, `index_02_r` etc.). HY-Motion e MoMask usam o corpo
     SMPL-22 (mãos em pose neutra).

## ⚠️ Limitações honestas (importante p/ um jogo)
- **Sem objetos**: o modelo não "sabe" onde está a xícara. O braço *finge* pegar.
  No UE, corre com **IK** (pegue a xícara como target da mão) ou use o plugin
  *Epic MetaHuman Animator* (markerless) para capturar a cena real com objetos.
- **Sem looping perfeito**: cada geração é única; para loops, gere e faça fade
  (Blend Space) ou alinhe início/fim manualmente.
- **Animais (gatos)**: nenhum desses modelos gera gato. Fica pra depois
  (keyframes no UE/Cascadeur ou outra solução).
- **Qualidade**: se a variação não ficou boa, mude o **seed** e rode GENERATE
  de novo (o modelo já está carregado — é rápido).

## 💡 Dicas de prompt (em inglês)
- Seja específico e físico: *"a maid tiptoes to the counter and pours tea slowly"*.
- Um movimento por vez funciona melhor que uma sequência longa.
- Para o cafe: "serves", "pours", "bows", "curtsies", "dusts the table",
  "holds a plate with both hands", "steps sideways gracefully".

## 🔄 Rotina de trabalho sugerida
1. CONFIG com 2-4 variações → GENERATE → EXPORT.
2. Teste o preview; gostou? Importe o FBX no UE e retargete.
3. Não gostou? Troque o seed (mesmo modelo, sem download) e gere de novo.
4. Quer outro modelo? `Runtime → Restart`, CONFIG com o outro modelo.
""")

# ═══════════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "cells": CELLS,
}
out_path = os.path.join(os.path.dirname(HERE), "notebooks", "ai-text-to-motion-colab.ipynb")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("written:", out_path, "| cells:", len(CELLS))

# sanity: json valid
json.load(open(out_path))
print("JSON valid")
