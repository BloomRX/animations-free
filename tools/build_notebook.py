#!/usr/bin/env python3
"""Builds notebooks/ai-text-to-motion-colab.ipynb from cell sources.

Nota importante sobre a "visualizacao limpa" no Colab:
  O Colab *ignora* os metadados clasicos do Jupyter (jupyter.source_hidden /
  jupyter.collapsed / jupyter.outputs_hidden). O que ele respeita e':
    - metadata.cellView = "form"  -> esconde o codigo e mostra so' o resultado
                                     (widgets, banners, preview 3D...);
    - metadata.id                -> id unico da celula;
    - metadata.colab.collapsed_sections (no notebook) -> lista de ids que
                                     comecam recolhidos (so' a barra de titulo).
  Combinado com "#@title <nome>" no topo da celula, isso deixa a mostra apenas
  os campos de texto / dropdown, que e' o objetivo do notebook.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = open(os.path.join(HERE, "ai_mocap_lib.py"), encoding="utf-8").read()

CELLS = []
COLLAPSED = []          # ids que o Colab deve abrir ja recolhidos


def md(src, cid, collapsed=False):
    """Celula de texto. collapsed -> entra em colab.collapsed_sections."""
    meta = {"id": cid}
    if collapsed:
        COLLAPSED.append(cid)
    # o id vai nos 2 lugares: nbformat 4.5 espera no nivel da celula,
    # o Colab le de metadata.id (e usa p/ collapsed_sections).
    CELLS.append({"cell_type": "markdown", "id": cid, "metadata": meta, "source": src})


def code(src, cid, form=True, collapsed=False):
    """Celula de codigo.

    form=True     -> Colab mostra so' o resultado (codigo escondido).
    form=False    -> codigo visivel (usado quando faz sentido inspecionar).
    collapsed=True-> celula comeca recolhida (so' a barra do #@title).
    """
    meta = {"id": cid}
    if form:
        meta["cellView"] = "form"
    if collapsed:
        COLLAPSED.append(cid)
    CELLS.append({"cell_type": "code", "id": cid, "execution_count": None,
                  "metadata": meta, "outputs": [], "source": src})

# ═══════════════════════════════════════════════════════════════════
md(r"""# 🎭 Texto → Animação 3D → UE5

Gere animações de personagem a partir de texto (ex.: *"a maid serves tea gracefully"*) e baixe
arquivos prontos para **Unreal Engine 5** (IK Retargeter → Manny). Roda no **Colab gratuito (T4)**.

## Como usar (4 passos, nesta ordem)

| # | Célula | O que faz |
|---|---|---|
| ① | **CONFIG** | escolha o modelo + escreva o prompt + ajustes (só esta célula tem campos) |
| ② | **biblioteca** | código de exportação (GLB/FBX/BVH) — nunca precisa mexer |
| ③ | **SETUP** | instala as dependências do modelo (1x por sessão, com cache) |
| ④ | **GENERATE** | gera as variações da animação |
| ⑤ | **EXPORT** | preview 3D + download do **GLB / FBX(UE5) / BVH / NPZ** |

> **Modo limpo:** as células de código abrem com o **código escondido** (`cellView: form`) — você vê
> só os campos, os logs e o preview. Para ver o código de qualquer célula, clique em **Mostrar código**.

## ⚠️ Antes de rodar (só para o Kimodo, o modelo padrão)

O Kimodo usa o **Llama-3-8B** como encoder de texto, e esse modelo é *gated* no Hugging Face:

1. Aceite a licença em <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct> (mesma conta do Colab).
2. Crie um token de leitura em <https://huggingface.co/settings/tokens>.
3. No Colab: ícone de **chave** (🔑) na barra esquerda → **Secrets** → nome `HF_TOKEN` → valor = seu token.
   Marque *"Notebook access"*.

Sem isso o SETUP avisa e a geração falha ao baixar o encoder.

**Primeira geração é demorada:** o encoder de texto (Llama-3-8B, ~16 GB) é baixado 1x por runtime.
Depois fica no cache local (`/content/ai_mocap_cache/hf`).

## Encoder de texto: qual escolher (campo no CONFIG)

O Llama-3-8B em bf16 pesa ~16 GB — não cabe inteiro nem na VRAM (16 GB) nem na RAM (≈12,7 GB) do
Colab gratuito. Por isso o padrão é **auto**, que decide assim:

| Situação | Escolha | Onde roda |
|---|---|---|
| GPU com ≥ 20 GB de VRAM (A100/L4/4090…) | `gpu` | bf16 na GPU — o mais rápido |
| RAM ≥ 24 GB (Kaggle, Colab Pro) | `cpu` | bf16 na CPU — ~<3 GB de VRAM |
| **T4 grátis / pouca RAM** (o caso comum) | `4bit` | **quantizado em 4-bit na GPU (~6 GB)** |

Se estourar memória, mude para `4bit` (ou `cpu`), reduza **Variações** e **Duração** e rode ④ de novo.

## Modelos (todos gratuitos)

| Modelo | Qualidades | Memória | Licença |
|---|---|---|---|
| ⭐ **Kimodo SOMA-RP v1.1** (NVIDIA) | Melhor qualidade; **77 joints (dedos!)**; realista | ~6 GB VRAM (encoder 4-bit) | NVIDIA Open Model ✅ comercial |
| **HY-Motion 1.0 Lite** (Tencent) | Muito rápido; bom p/ dança e movimentos casuais | ~16 GB (cabe no T4) | Tencent Community ✅ comercial |
| **MoMask** (CVPR 2024) | Segue bem instruções precisas (contar passos, direções) | ~8 GB | MIT ✅ comercial |

## Como funciona a sessão

- **1 modelo por sessão** (de propósito). Para trocar: `Runtime → Restart runtime`.
- Reexecutar o ③ SETUP **não re-baixa** nada (cache em `/content`, e no Drive se marcado).
- Marque **"Google Drive"** no ① CONFIG para o cache dos repos/pesos sobreviver a resets.

## Saída para o UE

- `*_ue.fbx` — ossos **já batizados com os nomes do Manny** (`pelvis`, `spine_01`, `thigh_l`,
  `thumb_01_l`…) → o IK Retargeter mapeia quase tudo automático.
- `.glb` — mesma animação para Blender/preview. `.bvh` — Blender/Mixamo. `.npy` — dados brutos.

## 🔧 Se o ③ SETUP falhar

- O SETUP mostra o **fim do log** completo (`/content/ai_mocap_cache/setup.log`) em vez de só
  “falhou o comando”.
- O Kimodo é instalado **sem** compilar o pós-processamento em C++ (é o que quebrava o `pip install`
  no Colab, por falta de `cmake`). A geração então roda com `--no-postprocess`.
- Quer o pós-processamento (limpa deslize dos pés)? Marque **"Compilar pos-processamento"** no
  ① CONFIG — o setup instala `cmake`/`eigen`/`pybind11` e compila (~5 min).
""", cid="introMd", collapsed=True)

# ═══════════════════════════════════════════════════════════════════
code(r"""#@title ① CONFIG — modelo + prompt + opções
# ════════════════════════ ① CONFIG ════════════════════════
# 1) Escolha o MODELO desta sessão (para trocar: Runtime -> Restart)
# 2) Escreva o PROMPT (em inglês, descrevendo o movimento)
# 3) Rode esta célula -> depois rode ③ SETUP (1a vez) e ④ GENERATE

import ipywidgets as widgets
import datetime

MODELS = {
    "kimodo":   "Kimodo SOMA-RP v1.1 (NVIDIA) - MELHOR qualidade, 77 joints (dedos!), ~6GB VRAM",
    "hymotion": "HY-Motion 1.0 Lite (Tencent) - rapido, bom p/ danca e movimentos casuais",
    "momask":   "MoMask (CVPR 2024) - segue bem instrucoes precisas",
}

# Onde o encoder de texto do Kimodo (Llama-3-8B) roda. So' afeta o Kimodo.
ENCODERS = {
    "auto": "auto - escolhe sozinho pela VRAM/RAM (recomendado)",
    "4bit": "GPU 4-bit - Llama-3-8B quantizado (~6GB VRAM) - cabe no T4 gratis",
    "gpu":  "GPU bf16 - o mais rapido (precisa de ~20GB de VRAM)",
    "cpu":  "CPU bf16 - precisa de ~20GB de RAM (Kaggle / Colab Pro)",
}

# o CONFIG pode ser re-executado so' p/ mudar o prompt: nao zera o historico.
try:
    STATE
except NameError:
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
_row3 = widgets.HBox([
    widgets.Dropdown(options=list(ENCODERS), value="auto",
                     description="Encoder de texto (Kimodo)",
                     layout=widgets.Layout(width="430px")),
    widgets.Checkbox(value=False,
                     description="Compilar pos-processamento foot-skate (+~5min no setup)",
                     layout=widgets.Layout(width="430px")),
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
        "text_encoder": _row3.children[0].value,
        "compile_postprocess": bool(_row3.children[1].value),
        "_ok": True,
    })
    print("=" * 74)
    print(f"MODELO (esta sessao): {CFG['model']}")
    print(f"                     {MODELS[CFG['model']]}")
    print(f"PROMPT:      {CFG['prompt']}")
    print(f"DURACAO:     {CFG['duration']}s | VARIACOES: {CFG['samples']} | SEED: {CFG['seed']}")
    print(f"OSSOS:       {CFG['skeleton']}  (ue5_manny = ossos batizados p/ Manny)")
    print(f"ROOT (UE):   {CFG['root_mode']}  (floor = pelvis a {CFG['pelvis_h']}m do chao)")
    if CFG["model"] == "kimodo":
        print(f"ENCODER:     {CFG['text_encoder']}  ({ENCODERS[CFG['text_encoder']]})")
        print(f"FOOT-SKATE:  {'vai compilar o C++ no setup' if CFG['compile_postprocess'] else 'desligado (setup rapido)'}")
    print("-" * 74)
    if STATE["loaded"] and STATE["model"] == CFG["model"]:
        print(f"  {CFG['model']} JA PRONTO nesta sessao.")
        print("  -> NAO vai baixar nada de novo. Rode direto a celula ④ GENERATE.")
        if CFG["model"] == "kimodo":
            enc = STATE.get("text_encoder", "auto")
            print(f"  -> encoder de texto: {enc} (recarrega do cache local a cada geracao)")
    elif STATE["model"] is not None and STATE["model"] != CFG["model"]:
        print(f"  ATT: a sessao esta presa ao modelo '{STATE['model']}'.")
        print("  Para trocar: Runtime -> Restart runtime (os pesos ficam em cache/Drive).")
    else:
        print("  1o uso deste modelo na sessao: rode a celula ③ SETUP")
        print("  (instala dependencias + baixa pesos - com cache, uma unica vez).")
    print("=" * 74)

_run_config()
display(widgets.VBox([_model_dd, _prompt_tb, _row1, _row2, _row3, _drive_cb]))""",
     cid="cellConfig")

# ═══════════════════════════════════════════════════════════════════
code("# ════════════════════════ BIBLIOTECA ════════════════════════\n"
     "# Converters SMPL/SOMA -> GLB / FBX(UE5) / BVH + FK + preview.\n"
     "# (codigo fixo - rode como esta)\n\n" + LIB,
     cid="cellLibrary", collapsed=True)

# ═══════════════════════════════════════════════════════════════════
code(r"""#@title ③ SETUP — instala o modelo da sessão (rode 1x)
# ════════════════════════ ③ SETUP (ambiente + pesos) ════════════════════════
# Rode 1x por sessao. Usa cache em /content (e no Drive, se marcado no CONFIG).
import os, sys, shutil, subprocess, glob, time
import torch

if not CFG.get("_ok"):
    raise RuntimeError("Rode a celula ① CONFIG primeiro.")

MODEL = CFG["model"]
CACHE = "/content/ai_mocap_cache"
REPO  = "/content/ai_mocap"
LOG   = f"{CACHE}/setup.log"
os.makedirs(CACHE, exist_ok=True)
os.makedirs(REPO, exist_ok=True)
open(LOG, "a").close()

DRIVE = os.path.expanduser("~/drive/MyDrive/ai_mocap_cache")
USE_DRIVE = CFG.get("use_drive", False) and os.path.isdir(os.path.expanduser("~/drive/MyDrive"))
if CFG.get("use_drive") and not USE_DRIVE:
    print("ATT: Google Drive nao montado -> cache apenas local.")

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (vai demorar!)")


def run(cmd, timeout=None, tail=60):
    'Roda um comando com saida AO VIVO e, se falhar, mostra o fim do log.'
    print("$ " + cmd, flush=True)
    p = subprocess.run(["bash", "-o", "pipefail", "-c", cmd + f" 2>&1 | tee -a {LOG}"],
                       timeout=timeout)
    if p.returncode != 0:
        try:
            txt = open(LOG, errors="replace").read().splitlines()[-tail:]
            print("-" * 70)
            print("\n".join(txt))
            print("-" * 70)
        except Exception:
            pass
        raise RuntimeError(f"FALHOU: {cmd}\n(log completo em {LOG})")
    return True


def pip(specs, label=""):
    'pip install tolerante: tenta o bloco inteiro e, se falhar, um por um.'
    q = " ".join('"%s"' % sp for sp in specs)
    try:
        return run("pip install -q " + q)
    except Exception:
        print("  (bloco '%s' falhou -> tentando 1 a 1)" % (label or q))
        ok = True
        for sp in specs:
            try:
                run('pip install -q "%s"' % sp)
            except Exception as e:
                print("  ATT: nao consegui instalar", sp, "->", str(e)[:160])
                ok = False
        return ok


def can_import(mod):
    'True se o modulo importa num processo novo (testado fora da pasta do repo).'
    kw = {"cwd": "/content"} if os.path.isdir("/content") else {}
    try:
        r = subprocess.run([sys.executable, "-c", "import %s" % mod],
                           capture_output=True, text=True, **kw)
        return r.returncode == 0
    except Exception:
        return False


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


def mem_info():
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    return vram, ram


def hf_token():
    'Pega o token do HF: Secrets do Colab (HF_TOKEN) ou variavel de ambiente.'
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            from google.colab import userdata
            tok = userdata.get("HF_TOKEN")
        except Exception:
            tok = None
    return tok


# dependencias do Kimodo em blocos (o resolvedor do pip trabalha melhor assim)
KIMODO_CORE = ["transformers==5.1.0", "peft>=0.18", "accelerate>=0.30",
               "safetensors", "huggingface_hub", "tokenizers"]
KIMODO_CFG  = ["hydra-core>=1.3", "omegaconf>=2.3", "einops>=0.7", "tqdm>=4.0",
               "packaging>=21.0", "pydantic>=2.0", "filelock>=3.20.3"]
KIMODO_MISC = ["trimesh>=3.21.7", "pillow>=9.0", "bvhio", "scipy>=1.10"]

VRAM, RAM = mem_info()
print(f"  VRAM: {VRAM:.1f} GB | RAM: {RAM:.1f} GB")
HF_TOK = hf_token()
os.environ["HF_HOME"] = f"{CACHE}/hf"
os.environ["HUGGINGFACE_CACHE_DIR"] = f"{CACHE}/hf/hub"
os.makedirs(f"{CACHE}/hf", exist_ok=True)

t0 = time.time()
if MODEL == "kimodo":
    # --- token do Hugging Face: o text encoder do Kimodo usa o Llama-3-8B (gated)
    if HF_TOK:
        os.environ["HF_TOKEN"] = HF_TOK
        os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOK
        try:
            from huggingface_hub import login, HfApi
            login(token=HF_TOK, add_to_git_credential=False)
            print("  HF: logado como", HfApi(token=HF_TOK).whoami().get("name", "?"))
        except Exception as e:
            print("  ATT: nao consegui validar o token HF ->", str(e)[:160])
    else:
        print("  ATT: HF_TOKEN nao encontrado.")
        print("       Colab: icone de chave (esquerda) -> Secrets -> nome: HF_TOKEN")
        print("       Sem ele o Kimodo NAO baixa o Llama-3-8B do encoder de texto.")

    repo = get_repo("kimodo", "https://github.com/nv-tlabs/kimodo")
    SKIP = "SKIP_MOTION_CORRECTION_IN_SETUP=1"   # nao compila o C++ do pos-processamento
    ok = False

    if CFG.get("compile_postprocess"):
        print("  Compilando o pos-processamento em C++ (leva alguns minutos)...")
        try:
            run("apt-get -qq update && apt-get -qq install -y cmake libeigen3-dev "
                "pybind11-dev || pip install -q cmake")
            run(f"pip install -q {repo}")
            ok = can_import("kimodo")
        except Exception as e:
            print("  ATT: instalação completa falhou ->", str(e)[:160])

    if not ok:
        print("  Instalando o Kimodo (sem o C++ do pos-processamento: nao precisa de cmake)...")
        try:
            run(f"{SKIP} pip install -q --no-deps {repo}")
            pip(KIMODO_CORE, "core")
            pip(KIMODO_CFG, "config")
            pip(KIMODO_MISC, "misc")
            ok = can_import("kimodo")
        except Exception as e:
            print("  ATT:", str(e)[:160])

    if not ok:
        print("  2a tentativa: deixando o pip resolver as dependencias do proprio Kimodo...")
        try:
            run(f"{SKIP} pip install -q {repo}")
            ok = can_import("kimodo")
        except Exception as e:
            print("  ATT:", str(e)[:160])

    if not ok:
        print("  3a tentativa: instalar cmake/eigen e compilar tudo...")
        run("apt-get -qq update && apt-get -qq install -y cmake libeigen3-dev "
            "pybind11-dev || pip install -q cmake")
        run(f"pip install -q {repo}")
        ok = can_import("kimodo")

    if not ok:
        raise RuntimeError("Nao consegui importar o Kimodo. Log completo em " + LOG)

    has_mc = can_import("motion_correction")
    enc = CFG.get("text_encoder", "auto")
    if enc == "auto":
        enc = "gpu" if VRAM >= 20 else ("cpu" if RAM >= 24 else "4bit")
    if enc in ("gpu", "4bit") and not torch.cuda.is_available():
        print("  ATT: sem GPU -> encoder no CPU")
        enc = "cpu"
    if enc == "cpu" and RAM < 20:
        print("  ATT: encoder no CPU com so %.1f GB de RAM - provavel OOM." % RAM)
        print("       No CONFIG escolha 'Encoder de texto' = 4bit.")
    if enc == "4bit":
        pip(["bitsandbytes"], "bitsandbytes")
    STATE.update(model=MODEL, weights_ready=True, text_encoder=enc,
                 has_motion_correction=has_mc)
    print("  Kimodo instalado.")
    print("  Encoder de texto:", enc, "->", {"4bit": "GPU 4-bit", "gpu": "GPU bf16",
                                             "cpu": "CPU bf16"}[enc])
    print("  Pos-processamento (foot-skate):", "disponivel" if has_mc else "indisponivel (gera com --no-postprocess)")

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
print("Agora rode a celula ④ GENERATE.")""",
     cid="cellSetup")

# ═══════════════════════════════════════════════════════════════════
code(r"""#@title ④ GENERATE — gerar as animações
# ════════════════════════ ④ GENERATE ════════════════════════
# Gera as variacoes da animacao com o modelo da sessao.
import os, sys, glob, subprocess, datetime, shlex

if not CFG.get("_ok"):
    raise RuntimeError("Rode a celula ① CONFIG primeiro.")
if STATE["model"] != CFG["model"]:
    if STATE["loaded"]:
        raise RuntimeError(
            f"Esta sessao esta usando '{STATE['model']}'. Para trocar de modelo: "
            "Runtime -> Restart runtime (1 modelo por sessao, de proposito).")
    raise RuntimeError("Rode a celula ③ SETUP primeiro.")

if STATE["loaded"]:
    print(f"  {CFG['model']} ja configurado nesta sessao -> gerando (sem baixar nada)")
else:
    if not STATE["weights_ready"]:
        raise RuntimeError("Rode a celula ③ SETUP primeiro.")
    print(f"  {CFG['model']}: 1a geracao desta sessao (o encoder recarrega do cache local)")

KIMODO_RUNNER = r'''# -*- coding: utf-8 -*-
# Chama o CLI do Kimodo aplicando (se preciso) o ajuste do encoder de texto.
# Roda como subprocesso: se estourar a memoria, o kernel do Colab sobrevive.
import os, sys

mode = os.environ.get("KM_ENCODER", "cpu")

if mode == "4bit":
    import torch
    from transformers import BitsAndBytesConfig
    from kimodo.model.llm2vec.llm2vec import LLM2Vec

    _orig_from_pretrained = LLM2Vec.from_pretrained.__func__

    def _from_pretrained_4bit(cls, *args, **kwargs):
        kwargs.setdefault("quantization_config", BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ))
        kwargs.setdefault("device_map", {"": 0})
        print("[kimodo] text encoder: Llama-3-8B em 4-bit (NF4) na GPU", flush=True)
        return _orig_from_pretrained(cls, *args, **kwargs)

    LLM2Vec.from_pretrained = classmethod(_from_pretrained_4bit)

    # rede de seguranca: alguns builds reclamam ao mover um modelo quantizado
    _orig_to = LLM2Vec.to

    def _to(self, device, *a, **k):
        try:
            return _orig_to(self, device, *a, **k)
        except Exception as e:
            print("[kimodo] ignorando .to(%s): %s" % (device, e), flush=True)
            return self

    LLM2Vec.to = _to

sys.argv = ["kimodo.scripts.generate"] + sys.argv[1:]
from kimodo.scripts.generate import main
main()
'''


ts = datetime.datetime.now().strftime("%H%M%S")
OUT = f"/content/ai_mocap_out/{ts}"
os.makedirs(OUT, exist_ok=True)
STATE["motions"] = []
MODEL = CFG["model"]

if MODEL == "kimodo":
    enc = STATE.get("text_encoder", "cpu")
    RUNNER = f"{CACHE}/kimodo_run.py"
    with open(RUNNER, "w", encoding="utf-8") as f:
        f.write(KIMODO_RUNNER)

    env = os.environ.copy()
    env.update({
        "HF_HOME": f"{CACHE}/hf",
        "HF_HUB_CACHE": f"{CACHE}/hf/hub",
        "HUGGINGFACE_CACHE_DIR": f"{CACHE}/hf/hub",
        "TRANSFORMERS_CACHE": f"{CACHE}/hf/transformers",
        "TEXT_ENCODER_MODE": "local",                 # nao procura o servico de encoder
        "TEXT_ENCODER_DEVICE": "cuda" if enc in ("gpu", "4bit") else "cpu",
        "KM_ENCODER": enc,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
    })
    tok = globals().get("HF_TOK") or os.environ.get("HF_TOKEN")
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGINGFACE_HUB_TOKEN"] = tok

    stem = f"{OUT}/gen"
    args = [sys.executable, RUNNER, CFG["prompt"],
            "--model", "Kimodo-SOMA-RP-v1.1",
            "--duration", str(CFG["duration"]),
            "--num_samples", str(CFG["samples"]),
            "--seed", str(CFG["seed"]),
            "--bvh", "--bvh_standard_tpose",
            "--output", stem]
    if not STATE.get("has_motion_correction"):
        args.append("--no-postprocess")   # o C++ do pos-processamento nao foi compilado
    cmd = " ".join(shlex.quote(a) for a in args)
    print("$ " + cmd, flush=True)

    genlog = f"{OUT}/generate.log"
    p = subprocess.run(["bash", "-o", "pipefail", "-c", cmd + f" 2>&1 | tee {genlog}"],
                       env=env, cwd="/content")
    if p.returncode != 0:
        txt = ""
        try:
            txt = open(genlog, errors="replace").read().lower()
        except Exception:
            pass
        if "out of memory" in txt or "cuda oom" in txt or "killed" in txt:
            print("  DICA: faltou memoria. No CONFIG tente 'Encoder de texto' = 4bit,")
            print("        menos variações / duração menor, ou um runtime com mais VRAM.")
        if ("401" in txt or "403" in txt or "gated" in txt
                or "restricted" in txt or "not authorized" in txt):
            print("  DICA: sem acesso ao modelo do encoder de texto (Llama-3-8B).")
            print("        Aceite a licenca em https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
            print("        e confira o Secret HF_TOKEN do Colab.")
        raise RuntimeError(f"Falha na geracao do Kimodo (log: {genlog})")

    STATE["motions"] = [{"model": "kimodo", "bvh": bp}
                        for bp in sorted(glob.glob(stem + "*.bvh") + glob.glob(stem + "/*.bvh"))]
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
print(f"{len(STATE['motions'])} animacao(oes) prontas -> rode a celula ⑤ EXPORT.")""",
     cid="cellGenerate")

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
print("Depois: veja o passo a passo de importacao no UE5 na ultima celula.")""",
     cid="cellExport")

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
""", cid="unrealGuideMd", collapsed=True)

# ═══════════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        # metadados que o proprio Colab usa p/ abrir o notebook no "modo limpo"
        "colab": {
            "provenance": [],
            "toc_visible": True,
            "collapsed_sections": COLLAPSED,
        },
    },
    "cells": CELLS,
}
# sanity 1: ids unicos
_ids = [c["id"] for c in CELLS]
assert len(_ids) == len(set(_ids)), "ids de celula duplicados: %s" % _ids

# sanity 2: toda celula de codigo compila
for _c in CELLS:
    if _c["cell_type"] == "code":
        compile(_c["source"], _c["metadata"]["id"], "exec")

out_path = os.path.join(os.path.dirname(HERE), "notebooks", "ai-text-to-motion-colab.ipynb")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# sanity 3: json valido + volta legivel
json.load(open(out_path))
print("written:", out_path, "| cells:", len(CELLS), "| collapsed:", COLLAPSED)
