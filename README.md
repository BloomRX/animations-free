# animations-free

Pipeline **gratuita** de texto → animação 3D → Unreal Engine 5, rodando em **Google Colab (T4 grátis)**.

```
"a maid bows and serves tea"  ──►  Kimodo (NVIDIA)  ──►  BVH  ──►  GLB / FBX (ossos do Manny) / NPZ
```

## Uso

Abra [`notebooks/ai-text-to-motion-colab.ipynb`](notebooks/ai-text-to-motion-colab.ipynb) no Colab
(ou arraste o arquivo para <https://colab.research.google.com>) e rode as células na ordem:

| # | Célula | O que faz |
|---|--------|-----------|
| ① | **CONFIG** | escolha o modelo, escreva o prompt, ajuste duração/variações/seed |
| ② | **biblioteca** | código de exportação (não precisa mexer) |
| ③ | **SETUP** | instala as dependências do modelo (1x por sessão, com cache) |
| ④ | **GENERATE** | gera as variações da animação |
| ⑤ | **EXPORT** | preview 3D + download do **GLB / FBX(UE5) / BVH / NPZ** |

Todas as células de código abrem com o **código escondido** (`cellView: form`): aparecem só os
campos, os logs e o preview. Para inspecionar o código, clique em **Mostrar código**.

## Antes de rodar (modelo Kimodo, o padrão)

O Kimodo usa o **Llama-3-8B** como encoder de texto — modelo *gated* no Hugging Face:

1. Aceite a licença em <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>.
2. Crie um token de leitura em <https://huggingface.co/settings/tokens>.
3. No Colab: 🔑 **Secrets** (barra esquerda) → nome `HF_TOKEN` → valor do token → *Notebook access*.

Na **primeira geração** o encoder (~16 GB) é baixado; depois fica em `/content/ai_mocap_cache/hf`.

## Encoder de texto e memória

O Llama-3-8B em bf16 pesa ~16 GB — não cabe inteiro nem na VRAM (16 GB) nem na RAM (~12,7 GB) do
Colab gratuito. O campo **Encoder de texto** do CONFIG resolve isso:

| Opção | Onde roda | Quando |
|-------|-----------|--------|
| `auto` (padrão) | decide por VRAM/RAM | ≥20 GB VRAM → `gpu` · ≥24 GB RAM → `cpu` · resto → `4bit` |
| `4bit` | GPU, quantizado em NF4 (~6 GB) | **T4 grátis** — o caso comum |
| `gpu` | GPU bf16 (~18 GB) | A100 / L4 / 4090 |
| `cpu` | CPU bf16 (~16 GB RAM, <3 GB VRAM) | Kaggle, Colab Pro |

No modo `4bit` o notebook quantiza o encoder em `bitsandbytes` ao carregar (o Kimodo não expõe
essa opção; o *runner* aplica o patch antes de chamar o CLI).

## Como o notebook é gerado

O `.ipynb` **não** é editado à mão — ele é gerado por `tools/build_notebook.py`:

```bash
python3 tools/build_notebook.py
```

- `tools/ai_mocap_lib.py` — conversores numpy-only (BVH/GLB/FBX + FK + preview 3D), embutidos
  na célula ②.
- `tools/build_notebook.py` — monta o notebook, valida (ids únicos + `compile()` de cada célula)
  e grava em `notebooks/`.

Fluxo: edite `tools/` → rode o build → abra o `.ipynb` no Colab.

## Correções recentes

- **③ SETUP (erro no `pip install` do Kimodo)**: o `setup.py` do Kimodo compila um módulo C++
  (`motion_correction`) via CMake, o que quebrava no Colab. Agora o install usa
  `SKIP_MOTION_CORRECTION_IN_SETUP=1` + dependências em blocos, com um *ladder* de 3 tentativas
  e um teste de `import kimodo`; a geração usa `--no-postprocess` quando o C++ não está disponível
  (marcando o checkbox do CONFIG é possível compilá-lo).
- **Logs**: `run()` mostra o comando ao vivo e, em caso de falha, imprime o fim de
  `/content/ai_mocap_cache/setup.log` em vez de só “falhou o comando”.
- **Modo limpo**: metadados que o Colab realmente respeita (`cellView: form`, `id`,
  `colab.collapsed_sections`) no lugar de `jupyter.source_hidden`, que o Colab ignora.
- **Encoder**: escolha `auto` / `4bit` / `gpu` / `cpu` no CONFIG, com detecção de VRAM/RAM.

## Licenças

- Código deste repo: uso livre.
- **Kimodo** (NVIDIA): NVIDIA Open Model + Apache 2.0 — ✅ uso comercial.
- **HY-Motion 1.0 Lite** (Tencent): Tencent Community License — ✅ comercial (<1M MAU).
- **MoMask**: MIT — ✅ comercial.
