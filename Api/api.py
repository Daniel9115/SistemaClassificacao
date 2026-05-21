"""
=============================================================
  Classificador de Pratos — API FastAPI
=============================================================
Equivalente completa ao script classificar_prato.py,
exposta via HTTP para uso com o frontend HTML.

Endpoints:
  GET  /status                     → status da API e se modelo existe
  POST /treinar                    → treina/retreina com as imagens do banco
  POST /classify                   → classifica imagem enviada (multipart)
  GET  /categorias                 → lista categorias e imagens do banco
  POST /categorias                 → cria nova categoria
  DELETE /categorias/{nome}        → remove categoria
  POST /categorias/{nome}/fotos    → adiciona foto (upload) a uma categoria
  DELETE /categorias/{nome}/fotos/{arquivo}  → remove foto específica
  POST /categorias/{nome}/itens            → cria subpasta (item) dentro de categoria
  DELETE /categorias/{nome}/itens/{item}   → remove subpasta/item
  POST /categorias/{nome}/itens/{item}/fotos → adiciona foto a um item

Uso:
  pip install fastapi uvicorn tensorflow opencv-python numpy python-multipart
  python api.py
  → http://localhost:8000
  → http://localhost:8000/docs  (Swagger UI)
=============================================================
"""

import os
import sys
import json
import shutil
import subprocess

# ── Instalar dependências automaticamente ────────────────────────────────────
DEPS = {
    "fastapi":     "fastapi",
    "uvicorn":     "uvicorn[standard]",
    "tensorflow":  "tensorflow",
    "cv2":         "opencv-python",
    "numpy":       "numpy",
    "multipart":   "python-multipart",
}
for modulo, pacote in DEPS.items():
    try:
        __import__(modulo)
    except ImportError:
        print(f"Instalando {pacote}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pacote, "-q"])

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import asyncio
from fastapi.responses import PlainTextResponse

# ── Caminhos ─────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
BANCO_DIR     = BASE_DIR / "BancoImagens"
MODELS_DIR    = BASE_DIR / "models"
TFLITE_PATH   = MODELS_DIR / "classificador.tflite"
LABELS_PATH   = MODELS_DIR / "labels.json"
PROTOS_PATH   = MODELS_DIR / "prototipos.npy"
SNAPSHOT_PATH = MODELS_DIR / "banco_snapshot.json"

IMG_SIZE   = 224
LIMIAR_CONF = 0.50
EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BANCO_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Estado global do modelo ───────────────────────────────────────────────────
_base_model    = None
_labels        = None
_prototipos    = None
_treino_em_andamento = False


# =============================================================================
#  CORE ML
# =============================================================================

def _carregar_base():
    global _base_model
    if _base_model is None:
        print("Carregando MobileNetV2...")
        _base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        _base_model.trainable = False
    return _base_model


def _extrair_embedding(base, img_bgr):
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    arr  = preprocess_input(rgb.astype(np.float32))
    arr  = np.expand_dims(arr, 0)
    feat = base.predict(arr, verbose=0)[0]
    return feat / (np.linalg.norm(feat) + 1e-9)


def _augmentar(img_bgr):
    variações = [img_bgr]
    h, w = img_bgr.shape[:2]
    variações.append(cv2.flip(img_bgr, 1))
    variações.append(np.clip(img_bgr.astype(np.float32) * 1.2, 0, 255).astype(np.uint8))
    variações.append(np.clip(img_bgr.astype(np.float32) * 0.8, 0, 255).astype(np.uint8))
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 10, 1.0)
    variações.append(cv2.warpAffine(img_bgr, M, (w, h)))
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -10, 1.0)
    variações.append(cv2.warpAffine(img_bgr, M, (w, h)))
    return variações


def _similaridade_cosseno(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _snapshot_banco():
    snap = {}
    for p in BANCO_DIR.rglob("*"):
        if p.suffix.lower() in EXTS:
            snap[str(p.relative_to(BANCO_DIR))] = p.stat().st_size
    return snap


def _banco_mudou():
    atual = _snapshot_banco()
    if not SNAPSHOT_PATH.exists():
        return True
    with open(SNAPSHOT_PATH, encoding="utf-8") as f:
        anterior = json.load(f)
    return atual != anterior


def _salvar_snapshot():
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(_snapshot_banco(), f, ensure_ascii=False, indent=2)


def _carregar_modelo_em_memoria():
    global _labels, _prototipos
    if not LABELS_PATH.exists() or not PROTOS_PATH.exists():
        return False
    with open(LABELS_PATH, encoding="utf-8") as f:
        _labels = json.load(f)
    _prototipos = np.load(PROTOS_PATH, allow_pickle=False)
    _carregar_base()
    return True


def _predizer(img_bgr):
    if _labels is None or _prototipos is None or _base_model is None:
        raise RuntimeError("Modelo não carregado. Execute /treinar primeiro.")

    emb      = _extrair_embedding(_base_model, img_bgr)
    sims     = [_similaridade_cosseno(emb, _prototipos[i]) for i in range(len(_labels))]
    sims_arr = np.array(sims)
    exp      = np.exp(sims_arr * 10)
    probs    = exp / exp.sum()

    idx_melhor = int(np.argmax(probs))
    label      = _labels[idx_melhor]
    confianca  = float(probs[idx_melhor])

    abaixo_limiar = confianca < LIMIAR_CONF

    partes = label.split("/", 1) if "/" in label else [label, ""]
    classe = "" if abaixo_limiar else partes[0]
    item   = "" if abaixo_limiar else (partes[1] if len(partes) > 1 else "")

    confiancas = {_labels[i]: round(float(probs[i]), 4) for i in range(len(_labels))}
    return {
        "classe":     classe,
        "item":       item,
        "confianca":  0.0 if abaixo_limiar else round(confianca, 4),
        "todas":      confiancas,
        "indefinido": abaixo_limiar,
    }


def _executar_treino():
    global _treino_em_andamento, _labels, _prototipos

    _treino_em_andamento = True
    log = []

    try:
        base = _carregar_base()
        prototipos = []
        labels_lista = []

        entradas = []
        for cat in sorted(BANCO_DIR.iterdir()):
            if not cat.is_dir():
                continue
            subcats = [d for d in cat.iterdir() if d.is_dir()]
            if subcats:
                for sub in sorted(subcats):
                    entradas.append((f"{cat.name}/{sub.name}", sub))
            else:
                entradas.append((cat.name, cat))

        if not entradas:
            return {"ok": False, "erro": "Nenhuma categoria encontrada no banco."}

        for label, cls_path in entradas:
            imagens = [p for p in cls_path.iterdir() if p.suffix.lower() in EXTS]
            if not imagens:
                log.append(f"Pulando '{label}' — sem imagens.")
                continue

            embs = []
            log.append(f"[{label}] {len(imagens)} imagem(ns)")

            for p in imagens:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                amostras = _augmentar(img) if len(imagens) < 5 else [img]
                for amostra in amostras:
                    embs.append(_extrair_embedding(base, amostra))

            if not embs:
                log.append(f"Nenhuma imagem válida para '{label}'")
                continue

            proto = np.mean(embs, axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-9)
            prototipos.append(proto)
            labels_lista.append(label)

        if not prototipos:
            return {"ok": False, "erro": "Nenhum protótipo gerado."}

        prototipos_np = np.array(prototipos, dtype=np.float32)

        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(labels_lista, f, ensure_ascii=False, indent=2)
        np.save(PROTOS_PATH, prototipos_np)

        # Exporta TFLite
        _exportar_tflite(base, labels_lista, prototipos_np)
        _salvar_snapshot()

        # Recarrega em memória
        _labels     = labels_lista
        _prototipos = prototipos_np

        log.append(f"Treino concluído. {len(labels_lista)} classe(s).")
        return {"ok": True, "classes": labels_lista, "log": log}

    except Exception as e:
        return {"ok": False, "erro": str(e), "log": log}
    finally:
        _treino_em_andamento = False


def _exportar_tflite(base, classes, prototipos_np):
    from tensorflow.keras import layers, models, Input

    proto_const = tf.constant(prototipos_np.T, dtype=tf.float32)
    inp    = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x      = layers.Lambda(lambda img: preprocess_input(img * 255.0), name="preprocess")(inp)
    emb    = base(x, training=False)
    emb_n  = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1), name="l2_norm")(emb)
    sims   = layers.Lambda(lambda v: tf.matmul(v, proto_const), name="cosine_sims")(emb_n)
    out    = layers.Softmax(name="probabilidades")(sims)
    modelo = models.Model(inputs=inp, outputs=out, name="classificador_prato")

    converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)


# =============================================================================
#  FASTAPI
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tenta carregar modelo existente na inicialização
    carregado = _carregar_modelo_em_memoria()
    if carregado:
        print(f"Modelo carregado: {len(_labels)} classe(s) → {_labels}")
    else:
        print("Nenhum modelo encontrado. Use POST /treinar para treinar.")
    yield


app = FastAPI(
    title="Classificador de Pratos API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── /status ───────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    modelo_ok = _labels is not None and _prototipos is not None
    return {
        "online":               True,
        "modelo_carregado":     modelo_ok,
        "classes":              _labels if modelo_ok else [],
        "banco_modificado":     _banco_mudou(),
        "treino_em_andamento":  _treino_em_andamento,
    }

# ── /Readme ───────────────────────────────────────────────────────────────────
@app.get("/readme", response_class=PlainTextResponse)
def readme():
    """Retorna a documentação da API em Markdown."""
    readme_path = BASE_DIR / "README.md"
    if not readme_path.exists():
        raise HTTPException(404, "README.md não encontrado.")
    return readme_path.read_text(encoding="utf-8")

# ── /treinar ──────────────────────────────────────────────────────────────────

@app.post("/treinar")
def treinar(background_tasks: BackgroundTasks, sincrono: bool = False):
    """
    Treina o modelo com as imagens do BancoImagens/.
    - sincrono=true  → aguarda conclusão (pode demorar)
    - sincrono=false → retorna imediatamente, treino roda em background
    """
    global _treino_em_andamento

    if _treino_em_andamento:
        raise HTTPException(409, "Treino já em andamento.")

    if sincrono:
        resultado = _executar_treino()
        if not resultado["ok"]:
            raise HTTPException(500, resultado["erro"])
        return resultado
    else:
        background_tasks.add_task(_executar_treino)
        return {"ok": True, "mensagem": "Treino iniciado em background. Consulte GET /status."}


# ── /classify ─────────────────────────────────────────────────────────────────

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """
    Classifica uma imagem enviada via multipart/form-data (campo 'file').
    Retorna: classe, item, confianca, todas (dict com probabilidades).
    """
    if _labels is None or _prototipos is None:
        raise HTTPException(503, "Modelo não carregado. Execute POST /treinar primeiro.")

    conteudo = await file.read()
    arr = np.frombuffer(conteudo, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "Imagem inválida ou formato não suportado.")

    resultado = _predizer(img)
    return resultado


# ── /categorias ───────────────────────────────────────────────────────────────

@app.get("/categorias")
def listar_categorias():
    """Lista todas as categorias, itens e quantidade de fotos."""
    resultado = []
    for cat in sorted(BANCO_DIR.iterdir()):
        if not cat.is_dir():
            continue
        subcats = sorted([d for d in cat.iterdir() if d.is_dir()])
        imgs_diretas = [p.name for p in cat.iterdir() if p.suffix.lower() in EXTS]

        if subcats:
            itens = []
            for sub in subcats:
                fotos = sorted([p.name for p in sub.iterdir() if p.suffix.lower() in EXTS])
                itens.append({"nome": sub.name, "fotos": fotos, "total": len(fotos)})
            resultado.append({
                "nome":   cat.name,
                "itens":  itens,
                "fotos":  [],
                "total":  sum(i["total"] for i in itens),
            })
        else:
            resultado.append({
                "nome":  cat.name,
                "itens": [],
                "fotos": imgs_diretas,
                "total": len(imgs_diretas),
            })
    return resultado


class NomeBody(BaseModel):
    nome: str


@app.post("/categorias")
def criar_categoria(body: NomeBody):
    """Cria uma nova categoria (pasta) no banco."""
    nome = body.nome.strip().lower()
    if not nome:
        raise HTTPException(400, "Nome inválido.")
    pasta = BANCO_DIR / nome
    if pasta.exists():
        raise HTTPException(409, f"Categoria '{nome}' já existe.")
    pasta.mkdir(parents=True)
    return {"ok": True, "categoria": nome}


@app.delete("/categorias/{nome}")
def remover_categoria(nome: str):
    """Remove uma categoria e todas as suas fotos."""
    pasta = BANCO_DIR / nome
    if not pasta.exists():
        raise HTTPException(404, f"Categoria '{nome}' não encontrada.")
    shutil.rmtree(str(pasta))
    return {"ok": True, "removido": nome}


# ── /categorias/{nome}/fotos ──────────────────────────────────────────────────

@app.post("/categorias/{nome}/fotos")
async def adicionar_foto_categoria(nome: str, file: UploadFile = File(...)):
    """Adiciona uma foto diretamente a uma categoria (sem subpasta)."""
    pasta = BANCO_DIR / nome
    if not pasta.exists():
        raise HTTPException(404, f"Categoria '{nome}' não encontrada.")

    ext = Path(file.filename).suffix.lower()
    if ext not in EXTS:
        raise HTTPException(400, f"Formato não suportado: {ext}")

    imgs_atuais  = [p for p in pasta.iterdir() if p.suffix.lower() in EXTS]
    proximo_n    = len(imgs_atuais) + 1
    nome_arquivo = f"{nome}_{proximo_n}{ext}"
    destino      = pasta / nome_arquivo

    conteudo = await file.read()
    with open(destino, "wb") as f:
        f.write(conteudo)

    return {"ok": True, "arquivo": nome_arquivo, "categoria": nome}


@app.delete("/categorias/{nome}/fotos/{arquivo}")
def remover_foto_categoria(nome: str, arquivo: str):
    """Remove uma foto de uma categoria."""
    foto = BANCO_DIR / nome / arquivo
    if not foto.exists():
        raise HTTPException(404, "Foto não encontrada.")
    foto.unlink()
    return {"ok": True, "removido": arquivo}


# ── /categorias/{nome}/itens ──────────────────────────────────────────────────

@app.post("/categorias/{nome}/itens")
def criar_item(nome: str, body: NomeBody):
    """Cria um item (subpasta) dentro de uma categoria."""
    cat = BANCO_DIR / nome
    if not cat.exists():
        raise HTTPException(404, f"Categoria '{nome}' não encontrada.")
    item_nome = body.nome.strip().lower()
    if not item_nome:
        raise HTTPException(400, "Nome inválido.")
    pasta = cat / item_nome
    if pasta.exists():
        raise HTTPException(409, f"Item '{item_nome}' já existe.")
    pasta.mkdir(parents=True)
    return {"ok": True, "categoria": nome, "item": item_nome}


@app.delete("/categorias/{nome}/itens/{item}")
def remover_item(nome: str, item: str):
    """Remove um item (subpasta) e todas as fotos dentro dele."""
    pasta = BANCO_DIR / nome / item
    if not pasta.exists():
        raise HTTPException(404, f"Item '{item}' não encontrado em '{nome}'.")
    shutil.rmtree(str(pasta))
    return {"ok": True, "removido": f"{nome}/{item}"}


# ── /categorias/{nome}/itens/{item}/fotos ────────────────────────────────────

@app.post("/categorias/{nome}/itens/{item}/fotos")
async def adicionar_foto_item(nome: str, item: str, file: UploadFile = File(...)):
    """Adiciona uma foto a um item (subpasta) de uma categoria."""
    pasta = BANCO_DIR / nome / item
    if not pasta.exists():
        raise HTTPException(404, f"Item '{nome}/{item}' não encontrado.")

    ext = Path(file.filename).suffix.lower()
    if ext not in EXTS:
        raise HTTPException(400, f"Formato não suportado: {ext}")

    imgs_atuais  = [p for p in pasta.iterdir() if p.suffix.lower() in EXTS]
    proximo_n    = len(imgs_atuais) + 1
    nome_arquivo = f"{item}_{proximo_n}{ext}"
    destino      = pasta / nome_arquivo

    conteudo = await file.read()
    with open(destino, "wb") as f:
        f.write(conteudo)

    return {"ok": True, "arquivo": nome_arquivo, "categoria": nome, "item": item}


@app.delete("/categorias/{nome}/itens/{item}/fotos/{arquivo}")
def remover_foto_item(nome: str, item: str, arquivo: str):
    """Remove uma foto de um item."""
    foto = BANCO_DIR / nome / item / arquivo
    if not foto.exists():
        raise HTTPException(404, "Foto não encontrada.")
    foto.unlink()
    return {"ok": True, "removido": arquivo}


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)