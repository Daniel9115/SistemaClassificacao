"""
=============================================================
  Classificador de Pratos — Abordagem por Protótipos
=============================================================
Funciona com POUCAS imagens (até 1 por categoria!).
Não precisa de treino tradicional: usa MobileNetV2 pré-treinado
para extrair embeddings e classifica por similaridade de cosseno.

Estrutura de pastas:
  BancoImagens/
    japonesa/   <- fotos de comida japonesa (mínimo: 1)
    chinesa/    <- fotos de comida chinesa  (mínimo: 1)
    almoco/     <- fotos de almoço          (mínimo: 1)
    sobremesa/  <- fotos de sobremesa       (mínimo: 1)

Uso:
  1. Instalar dependências:
       pip install tensorflow opencv-python numpy

  2. Rodar o programa:
       python classificar_prato.py

  3. No menu interativo:
       1 → Treinar com as imagens do banco
       2 → Câmera ao vivo (retreina automaticamente se banco mudou)
       3 → Gerenciar banco (adicionar/remover categorias e fotos)
       9 → Ver ID desta instalação
       0 → Sair
=============================================================
"""

import os
import sys
import json
import subprocess

# ── Instalar dependências automaticamente ─────────────────────────────────────
DEPS = {
    "tensorflow": "tensorflow",
    "cv2":        "opencv-python",
    "numpy":      "numpy",
}
for modulo, pacote in DEPS.items():
    try:
        __import__(modulo)
    except ImportError:
        print(f"Instalando {pacote}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pacote, "-q"])

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

# ── Caminho base (funciona tanto em .py quanto em .exe) ───────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).parent



# ── Configurações ─────────────────────────────────────────────────────────────
BANCO_DIR    = str(BASE_DIR / "BancoImagens")
MODELS_DIR   = str(BASE_DIR / "models")
TFLITE_PATH  = str(BASE_DIR / "models" / "classificador.tflite")
LABELS_PATH  = str(BASE_DIR / "models" / "labels.json")
PROTOS_PATH  = str(BASE_DIR / "models" / "prototipos.npy")
SNAPSHOT_PATH = str(BASE_DIR / "models" / "banco_snapshot.json")
IMG_SIZE      = 224
LIMIAR_CONF  = 0.50
EXTS         = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Cores para exibição ───────────────────────────────────────────────────────
COR_OK       = (80, 220, 100)
COR_INCERTO  = (0, 165, 255)
COR_FUNDO    = (15, 15, 15)
COR_BARRA    = (50, 50, 50)
COR_TEXTO    = (230, 230, 230)
ALTURA_TOPO  = 90
ALTURA_BARRA = 26
MARGEM       = 12

os.makedirs(MODELS_DIR, exist_ok=True)


# =============================================================================
#  UTILITÁRIOS
# =============================================================================

def carregar_base():
    print("Carregando MobileNetV2 (pode demorar na primeira vez)...")
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False
    return base


def extrair_embedding(base, img_bgr):
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    arr  = preprocess_input(rgb.astype(np.float32))
    arr  = np.expand_dims(arr, 0)
    feat = base.predict(arr, verbose=0)[0]
    return feat / (np.linalg.norm(feat) + 1e-9)


def augmentar(img_bgr):
    variações = [img_bgr]
    h, w = img_bgr.shape[:2]
    variações.append(cv2.flip(img_bgr, 1))
    bright = np.clip(img_bgr.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
    variações.append(bright)
    dark = np.clip(img_bgr.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    variações.append(dark)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 10, 1.0)
    variações.append(cv2.warpAffine(img_bgr, M, (w, h)))
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -10, 1.0)
    variações.append(cv2.warpAffine(img_bgr, M, (w, h)))
    return variações


def similaridade_cosseno(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# =============================================================================
#  SNAPSHOT — detecta mudanças no banco
# =============================================================================

def _snapshot_banco():
    """Retorna dict {caminho_relativo: tamanho} de todas as imagens em banco/."""
    snap = {}
    for p in Path(BANCO_DIR).rglob("*"):
        if p.suffix.lower() in EXTS:
            snap[str(p.relative_to(BANCO_DIR))] = p.stat().st_size
    return snap


def _banco_mudou():
    atual = _snapshot_banco()
    if not Path(SNAPSHOT_PATH).exists():
        return True
    with open(SNAPSHOT_PATH, encoding="utf-8") as f:
        anterior = json.load(f)
    return atual != anterior


def _salvar_snapshot():
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(_snapshot_banco(), f, ensure_ascii=False, indent=2)


def treinar(silencioso=False):
    banco_path = Path(BANCO_DIR)
    banco_path.mkdir(parents=True, exist_ok=True)

    # Cria apenas pasta de exemplo se BancoImagens estiver vazio
    subpastas = [d for d in banco_path.iterdir() if d.is_dir()]
    if not subpastas:
        (banco_path / "almoco_exemplo").mkdir()
        print("\n============================================")
        print(" Pasta de exemplo criada:")
        print(f" {(banco_path / 'almoco_exemplo').resolve()}")
        print("\n Crie suas proprias pastas dentro de:")
        print(f" {banco_path.resolve()}")
        print(" Ex: BancoImagens\\almoco\\  BancoImagens\\japonesa\\")
        print(" Nome padrao das imagens: almoco_1.jpg, almoco_2.jpg ...")
        print("============================================")
        input(" Adicione as pastas e imagens e pressione ENTER...")

    base = carregar_base()
    prototipos = []
    stats = {}
    labels_lista = []

    # Coleta entradas: (label, path) — suporta 1 e 2 níveis
    entradas = []
    for cat in sorted(banco_path.iterdir()):
        if not cat.is_dir():
            continue
        subcats = [d for d in cat.iterdir() if d.is_dir()]
        if subcats:
            for sub in sorted(subcats):
                entradas.append((f"{cat.name}/{sub.name}", sub))
        else:
            entradas.append((cat.name, cat))

    for label, cls_path in entradas:
        imagens = [p for p in cls_path.iterdir() if p.suffix.lower() in EXTS]

        if not imagens:
            print(f"  Pulando '{label}' — sem imagens.")
            continue

        embs = []
        print(f"\n  [{label}] — {len(imagens)} imagem(ns) encontrada(s)")

        for p in imagens:
            img = cv2.imread(str(p))
            if img is None:
                print(f"    Ignorando (não lido): {p.name}")
                continue
            amostras = augmentar(img) if len(imagens) < 5 else [img]
            for amostra in amostras:
                embs.append(extrair_embedding(base, amostra))
            print(f"    ✓ {p.name} ({len(amostras)} amostras geradas por augmentation)")

        if not embs:
            print(f"  AVISO: Nenhuma imagem válida para '{label}'")
            continue

        proto = np.mean(embs, axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-9)
        prototipos.append(proto)
        labels_lista.append(label)
        stats[label] = len(embs)

    if not prototipos:
        print("\nERRO: Nenhum protótipo gerado. Verifique as imagens.")
        sys.exit(1)

    prototipos_np = np.array(prototipos, dtype=np.float32)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_lista, f, ensure_ascii=False, indent=2)
    np.save(PROTOS_PATH, prototipos_np)

    print(f"\n{'='*50}")
    print("Indexação concluída!")
    for cls, n in stats.items():
        print(f"  {cls:20s}  {n} embeddings")
    print(f"\nArquivos salvos em '{MODELS_DIR}/':")
    print(f"  labels.json       → {labels_lista}")
    print(f"  prototipos.npy    → shape {prototipos_np.shape}")

    exportar_tflite(base, labels_lista, prototipos_np)
    _salvar_snapshot()
    print("\nPronto!")


# =============================================================================
#  EXPORTAR TFLITE
# =============================================================================

def exportar_tflite(base, classes, prototipos_np):
    from tensorflow.keras import layers, models, Input

    print("\nExportando modelo TFLite...")
    proto_const = tf.constant(prototipos_np.T, dtype=tf.float32)

    inp   = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x     = layers.Lambda(lambda img: preprocess_input(img * 255.0), name="preprocess")(inp)
    emb   = base(x, training=False)
    emb_n = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1), name="l2_norm")(emb)
    sims  = layers.Lambda(lambda v: tf.matmul(v, proto_const), name="cosine_sims")(emb_n)
    out   = layers.Softmax(name="probabilidades")(sims)

    modelo = models.Model(inputs=inp, outputs=out, name="classificador_prato")

    converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    kb = os.path.getsize(TFLITE_PATH) / 1024
    print(f"  classificador.tflite  ({kb:.0f} KB) — pronto para mobile/outras linguagens")


# =============================================================================
#  CLASSIFICAR UMA IMAGEM
# =============================================================================

def classificar_imagem(caminho_imagem: str):
    labels, prototipos_np, base = _carregar_modelo()
    img = cv2.imread(caminho_imagem)
    if img is None:
        print(f"ERRO: Não foi possível ler '{caminho_imagem}'")
        sys.exit(1)
    resultado, confiancas = _predizer(base, labels, prototipos_np, img)
    _exibir_resultado_terminal(resultado, confiancas)
    _mostrar_imagem_com_resultado(img, resultado, confiancas)


def _predizer(base, labels, prototipos_np, img_bgr):
    emb      = extrair_embedding(base, img_bgr)
    sims     = [similaridade_cosseno(emb, prototipos_np[i]) for i in range(len(labels))]
    sims_arr = np.array(sims)
    exp      = np.exp(sims_arr * 10)
    probs    = exp / exp.sum()

    idx_melhor = int(np.argmax(probs))
    label      = labels[idx_melhor]
    confianca  = float(probs[idx_melhor])

    if confianca < LIMIAR_CONF:
        label = "indefinido"

    # Separa categoria e item se for 2 níveis
    partes = label.split("/", 1) if "/" in label else [label, ""]
    classe = partes[0]
    item   = partes[1] if len(partes) > 1 else ""

    confiancas = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return {"classe": classe, "item": item, "confianca": confianca}, confiancas

def _carregar_modelo():
    if not Path(LABELS_PATH).exists() or not Path(PROTOS_PATH).exists():
        print("ERRO: Modelo não encontrado. Rode primeiro:  --modo treinar")
        sys.exit(1)
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    prototipos_np = np.load(PROTOS_PATH, allow_pickle=False)
    base = carregar_base()
    return labels, prototipos_np, base


def _exibir_resultado_terminal(resultado, confiancas):
    classe    = resultado['classe']
    confianca = resultado['confianca']
    item      = resultado.get('item', '')
    icone     = "✓" if confianca >= LIMIAR_CONF else "?"
    label_str = f"{classe.upper()} / {item.upper()}" if item else classe.upper()
    print(f"\n{'─'*42}")
    print(f"  {icone}  {label_str:<30s}  {confianca*100:.1f}%")
    print(f"{'─'*42}")
    for cls, prob in sorted(confiancas.items(), key=lambda x: -x[1]):
        filled = int(prob * 28)
        barra  = "▓" * filled + "░" * (28 - filled)
        marca  = "◀" if cls == classe else " "
        print(f"  {cls:15s} {barra} {prob*100:5.1f}% {marca}")
    print(f"{'─'*42}")


def _desenhar_rodape(display, hint="Q=sair  ESPACO=classificar"):
    """Instruções à esquerda e marca d'água à direita na parte inferior."""
    h, w = display.shape[:2]

    # fundo semitransparente
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - 36), (w, h), COR_FUNDO, -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    # instruções — esquerda
    cv2.putText(display, hint,
                (MARGEM, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # marca d'água — direita
    wm1 = "Developed by Daniel Bitencourt"
    wm2 = "github.com/Daniel9115"
    (tw1, _), _ = cv2.getTextSize(wm1, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    (tw2, _), _ = cv2.getTextSize(wm2, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
    cv2.putText(display, wm1, (w - tw1 - MARGEM, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
    cv2.putText(display, wm2, (w - tw2 - MARGEM, h - 6),  cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)


def _desenhar_overlay(display, resultado, confiancas):
    h, w      = display.shape[:2]
    classe    = resultado["classe"]
    confianca = resultado["confianca"]
    cor       = COR_OK if confianca >= LIMIAR_CONF else COR_INCERTO

    # ── Painel superior ──────────────────────────────────────────────────────
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, ALTURA_TOPO), COR_FUNDO, -1)
    cv2.addWeighted(overlay, 0.82, display, 0.18, 0, display)
    cv2.rectangle(display, (0, 0), (w, 4), cor, -1)

    item  = resultado.get("item", "")
    icone = "OK" if confianca >= LIMIAR_CONF else "??"
    cv2.putText(display, icone,
                (MARGEM, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, cor, 2)
    cv2.putText(display, classe.upper(),
                (MARGEM + 48, 42), cv2.FONT_HERSHEY_DUPLEX, 1.2, cor, 2)
    if item:
        cv2.putText(display, item.upper(),
                    (MARGEM + 48, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
    else:
        cv2.putText(display, f"{confianca*100:.1f}%",
                    (MARGEM + 48, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_TEXTO, 1)


def _mostrar_imagem_com_resultado(img, resultado, confiancas):
    try:
        display = img.copy()
        _desenhar_overlay(display, resultado, confiancas)
        _desenhar_rodape(display, hint="pressione qualquer tecla")
        cv2.imshow("Classificador de Prato", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass


# =============================================================================
#  MODO CÂMERA AO VIVO
# =============================================================================

def modo_camera():
    labels, prototipos_np, base = _carregar_modelo()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERRO: Câmera não encontrada.")
        sys.exit(1)

    print("\nCâmera iniciada. Pressione:")
    print("  ESPAÇO → classificar frame atual")
    print("  Q      → sair\n")

    cv2.namedWindow("Classificador de Prato", cv2.WINDOW_AUTOSIZE)

    ultimo_resultado  = None
    ultimo_confiancas = {}
    frame_count       = 0
    INTERVALO_AUTO    = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_flip = cv2.flip(frame, 1)
        display = frame_flip.copy()
        h, w    = display.shape[:2]

        if frame_count == 1:
            try:
                import ctypes
                user32 = ctypes.windll.user32
                sw = user32.GetSystemMetrics(0)
                sh = user32.GetSystemMetrics(1)
                cv2.moveWindow("Classificador de Prato", (sw - w) // 2, (sh - h) // 2)
            except Exception:
                pass

        if frame_count % INTERVALO_AUTO == 0:
            ultimo_resultado, ultimo_confiancas = _predizer(base, labels, prototipos_np, frame)
            _exibir_resultado_terminal(ultimo_resultado, ultimo_confiancas)

        if ultimo_resultado and ultimo_resultado["classe"] != "indefinido":
            _desenhar_overlay(display, ultimo_resultado, ultimo_confiancas)
        else:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, ALTURA_TOPO), COR_FUNDO, -1)
            cv2.addWeighted(overlay, 0.82, display, 0.18, 0, display)
            cv2.rectangle(display, (0, 0), (w, 4), COR_INCERTO, -1)
            cv2.putText(display, "Coloque o prato na balanca",
                        (MARGEM, 42), cv2.FONT_HERSHEY_DUPLEX, 0.9, COR_INCERTO, 2)
            cv2.putText(display, "Aguardando deteccao...",
                        (MARGEM, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        _desenhar_rodape(display)

        cv2.imshow("Classificador de Prato", display)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break
        elif tecla == ord(" "):
            ultimo_resultado, ultimo_confiancas = _predizer(base, labels, prototipos_np, frame)
            _exibir_resultado_terminal(ultimo_resultado, ultimo_confiancas)

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
#  TESTAR MODELO
# =============================================================================

def testar():
    labels, prototipos_np, base = _carregar_modelo()

    banco_path   = Path(BANCO_DIR)
    acertos      = {c: 0 for c in labels}
    totais       = {c: 0 for c in labels}

    print(f"\nTestando com imagens do banco...")
    for cls_name in labels:
        cls_path = banco_path / cls_name
        if not cls_path.exists():
            continue
        for p in cls_path.iterdir():
            if p.suffix.lower() not in EXTS:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            resultado, _ = _predizer(base, labels, prototipos_np, img)
            totais[cls_name] += 1
            if resultado["classe"] == cls_name:
                acertos[cls_name] += 1
            print(f"  {cls_name}/{p.name:30s} → {resultado['classe']} "
                  f"({'✓' if resultado['classe'] == cls_name else '✗'})")

    print(f"\n{'='*40}")
    total_geral  = sum(totais.values())
    acerto_geral = sum(acertos.values())
    for cls in labels:
        n = totais[cls]
        a = acertos[cls]
        print(f"  {cls:15s}  {a}/{n}  ({(a/n*100 if n else 0):.0f}%)")
    print(f"  {'TOTAL':15s}  {acerto_geral}/{total_geral}  "
          f"({(acerto_geral/total_geral*100 if total_geral else 0):.0f}%)")
    print("="*40)


# =============================================================================
#  GERENCIAR BANCO
# =============================================================================

def _listar_categorias():
    banco_path = Path(BANCO_DIR)
    banco_path.mkdir(exist_ok=True)
    cats = sorted([d for d in banco_path.iterdir() if d.is_dir()])
    print("\n  Categorias no banco:")
    if not cats:
        print("    (nenhuma)")
    for i, c in enumerate(cats, 1):
        subcats = sorted([d for d in c.iterdir() if d.is_dir()])
        imgs_diretas = [p for p in c.iterdir() if p.suffix.lower() in EXTS]
        if subcats:
            total = sum(len([p for p in s.iterdir() if p.suffix.lower() in EXTS]) for s in subcats)
            print(f"    {i}. {c.name}  ({total} imagem(ns))")
            for s in subcats:
                n = len([p for p in s.iterdir() if p.suffix.lower() in EXTS])
                print(f"         - {s.name}  ({n} imagem(ns))")
        else:
            print(f"    {i}. {c.name}  ({len(imgs_diretas)} imagem(ns))")
    return cats


def _capturar_foto_camera(destino: Path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERRO: Camera nao encontrada.")
        return False

    print("  Camera aberta. ESPACO = capturar  |  Q = cancelar")
    cv2.namedWindow("Capturar foto", cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = cv2.flip(frame, 1).copy()
        h, w = display.shape[:2]
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - 32), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
        cv2.putText(display, "ESPACO = capturar   Q = cancelar",
                    (MARGEM, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        cv2.imshow("Capturar foto", display)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord(" "):
            cv2.imwrite(str(destino), cv2.flip(frame, 1))
            cap.release()
            cv2.destroyAllWindows()
            print(f"  Foto salva: {destino.name}")
            return True
        if tecla == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False


def gerenciar_banco():
    while True:
        print("\n--------------------------------------------")
        print("  Gerenciar Banco")
        print("--------------------------------------------")
        _listar_categorias()
        print()
        print("  1. Nova categoria")
        print("  2. Adicionar foto a categoria existente")
        print("  3. Remover categoria")
        print("  4. Remover item (subpasta)")
        print("  5. Remover foto")
        print("  0. Voltar")
        print("--------------------------------------------")
        op = input("  Escolha: ").strip()

        if op == "0":
            return

        elif op == "1":
            nome = input("  Nome da nova categoria: ").strip().lower()
            if not nome:
                print("  Nome invalido.")
                continue
            pasta = Path(BANCO_DIR) / nome
            if pasta.exists():
                print(f"  Categoria '{nome}' ja existe.")
                continue
            pasta.mkdir(parents=True)
            print(f"  Categoria '{nome}' criada.")
            print("  Deseja adicionar uma foto agora? (s/n): ", end="")
            if input().strip().lower() == "s":
                _adicionar_foto(pasta, nome)

        elif op == "2":
            cats = [d for d in Path(BANCO_DIR).iterdir() if d.is_dir()]
            if not cats:
                print("  Nenhuma categoria encontrada. Crie uma primeiro.")
                continue
            cats = sorted(cats)
            for i, c in enumerate(cats, 1):
                print(f"    {i}. {c.name}")
            sel = input("  Numero da categoria: ").strip()
            if not sel.isdigit() or not (1 <= int(sel) <= len(cats)):
                print("  Selecao invalida.")
                continue
            pasta = cats[int(sel) - 1]
            _adicionar_foto(pasta, pasta.name)

        elif op == "3":
            cats = sorted([d for d in Path(BANCO_DIR).iterdir() if d.is_dir()])
            if not cats:
                print("  Nenhuma categoria para remover.")
                continue
            for i, c in enumerate(cats, 1):
                print(f"    {i}. {c.name}")
            sel = input("  Numero da categoria a remover: ").strip()
            if not sel.isdigit() or not (1 <= int(sel) <= len(cats)):
                print("  Selecao invalida.")
                continue
            pasta = cats[int(sel) - 1]
            conf = input(f"  Confirma remover '{pasta.name}' e todas as fotos? (s/n): ").strip().lower()
            if conf == "s":
                import shutil
                shutil.rmtree(str(pasta))
                print(f"  Categoria '{pasta.name}' removida.")

        elif op == "4":
            cats = sorted([d for d in Path(BANCO_DIR).iterdir() if d.is_dir()])
            if not cats:
                print("  Nenhuma categoria encontrada.")
                continue
            for i, c in enumerate(cats, 1):
                print(f"    {i}. {c.name}")
            sel = input("  Numero da categoria: ").strip()
            if not sel.isdigit() or not (1 <= int(sel) <= len(cats)):
                print("  Selecao invalida.")
                continue
            cat = cats[int(sel) - 1]
            subcats = sorted([d for d in cat.iterdir() if d.is_dir()])
            if not subcats:
                print("  Essa categoria nao tem itens (subpastas).")
                continue
            for i, s in enumerate(subcats, 1):
                print(f"    {i}. {s.name}")
            sel2 = input("  Numero do item a remover: ").strip()
            if not sel2.isdigit() or not (1 <= int(sel2) <= len(subcats)):
                print("  Selecao invalida.")
                continue
            subpasta = subcats[int(sel2) - 1]
            conf = input(f"  Confirma remover '{subpasta.name}' e todas as fotos? (s/n): ").strip().lower()
            if conf == "s":
                import shutil
                shutil.rmtree(str(subpasta))
                print(f"  Item '{subpasta.name}' removido.")

        elif op == "5":
            cats = sorted([d for d in Path(BANCO_DIR).iterdir() if d.is_dir()])
            if not cats:
                print("  Nenhuma categoria encontrada.")
                continue
            for i, c in enumerate(cats, 1):
                print(f"    {i}. {c.name}")
            sel = input("  Numero da categoria: ").strip()
            if not sel.isdigit() or not (1 <= int(sel) <= len(cats)):
                print("  Selecao invalida.")
                continue
            pasta_alvo = cats[int(sel) - 1]
            subcats = sorted([d for d in pasta_alvo.iterdir() if d.is_dir()])
            if subcats:
                for i, s in enumerate(subcats, 1):
                    print(f"    {i}. {s.name}")
                sel2 = input("  Numero do item: ").strip()
                if not sel2.isdigit() or not (1 <= int(sel2) <= len(subcats)):
                    print("  Selecao invalida.")
                    continue
                pasta_alvo = subcats[int(sel2) - 1]
            imgs = sorted([p for p in pasta_alvo.iterdir() if p.suffix.lower() in EXTS])
            if not imgs:
                print("  Nenhuma foto encontrada.")
                continue
            for i, p in enumerate(imgs, 1):
                print(f"    {i}. {p.name}")
            sel3 = input("  Numero da foto a remover: ").strip()
            if not sel3.isdigit() or not (1 <= int(sel3) <= len(imgs)):
                print("  Selecao invalida.")
                continue
            foto = imgs[int(sel3) - 1]
            conf = input(f"  Confirma remover '{foto.name}'? (s/n): ").strip().lower()
            if conf == "s":
                foto.unlink()
                print(f"  Foto '{foto.name}' removida.")

        else:
            print("  Opcao invalida.")


def _adicionar_foto(pasta: Path, categoria: str):
    # Verifica se tem subpastas (2 níveis)
    subcats = sorted([d for d in pasta.iterdir() if d.is_dir()])
    if subcats:
        print(f"  Itens em '{categoria}':")
        for i, s in enumerate(subcats, 1):
            print(f"    {i}. {s.name}")
        print(f"    {len(subcats)+1}. Criar novo item")
        sel = input("  Escolha: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(subcats):
            pasta = subcats[int(sel) - 1]
            categoria = pasta.name
        elif sel == str(len(subcats) + 1):
            nome = input("  Nome do novo item: ").strip().lower()
            if not nome:
                print("  Nome invalido.")
                return
            pasta = pasta / nome
            pasta.mkdir(parents=True, exist_ok=True)
            categoria = nome
        else:
            print("  Selecao invalida.")
            return
    else:
        # Pergunta se quer criar subpasta
        imgs_diretas = [p for p in pasta.iterdir() if p.suffix.lower() in EXTS]
        if not imgs_diretas:
            resp = input(f"  Criar subpasta de item dentro de '{categoria}'? (s/n): ").strip().lower()
            if resp == "s":
                nome = input("  Nome do item: ").strip().lower()
                if not nome:
                    print("  Nome invalido.")
                    return
                pasta = pasta / nome
                pasta.mkdir(parents=True, exist_ok=True)
                categoria = nome

    imgs_atuais  = [p for p in pasta.iterdir() if p.suffix.lower() in EXTS]
    proximo_n    = len(imgs_atuais) + 1
    nome_arquivo = f"{categoria}_{proximo_n}.jpg"
    destino      = pasta / nome_arquivo

    print(f"  Como deseja adicionar a foto?")
    print(f"    1. Tirar foto pela camera")
    print(f"    2. Informar caminho do arquivo")
    op = input("  Escolha: ").strip()

    if op == "1":
        _capturar_foto_camera(destino)
    elif op == "2":
        caminho = input("  Caminho da imagem: ").strip().strip('"')
        src = Path(caminho)
        if not src.exists() or src.suffix.lower() not in EXTS:
            print("  Arquivo invalido ou formato nao suportado.")
            return
        import shutil
        shutil.copy2(str(src), str(destino))
        print(f"  Imagem copiada como: {nome_arquivo}")
    else:
        print("  Opcao invalida.")


# =============================================================================
#  MAIN
# =============================================================================

def main():
        print("\n============================================")
        print("   Classificador de Pratos")
        print("   Developed by Daniel Bitencourt")
        print("============================================")

        if _banco_mudou():
            print("  ! Mudancas detectadas no banco — retreino necessario")

        print("  1. Treinar")
        print("  2. Camera ao vivo")
        print("  3. Gerenciar banco")
        print("  0. Sair")
        print("--------------------------------------------")
        opcao = input("  Escolha uma opcao: ").strip()

        if opcao == "1":
            treinar()
        elif opcao == "2":
            if _banco_mudou():
                print("\n  Banco alterado desde o ultimo treino. Retreinando...")
                treinar(silencioso=True)
            modo_camera()
        elif opcao == "3":
            gerenciar_banco()
        elif opcao == "0":
            print("\nAte logo!")
            sys.exit(0)
        else:
            print("  Opcao invalida. Tente novamente.")


if __name__ == "__main__":
    main()