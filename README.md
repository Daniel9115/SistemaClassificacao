# 🍽️ Classificador de Pratos — API

> API REST construída com **FastAPI** que utiliza **MobileNetV2** (transfer learning) para classificar imagens de pratos/alimentos de forma rápida e sem necessidade de treinamento tradicional com GPU. O sistema usa uma abordagem de **protótipos por similaridade de cosseno**, tornando o retreino instantâneo ao adicionar novas fotos.

---

## 📑 Sumário

- [Como funciona](#-como-funciona)
- [Estrutura de pastas](#-estrutura-de-pastas)
- [Instalação e execução](#-instalação-e-execução)
- [Fluxo completo de uso](#-fluxo-completo-de-uso)
- [Endpoints](#-endpoints)
  - [GET /status](#get-status)
  - [POST /treinar](#post-treinar)
  - [POST /classify](#post-classify)
  - [GET /categorias](#get-categorias)
  - [POST /categorias](#post-categorias)
  - [DELETE /categorias/{nome}](#delete-categoriasnome)
  - [POST /categorias/{nome}/fotos](#post-categoriasнomefotos)
  - [DELETE /categorias/{nome}/fotos/{arquivo}](#delete-categoriasnomefotosarquivo)
  - [POST /categorias/{nome}/itens](#post-categoriasnomeitens)
  - [DELETE /categorias/{nome}/itens/{item}](#delete-categoriasnomeitensitem)
  - [POST /categorias/{nome}/itens/{item}/fotos](#post-categoriasnomeitensitemfotos)
  - [DELETE /categorias/{nome}/itens/{item}/fotos/{arquivo}](#delete-categoriasnomeitensitemfotosarquivo)
- [Conceitos técnicos](#-conceitos-técnicos)
- [Perguntas frequentes](#-perguntas-frequentes)

---

## 🧠 Como funciona

O sistema **não realiza treinamento clássico com backpropagation**. Em vez disso, usa uma estratégia chamada **classificação por protótipos**:

```
Imagem de entrada
       │
       ▼
  MobileNetV2               ← rede pré-treinada no ImageNet (congelada)
  (extrator de features)
       │
       ▼
  Embedding (vetor)         ← representação numérica de 1280 dimensões
       │
       ▼
  Similaridade de cosseno   ← compara com todos os protótipos salvos
       │
       ▼
  Softmax → probabilidades  ← quanto a imagem se parece com cada classe
       │
       ▼
  Classe + confiança
```

**Durante o treino**, para cada categoria o sistema:
1. Lê todas as imagens da pasta correspondente
2. Aplica data augmentation (flip, brilho, rotação) se houver poucas fotos
3. Extrai o embedding de cada imagem
4. Calcula a **média** dos embeddings → isso é o **protótipo** da classe
5. Salva os protótipos em disco (`models/prototipos.npy`)

**Durante a classificação**, a imagem nova é comparada com todos os protótipos. A classe com maior similaridade de cosseno vence — desde que a confiança supere o limiar de **50%**; caso contrário, o resultado é marcado como `indefinido`.

---

## 📁 Estrutura de pastas

```
projeto/
│
├── api.py                        ← código da API (este arquivo)
│
├── BancoImagens/                 ← banco de imagens para treino
│   ├── pizza/                    ← categoria simples
│   │   ├── pizza_1.jpg
│   │   └── pizza_2.jpg
│   ├── massas/                   ← categoria com itens (subpastas)
│   │   ├── carbonara/
│   │   │   ├── carbonara_1.jpg
│   │   │   └── carbonara_2.jpg
│   │   └── bolonhesa/
│   │       └── bolonhesa_1.jpg
│   └── ...
│
└── models/                       ← gerado automaticamente após treino
    ├── classificador.tflite      ← modelo exportado (uso mobile/edge)
    ├── labels.json               ← lista de classes na ordem dos protótipos
    ├── prototipos.npy            ← vetores protótipo de cada classe
    └── banco_snapshot.json       ← snapshot do banco para detectar mudanças
```

### Lógica de organização do BancoImagens

| Estrutura | Resultado no label |
|---|---|
| `BancoImagens/pizza/` com fotos diretas | classe = `"pizza"` |
| `BancoImagens/massas/carbonara/` com fotos | classe = `"massas"`, item = `"carbonara"` |

Categorias com subpastas (itens) geram labels no formato `"categoria/item"`.

---

## 🚀 Instalação e execução

### Pré-requisitos

- Python 3.9+
- pip

### Passos

```bash
# 1. Clone o repositório ou copie os arquivos
git clone <url-do-repo>
cd projeto

# 2. (Opcional) Crie um ambiente virtual
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Execute a API — as dependências são instaladas automaticamente
python api.py
```

> As dependências (`fastapi`, `uvicorn`, `tensorflow`, `opencv-python`, `numpy`, `python-multipart`) são instaladas automaticamente na primeira execução caso não estejam presentes.

A API estará disponível em:

| URL | Descrição |
|---|---|
| `http://localhost:8000` | Raiz da API |
| `http://localhost:8000/docs` | Swagger UI (documentação interativa) |
| `http://localhost:8000/redoc` | ReDoc (documentação alternativa) |

---

## 🔄 Fluxo completo de uso

O diagrama abaixo mostra o caminho típico do zero até classificar uma imagem:

```
┌─────────────────────────────────────────────────────────┐
│  1. Preparar o banco de imagens                         │
│                                                         │
│  POST /categorias          → cria categoria "pizza"     │
│  POST /categorias/pizza/fotos  → envia foto_1.jpg       │
│  POST /categorias/pizza/fotos  → envia foto_2.jpg       │
│                                                         │
│  (opcional: usar subpastas/itens)                       │
│  POST /categorias/massas/itens         → cria "carbonara"│
│  POST /categorias/massas/itens/carbonara/fotos → foto   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  2. Treinar o modelo                                    │
│                                                         │
│  POST /treinar             → inicia treino (background) │
│  GET  /status              → monitora até concluir      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  3. Classificar imagens                                 │
│                                                         │
│  POST /classify (multipart com a imagem)                │
│  ← retorna classe, item, confiança e probabilidades     │
└─────────────────────────────────────────────────────────┘
```

**Quando retreinar?** Sempre que adicionar, remover ou substituir fotos no banco. O endpoint `GET /status` informa o campo `banco_modificado: true` quando o banco mudou desde o último treino.

---

## 📡 Endpoints

### GET /status

Retorna o estado atual da API: se o modelo está carregado, quais classes existem, se o banco foi modificado e se há treino em andamento.

**Resposta:**
```json
{
  "online": true,
  "modelo_carregado": true,
  "classes": ["pizza", "massas/carbonara", "massas/bolonhesa"],
  "banco_modificado": false,
  "treino_em_andamento": false
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `online` | bool | Sempre `true` se a API estiver respondendo |
| `modelo_carregado` | bool | `true` se há protótipos e labels carregados na memória |
| `classes` | array | Lista de classes conhecidas pelo modelo atual |
| `banco_modificado` | bool | `true` se o banco de imagens mudou desde o último treino |
| `treino_em_andamento` | bool | `true` enquanto um treino está sendo executado |

---

### POST /treinar

Inicia o treinamento do modelo com todas as imagens presentes em `BancoImagens/`.

**Query parameters:**

| Parâmetro | Tipo | Padrão | Descrição |
|---|---|---|---|
| `sincrono` | bool | `false` | Se `true`, a requisição aguarda o treino concluir antes de retornar |

**Exemplo — treino assíncrono (recomendado):**
```bash
curl -X POST http://localhost:8000/treinar
```
```json
{
  "ok": true,
  "mensagem": "Treino iniciado em background. Consulte GET /status."
}
```

**Exemplo — treino síncrono:**
```bash
curl -X POST "http://localhost:8000/treinar?sincrono=true"
```
```json
{
  "ok": true,
  "classes": ["pizza", "massas/carbonara"],
  "log": [
    "[pizza] 3 imagem(ns)",
    "[massas/carbonara] 2 imagem(ns)",
    "Treino concluído. 2 classe(s)."
  ]
}
```

**Erros possíveis:**

| Código | Motivo |
|---|---|
| `409` | Já existe um treino em andamento |
| `500` | Nenhuma categoria encontrada ou erro interno |

> **Atenção:** O treino recarrega o MobileNetV2 na primeira execução, o que pode levar alguns segundos (ou minutos na primeira vez que baixar os pesos). Execuções subsequentes são muito mais rápidas.

---

### POST /classify

Classifica uma imagem enviada via `multipart/form-data`. O campo do arquivo deve se chamar `file`.

**Requisição:**
```bash
curl -X POST http://localhost:8000/classify \
  -F "file=@minha_foto.jpg"
```

**Resposta — classificação bem-sucedida:**
```json
{
  "classe": "pizza",
  "item": "",
  "confianca": 0.8732,
  "todas": {
    "pizza": 0.8732,
    "massas/carbonara": 0.0891,
    "massas/bolonhesa": 0.0377
  },
  "indefinido": false
}
```

**Resposta — abaixo do limiar de confiança:**
```json
{
  "classe": "",
  "item": "",
  "confianca": 0.0,
  "todas": {
    "pizza": 0.3821,
    "massas/carbonara": 0.3512,
    "massas/bolonhesa": 0.2667
  },
  "indefinido": true
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `classe` | string | Nome da categoria identificada (vazio se indefinido) |
| `item` | string | Subitem identificado, se houver (ex: `"carbonara"`) |
| `confianca` | float | Probabilidade da melhor classe (0.0 a 1.0) |
| `todas` | object | Probabilidade de todas as classes conhecidas |
| `indefinido` | bool | `true` se a confiança ficou abaixo de 50% |

**Erros possíveis:**

| Código | Motivo |
|---|---|
| `400` | Arquivo enviado não é uma imagem válida |
| `503` | Modelo não carregado — execute `POST /treinar` primeiro |

---

### GET /categorias

Lista todas as categorias cadastradas no banco, incluindo itens (subpastas) e nomes dos arquivos de foto.

**Resposta:**
```json
[
  {
    "nome": "pizza",
    "itens": [],
    "fotos": ["pizza_1.jpg", "pizza_2.jpg"],
    "total": 2
  },
  {
    "nome": "massas",
    "itens": [
      {
        "nome": "carbonara",
        "fotos": ["carbonara_1.jpg"],
        "total": 1
      }
    ],
    "fotos": [],
    "total": 1
  }
]
```

---

### POST /categorias

Cria uma nova categoria (pasta) no banco de imagens.

**Body (JSON):**
```json
{ "nome": "sushi" }
```

**Resposta:**
```json
{ "ok": true, "categoria": "sushi" }
```

**Erros possíveis:**

| Código | Motivo |
|---|---|
| `400` | Nome vazio ou inválido |
| `409` | Categoria já existe |

---

### DELETE /categorias/{nome}

Remove uma categoria e **todas** as fotos e subpastas dentro dela.

```bash
curl -X DELETE http://localhost:8000/categorias/sushi
```

**Resposta:**
```json
{ "ok": true, "removido": "sushi" }
```

**Erros possíveis:**

| Código | Motivo |
|---|---|
| `404` | Categoria não encontrada |

---

### POST /categorias/{nome}/fotos

Adiciona uma foto diretamente a uma categoria (sem subpasta/item). O arquivo é salvo com nome automático no formato `{categoria}_{n}.ext`.

```bash
curl -X POST http://localhost:8000/categorias/pizza/fotos \
  -F "file=@nova_pizza.jpg"
```

**Resposta:**
```json
{ "ok": true, "arquivo": "pizza_3.jpg", "categoria": "pizza" }
```

**Erros possíveis:**

| Código | Motivo |
|---|---|
| `404` | Categoria não encontrada |
| `400` | Formato de imagem não suportado |

> Formatos aceitos: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

---

### DELETE /categorias/{nome}/fotos/{arquivo}

Remove uma foto específica de uma categoria.

```bash
curl -X DELETE http://localhost:8000/categorias/pizza/fotos/pizza_1.jpg
```

**Resposta:**
```json
{ "ok": true, "removido": "pizza_1.jpg" }
```

---

### POST /categorias/{nome}/itens

Cria um item (subpasta) dentro de uma categoria existente. Itens são usados para classificar variações de uma mesma categoria.

**Body (JSON):**
```json
{ "nome": "quattro-stagioni" }
```

```bash
curl -X POST http://localhost:8000/categorias/pizza/itens \
  -H "Content-Type: application/json" \
  -d '{"nome": "quattro-stagioni"}'
```

**Resposta:**
```json
{ "ok": true, "categoria": "pizza", "item": "quattro-stagioni" }
```

**Erros possíveis:**

| Código | Motivo |
|---|---|
| `404` | Categoria não encontrada |
| `400` | Nome vazio ou inválido |
| `409` | Item já existe nessa categoria |

---

### DELETE /categorias/{nome}/itens/{item}

Remove um item e **todas** as fotos dentro dele.

```bash
curl -X DELETE http://localhost:8000/categorias/pizza/itens/quattro-stagioni
```

**Resposta:**
```json
{ "ok": true, "removido": "pizza/quattro-stagioni" }
```

---

### POST /categorias/{nome}/itens/{item}/fotos

Adiciona uma foto a um item (subpasta) de uma categoria. O arquivo é salvo com nome automático no formato `{item}_{n}.ext`.

```bash
curl -X POST http://localhost:8000/categorias/massas/itens/carbonara/fotos \
  -F "file=@carbonara_nova.jpg"
```

**Resposta:**
```json
{
  "ok": true,
  "arquivo": "carbonara_2.jpg",
  "categoria": "massas",
  "item": "carbonara"
}
```

---

### DELETE /categorias/{nome}/itens/{item}/fotos/{arquivo}

Remove uma foto específica de um item.

```bash
curl -X DELETE http://localhost:8000/categorias/massas/itens/carbonara/fotos/carbonara_1.jpg
```

**Resposta:**
```json
{ "ok": true, "removido": "carbonara_1.jpg" }
```

---

## 🔬 Conceitos técnicos

### MobileNetV2 como extrator de features

O MobileNetV2 é uma rede neural treinada em 1.2 milhão de imagens (ImageNet). Em vez de usar sua camada de classificação final, a API usa apenas o **corpo da rede** (`include_top=False, pooling="avg"`), que transforma qualquer imagem 224×224 em um vetor de **1280 números** que representa as características visuais daquela imagem — texturas, formas, cores, padrões.

### Protótipos e similaridade de cosseno

Para cada classe, o **protótipo** é a média normalizada de todos os embeddings das imagens de treino daquela classe. É um vetor que representa o "centro visual" da categoria.

Na classificação, a similaridade de cosseno mede o ângulo entre o embedding da imagem nova e cada protótipo. Valores próximos de `1.0` indicam alta similaridade; próximos de `0.0` indicam imagens completamente diferentes.

### Data Augmentation

Quando uma categoria tem menos de 5 imagens, o sistema gera variações automáticas de cada foto:
- Espelhamento horizontal
- Brilho aumentado (+20%)
- Brilho reduzido (-20%)
- Rotação de +10°
- Rotação de -10°

Isso amplia artificialmente o banco e melhora a robustez do protótipo.

### Limiar de confiança

O sistema aplica **softmax com temperatura 10** sobre as similaridades de cosseno, gerando probabilidades. Se a maior probabilidade for menor que **50% (0.50)**, a classificação é marcada como `indefinido: true` — indicando que a imagem não se encaixa com confiança em nenhuma classe conhecida.

### Exportação TFLite

Após cada treino, o modelo completo (extrator + comparação com protótipos + softmax) é exportado como um arquivo `.tflite` em `models/classificador.tflite`, pronto para uso em aplicações mobile (Android/iOS) ou dispositivos de borda (Raspberry Pi, Coral, etc.).

### Detecção de mudanças no banco

O sistema mantém um snapshot (`banco_snapshot.json`) com o caminho e tamanho de cada imagem no banco. A cada chamada a `GET /status`, o snapshot atual é comparado com o salvo. Se houver diferença, `banco_modificado` retorna `true`, sinalizando que um novo treino é necessário.

---

## ❓ Perguntas frequentes

**Quantas fotos preciso por categoria?**
O mínimo funcional é 1 foto. Com menos de 5 fotos, data augmentation é aplicada automaticamente. Para melhores resultados, recomenda-se ao menos 5–10 fotos variadas por categoria.

**O modelo é salvo entre reinicializações?**
Sim. Os arquivos `labels.json` e `prototipos.npy` são salvos em disco após cada treino e recarregados automaticamente quando a API inicia.

**Posso usar GPU?**
Sim. Se o TensorFlow encontrar uma GPU disponível (CUDA configurado), ele a utilizará automaticamente. A API não requer configuração adicional para isso.

**Posso rodar com Docker?**
Sim. Exemplo mínimo de `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn tensorflow opencv-python numpy python-multipart
EXPOSE 8000
CMD ["python", "api.py"]
```

**Como integrar com um frontend?**
A API tem CORS liberado para todas as origens (`*`). Basta fazer requisições HTTP normais do seu frontend. Para classificar, envie um `FormData` com o campo `file` contendo a imagem:
```javascript
const form = new FormData();
form.append("file", inputFile.files[0]);
const res = await fetch("http://localhost:8000/classify", {
  method: "POST",
  body: form,
});
const resultado = await res.json();
```

**O que acontece se eu enviar uma imagem de algo completamente diferente?**
O campo `indefinido` retornará `true` e `confianca` será `0.0`. O sistema só afirma uma classe quando tem pelo menos 50% de certeza.

---

## 📄 Licença

Distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
