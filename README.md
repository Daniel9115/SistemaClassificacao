# 🍽️ Classificador de Pratos + QR/Barcode de Comanda — API

API REST construída com **FastAPI** que utiliza **MobileNetV2** (transfer learning) para classificar imagens de pratos/alimentos usando uma abordagem de **protótipos por similaridade de cosseno**.

Além da classificação de pratos, a API também lê números de comanda via:

- QR Code
- Código de barras

utilizando `pyzbar`.

A solução foi projetada para:

- funcionar bem em CPU
- não exigir GPU
- permitir retreino instantâneo
- adicionar novas categorias facilmente
- operar com poucos exemplos por classe

---

# 📑 Sumário

- [Visão geral](#-visão-geral)
- [Como funciona](#-como-funciona)
- [Estrutura do projeto](#-estrutura-do-projeto)
- [Fluxo completo](#-fluxo-completo)
- [Instalação](#-instalação)
- [Execução](#-execução)
- [Documentação automática](#-documentação-automática)
- [Endpoints](#-endpoints)
  - [GET /status](#get-status)
  - [POST /treinar](#post-treinar)
  - [POST /classify](#post-classify)
  - [GET /categorias](#get-categorias)
  - [POST /categorias](#post-categorias)
  - [DELETE /categorias/{categoria}](#delete-categoriascategoria)
  - [POST /categorias/{categoria}/itens](#post-categoriascategoriaitens)
  - [DELETE /categorias/{categoria}/itens/{item}](#delete-categoriascategoriaitensitem)
  - [POST /categorias/{categoria}/fotos](#post-categoriascategoriafotos)
  - [DELETE /categorias/{categoria}/fotos/{foto}](#delete-categoriascategoriafotosfoto)
  - [POST /categorias/{categoria}/itens/{item}/fotos](#post-categoriascategoriaitensitemfotos)
  - [DELETE /categorias/{categoria}/itens/{item}/fotos/{foto}](#delete-categoriascategoriaitensitemfotosfoto)
- [Conceitos técnicos](#-conceitos-técnicos)
- [Performance](#-performance)
- [Limitações](#-limitações)
- [Perguntas frequentes](#-perguntas-frequentes)
- [Licença](#-licença)

---

# 🧠 Visão geral

A API possui dois objetivos principais:

## 1. Classificação de pratos

Identifica automaticamente o prato/alimento presente em uma imagem utilizando:

- MobileNetV2
- embeddings vetoriais
- similaridade de cosseno
- classificação por protótipos

---

## 2. Leitura de comandas

Detecta QR Codes ou códigos de barras presentes na imagem e extrai números de comanda automaticamente.

---

# ⚙️ Como funciona

A API não realiza treinamento clássico com backpropagation.

Em vez disso, utiliza uma abordagem extremamente rápida chamada:

# Classificação por Protótipos

Fluxo completo:

```text
Imagem de entrada
       │
       ├──────────────────────────────┐
       │                              │
       ▼                              ▼
  MobileNetV2                 QR / Barcode
 (feature extractor)            (pyzbar)
       │                              │
       ▼                              ▼
 Embedding 1280d               Número da comanda
       │
       ▼
Similaridade de cosseno
       │
       ▼
 Softmax
       │
       ▼
Classe + confiança + comanda
```

---

# 🧩 Pipeline de treino

Durante o treino:

1. A API percorre todas as categorias do `BancoImagens`
2. Cada imagem é convertida em embedding
3. Embeddings da mesma classe são agrupados
4. A média vetorial é calculada
5. O resultado vira o protótipo daquela classe

---

# 🧠 Pipeline de classificação

Durante a classificação:

1. A imagem vira um embedding
2. O embedding é comparado com todos os protótipos
3. A maior similaridade vence
4. Softmax transforma similaridade em probabilidade
5. Se a confiança for menor que `70%`, o resultado vira `indefinido`

---

# 📁 Estrutura do projeto

```text
projeto/
│
├── api.py
│
├── BancoImagens/
│   ├── pizza/
│   │   ├── pizza_1.jpg
│   │   └── pizza_2.jpg
│   │
│   ├── massas/
│   │   ├── carbonara/
│   │   │   ├── carbonara_1.jpg
│   │   │   └── carbonara_2.jpg
│   │   │
│   │   └── bolonhesa/
│   │       └── bolonhesa_1.jpg
│   │
│   └── sushi/
│       ├── sushi_1.jpg
│       └── sushi_2.jpg
│
└── models/
    ├── labels.json
    ├── prototipos.npy
    └── banco_snapshot.json
```

---

# 📌 Regras de organização

## Categoria simples

```text
BancoImagens/pizza/
```

Resultado:

```json
{
  "classe": "pizza"
}
```

---

## Categoria com itens

```text
BancoImagens/massas/carbonara/
```

Resultado:

```json
{
  "classe": "massas",
  "item": "carbonara"
}
```

---

# 🔄 Fluxo completo

```text
1. Criar categorias
2. Enviar fotos
3. Executar treino
4. Classificar imagens
5. Ler QR/barcode da comanda
```

Exemplo:

```text
POST /categorias
POST /categorias/pizza/fotos
POST /treinar
POST /classify
```

---

# 🚀 Instalação

## Requisitos

- Python 3.10+
- pip

---

## Clone o projeto

```bash
git clone https://github.com/seuusuario/seuprojeto.git
cd seuprojeto
```

---

## Instale as dependências

```bash
pip install fastapi uvicorn[standard] tensorflow opencv-python numpy python-multipart pyzbar
```

---

# ▶️ Execução

```bash
python api.py
```

Servidor:

```text
http://localhost:8000
```

---

# 📚 Documentação automática

Swagger UI:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

---

# 📡 Endpoints

# GET /status

Retorna o estado atual da API.

## Exemplo

```json
{
  "online": true,
  "modelo_carregado": true,
  "classes": [
    "pizza",
    "massas/carbonara"
  ],
  "banco_modificado": false,
  "treino_em_andamento": false,
  "qr_mode": true
}
```

---

## Campos

| Campo | Tipo | Descrição |
|---|---|---|
| `online` | bool | API online |
| `modelo_carregado` | bool | Modelo carregado |
| `classes` | array | Classes disponíveis |
| `banco_modificado` | bool | Banco mudou desde último treino |
| `treino_em_andamento` | bool | Existe treino em andamento |
| `qr_mode` | bool | Leitura QR/barcode habilitada |

---

# POST /treinar

Executa o treino do modelo.

---

## Query Params

| Parâmetro | Tipo | Padrão |
|---|---|---|
| `sincrono` | bool | false |

---

## Treino assíncrono

```bash
curl -X POST http://localhost:8000/treinar
```

Resposta:

```json
{
  "ok": true,
  "mensagem": "Treino iniciado."
}
```

---

## Treino síncrono

```bash
curl -X POST "http://localhost:8000/treinar?sincrono=true"
```

Resposta:

```json
{
  "ok": true,
  "classes": [
    "pizza",
    "massas/carbonara"
  ],
  "log": [
    "[pizza] 3 imagem(ns)",
    "[massas/carbonara] 2 imagem(ns)",
    "Treino concluído (2 classes)"
  ]
}
```

---

## Possíveis erros

| Código | Motivo |
|---|---|
| `409` | Treino já em andamento |
| `500` | Erro interno |

---

# POST /classify

Classifica uma imagem e tenta ler QR/barcode simultaneamente.

---

## Requisição

```bash
curl -X POST http://localhost:8000/classify \
  -F "file=@foto.jpg"
```

---

## Resposta

```json
{
  "classe": "pizza",
  "item": "",
  "confianca": 0.8732,
  "todas": {
    "pizza": 0.8732,
    "massas/carbonara": 0.1268
  },
  "indefinido": false,
  "comanda": "42"
}
```

---

## Resultado indefinido

```json
{
  "classe": "",
  "item": "",
  "confianca": 0.0,
  "todas": {
    "pizza": 0.45,
    "massas/carbonara": 0.55
  },
  "indefinido": true,
  "comanda": "42"
}
```

---

## Campos

| Campo | Tipo | Descrição |
|---|---|---|
| `classe` | string | Categoria identificada |
| `item` | string | Subitem identificado |
| `confianca` | float | Confiança da predição |
| `todas` | object | Todas as probabilidades |
| `indefinido` | bool | Resultado abaixo do limiar |
| `comanda` | string | Número extraído do QR/barcode |

---

## Possíveis erros

| Código | Motivo |
|---|---|
| `400` | Imagem inválida |
| `503` | Modelo não carregado |

---

# GET /categorias

Lista todas as categorias.

---

## Resposta

```json
[
  {
    "nome": "pizza",
    "itens": [],
    "fotos": [
      "pizza_1.jpg"
    ],
    "total": 1
  },
  {
    "nome": "massas",
    "itens": [
      {
        "nome": "carbonara",
        "fotos": [
          "carbonara_1.jpg"
        ],
        "total": 1
      }
    ],
    "fotos": [],
    "total": 1
  }
]
```

---

# POST /categorias

Cria uma nova categoria.

---

## Body

```json
{
  "nome": "sushi"
}
```

---

## Resposta

```json
{
  "ok": true,
  "categoria": "sushi"
}
```

---

# DELETE /categorias/{categoria}

Remove uma categoria inteira.

---

## Exemplo

```bash
curl -X DELETE http://localhost:8000/categorias/sushi
```

---

# POST /categorias/{categoria}/itens

Cria um subitem.

---

## Body

```json
{
  "nome": "carbonara"
}
```

---

## Exemplo

```bash
curl -X POST http://localhost:8000/categorias/massas/itens \
  -H "Content-Type: application/json" \
  -d '{"nome":"carbonara"}'
```

---

# DELETE /categorias/{categoria}/itens/{item}

Remove um subitem.

---

# POST /categorias/{categoria}/fotos

Envia uma foto para categoria ou item.

---

## Categoria simples

```bash
curl -X POST http://localhost:8000/categorias/pizza/fotos \
  -F "file=@pizza.jpg"
```

---

## Subitem

```bash
curl -X POST "http://localhost:8000/categorias/massas/fotos?item=carbonara" \
  -F "file=@carbonara.jpg"
```

---

## Resposta

```json
{
  "ok": true,
  "categoria": "pizza",
  "item": null,
  "arquivo": "pizza.jpg"
}
```

---

# DELETE /categorias/{categoria}/fotos/{foto}

Remove foto de categoria.

---

# POST /categorias/{categoria}/itens/{item}/fotos

Upload direto para item.

---

# DELETE /categorias/{categoria}/itens/{item}/fotos/{foto}

Remove foto de item.

---

# 🔬 Conceitos técnicos

# MobileNetV2

A MobileNetV2 foi treinada originalmente na base ImageNet com milhões de imagens.

Na API ela é usada apenas como:

# Extrator de Features

A saída da rede é um vetor de:

```text
1280 dimensões
```

representando características visuais da imagem.

---

# Similaridade de cosseno

A classificação usa:

```text
produto escalar normalizado
```

para medir o quanto dois embeddings são parecidos.

Valores próximos de:

```text
1.0
```

indicam imagens semelhantes.

---

# Softmax

A API transforma similaridades em probabilidades utilizando:

```text
exp(similaridade * 8)
```

---

# Data augmentation

Quando uma classe possui menos de 5 imagens, a API gera automaticamente:

- flip horizontal
- brilho +20%
- brilho -20%
- rotação +10°
- rotação -10°

---

# Snapshot do banco

A API monitora alterações no banco de imagens através de:

```text
models/banco_snapshot.json
```

Sempre que imagens forem:

- adicionadas
- removidas
- substituídas

o endpoint `/status` retornará:

```json
{
  "banco_modificado": true
}
```

---

# QR Code / Barcode

A leitura de comandas utiliza:

```text
pyzbar
```

A API:

1. detecta QR/barcode
2. extrai texto
3. mantém apenas números
4. aceita códigos de 1 a 4 dígitos

---

# ⚡ Performance

Valores médios em CPU:

| Operação | Tempo |
|---|---|
| Classificação | ~80–150ms |
| Leitura QR | ~5–20ms |
| Treino pequeno | ~5–20s |
| Treino médio | ~20–60s |

Resultados variam conforme hardware.

---

# ✅ Vantagens da arquitetura

- Não exige GPU
- Retreino extremamente rápido
- Fácil adicionar novas classes
- Baixo uso de memória
- Excelente para datasets pequenos
- Ótimo para sistemas internos
- Funciona bem em CPU

---

# ⚠️ Limitações

- Não substitui treinamento supervisionado em datasets gigantes
- Pode confundir pratos visualmente similares
- Sensível a iluminação extrema
- Funciona melhor com enquadramento consistente
- QR/barcode precisam estar visíveis

---

# ❓ Perguntas frequentes

# Quantas fotos preciso?

Mínimo funcional:

```text
1 foto
```

Recomendado:

```text
5–10 fotos variadas por categoria
```

---

# Precisa de GPU?

Não.

A API foi projetada para funcionar bem em CPU.

---

# O modelo persiste?

Sim.

Arquivos salvos:

```text
models/labels.json
models/prototipos.npy
```

---

# Preciso retreinar quando adicionar fotos?

Sim.

Sempre que:

- adicionar
- remover
- substituir imagens

---

# Posso usar Docker?

Sim.

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn[standard] tensorflow opencv-python numpy python-multipart pyzbar

EXPOSE 8000

CMD ["python", "api.py"]
```

---

# Como integrar frontend?

Exemplo JavaScript:

```javascript
const form = new FormData();

form.append("file", input.files[0]);

const res = await fetch("http://localhost:8000/classify", {
  method: "POST",
  body: form
});

const data = await res.json();

console.log(data);
```

---

# E se a imagem não pertencer a nenhuma classe?

A API retornará:

```json
{
  "indefinido": true
}
```

---

# 📄 Licença

Distribuído sob licença MIT.

Consulte o arquivo:

```text
LICENSE
```

para mais detalhes.
