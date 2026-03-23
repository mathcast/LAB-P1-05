# LAB P1-05: Treinamento Fim-a-Fim do Transformer

## Como baixar o projeto

```
git clone https://github.com/mathcast/LAB-P1-05.git
cd LAB-P1-05
```

## Conceito implementado

Este projeto implementa uma versão simplificada de um Transformer Encoder, utilizando o mecanismo de atenção.

A principal operação utilizada é:

```
Attention(Q, K, V) = softmax((Q K^T) / √d_k) V
```

- Q (Query), K (Key), V (Value) são projeções da entrada.
- O produto escalar Q @ K^T mede similaridade entre tokens.
- O resultado é escalado por √d_k para estabilidade numérica.
- O softmax transforma os valores em probabilidades.
- A saída é obtida multiplicando os pesos por V.

## Objetivo do laboratório
Implementar um Transformer do zero

## Entender o fluxo:

```
texto → tokenização → embedding → self-attention → encoder → saída → loss
``` 

Treinar um modelo simples com um subconjunto do dataset opus_books

## Estrutura do repositório

```
LAB-P1-05/
│
├── data/
│   └── dataset.py
│
├── models/
│   ├── add_norm.py
│   ├── ffn.py 
│   ├── transformer.py
│   ├── encoder.py
│   ├── decoder.py
│   └── attention.py
│
├── training/
│   ├── train.py
│   ├── loss.py
│   └── optimizer.py
│
├── utils/
│   ├── tokenizer.py
│   ├── padding.py
│   └── inference.py
│
├── config/
│   └── config.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Dataset utilizado

O dataset utilizado foi:

- opus_books
- Idiomas: inglês → português
- Subconjunto: primeiras 1000 frases

Isso permite rodar rapidamente em CPU ou Google Colab gratuito.

## Como rodar
### 1. Ambiente virtual e dependências

No diretório do projeto:

#### Windows (PowerShell):

```
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
``` 

#### Linux/macOS:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Executar o treinamento

```
python main.py
```

#### Saída esperada

Durante a execução, o modelo exibirá:

```
🚀 Iniciando treino...

📊 Vocab size: XX

📚 Epoch 1/2
Loss: 2.78

📚 Epoch 2/2
Loss: 2.50
```

#### Interpretação:

- Loss alta (~2–4) → comportamento esperado
- Loss diminuindo → modelo aprendendo
- Loss ≈ 0 → erro (modelo apenas copiando entrada)

#### Tokenização

Foi utilizada uma abordagem simples:

- Nível de caractere
- Criação de vocabulário manual (stoi)

Conversão:

```
texto → caracteres → índices → tensor
```

Isso simplifica o treinamento e evita dependências externas.

#### Limitações do modelo

Este projeto é educacional, portanto possui limitações:

- Não realiza tradução real
- Tokenização simplificada (sem BPE ou WordPiece)
- Dataset pequeno

## Requisitos técnicos
- Linguagem: Python 3
- Bibliotecas:
    - PyTorch
    - datasets (Hugging Face)

## Conclusão

Este laboratório demonstra, de forma prática, o funcionamento interno de um Transformer:

- Implementação do Self-Attention
- Uso de embeddings
- Estrutura de encoder
- Treinamento com dados reais

Mesmo simplificado, o projeto permite entender os fundamentos por trás de modelos modernos de NLP.
