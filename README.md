# LLM Text Preprocessing Foundations and Embeddings

**Student:** Daniel Rodriguez

This repository contains a practical, end-to-end notebook implementation of core Chapter 2 ideas from *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

The project walks through the complete preprocessing pipeline used before training LLMs:

- loading raw text,
- tokenizing and converting text to token IDs,
- creating sliding-window training samples,
- building token and positional embeddings.

It also includes a focused experiment on `max_length` and `stride` to explain how overlap changes the number of training samples and why that matters.

---

## Overview

High-level pipeline implemented in this repo:

```text
Raw Text
	↓
Tokenization (GPT-2 BPE via tiktoken)
	↓
Token IDs
	↓
Sliding-Window (input, target) pairs
	↓
Token Embeddings + Positional Embeddings
	↓
LLM-ready input tensors
```

---

## What is implemented

### 1) Base chapter notebook execution

- Downloaded and used the official chapter notebook and text sample.
- Executed the chapter notebook from start to finish in the local environment.
- Confirmed package availability (`torch`, `tiktoken`) and working outputs.

### 2) Custom explanatory notebook

Created `embeddings.ipynb` with:

- core code patterns from the chapter,
- personal explanations in multiple markdown sections,
- emphasis on LLM and agentic-system relevance,
- an explicit conceptual answer to:
	- **Why do embeddings encode meaning, and how are they related to neural network concepts?**

### 3) Sliding-window experiment

Implemented a direct experiment varying `max_length` and `stride` and reporting sample counts.

Observed pattern:

- smaller `stride` (more overlap) -> more samples,
- larger `stride` (less overlap) -> fewer samples,
- overlap helps continuity learning because nearby windows preserve shared context.

---

## Key definitions (plain language)

### Tokenization

Splitting text into model-readable units called tokens.

### Vocabulary

A mapping between each token and a unique integer ID.

### Embedding

A trainable vector representation of a token ID. Instead of reading words directly, neural networks read these vectors.

### Positional embedding

A vector that tells the model where each token is located in the sequence (word order information).

### Sliding window

A method that cuts long token streams into many short training examples for next-token prediction.

### `max_length`

How many tokens each training example contains.

### `stride`

How far the window moves before creating the next sample.

---

## Why embeddings encode meaning (NN connection)

Embeddings are part of a trainable neural network weight matrix.

During training, the model updates embedding vectors through backpropagation to reduce prediction error. Because of this:

- tokens that appear in similar contexts move closer in vector space,
- tokens used in different contexts stay farther apart.

So “meaning” emerges from usage patterns learned by the network, not from manually assigned rules.

For agentic systems, this is foundational because retrieval, matching, and semantic similarity depend on vector closeness in embedding space.

---

## Conclusions

1. Good preprocessing is not optional; it directly affects model quality.
2. Embeddings are the bridge between human language and neural computation.
3. Sliding-window overlap is a practical tradeoff:
	 - more overlap -> richer learning signal,
	 - but higher computational cost and potential redundancy.
4. Understanding these basics improves downstream work in RAG, retrieval, and agent pipelines.

---

## Prerequisites

- Python 3.10+
- Jupyter support (VS Code or JupyterLab)
- `torch`
- `tiktoken`

---

## Run

1. Open `Notebook.ipynb` and execute all cells.
2. Open `embeddings.ipynb` and execute all cells.
3. Review the experiment section and final conclusions.

---

## Repository structure

```text
.
├── Notebook.ipynb
├── embeddings.ipynb
├── the-verdict.txt
└── README.md
```

---

## References

- Sebastian Raschka, *Build a Large Language Model (From Scratch)*
- Official repository: https://github.com/rasbt/LLMs-from-scratch
- Chapter 2 notebook source:
	- https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/ch02.ipynb
- Chapter 2 text source:
	- https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- tiktoken repository: https://github.com/openai/tiktoken