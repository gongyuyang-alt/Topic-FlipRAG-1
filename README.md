# 🎯 Topic-FlipRAG: Topic-Oriented Adversarial Opinion Manipulation Attacks on RAG Models

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**Core implementation of Paper:** [**Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**](https://arxiv.org/abs/2502.01386)

This repository contains the implementation of **Topic-FlipRAG**, a two-stage adversarial attack framework targeting Retrieval-Augmented Generation (RAG) systems. The method illustrates how topic-specific knowledge perturbations can systematically manipulate LLM outputs in opinion-oriented tasks.

---

## 🔧 Key Features

- 🎯 **Topic-oriented trigger attacks** targeting multi-perspective generation
- 🧠 **Two-stage attack pipeline**:
  - Stage 1: Knowledge-guided adversarial sampling
  - Stage 2: Gradient-based trigger optimization
- 📏 Integrated **evaluation suite** for measuring stance/opinion drift

---

## 📁 Repository Structure

| File / Notebook | Description |
|------------------|-------------|
| `PROCON_data.json` | PROCON dataset used in the paper |
| `RAG_pipeline.ipynb` | RAG execution and evaluation pipeline |
| `Stage1_knowledge_guided_attack.ipynb` | Stage 1 attack guided by LLM knowledge |
| `Stage2_adversarial_trigger_generation.ipynb` | Stage 2 attack guided by NRM gradients |


---

## 🚀 Run in Colab (Recommended)

| Notebook | Colab Link |
|----------|------------|
| `RAG_pipeline` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gongyuyang-alt/Topic-FlipRAG-1/blob/main/RAG_pipeline.ipynb) |
| `Stage 1 - Knowledge Attack` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gongyuyang-alt/Topic-FlipRAG-1/blob/main/Stage1_knowledge_guided_attack.ipynb) |
| `Stage 2 - Trigger Optimization` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gongyuyang-alt/Topic-FlipRAG-1/blob/main/Stage2_adversarial_trigger_generation.ipynb) |

---



## License

MIT License (see [LICENSE](LICENSE) for details)

---

**Disclaimer**: This implementation is provided for research purposes only. Users must adhere to ethical AI guidelines and applicable laws when using this code.
