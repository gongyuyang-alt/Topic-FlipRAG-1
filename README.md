# 🎯 Topic-FlipRAG: Topic-Oriented Adversarial Opinion Manipulation Attacks on RAG Models

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**Core implementation of Paper:**  
[**Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**](https://arxiv.org/abs/2502.01386)

---

## 🧠 Overview

This repository contains the full implementation of **Topic-FlipRAG**, a two-stage adversarial attack framework targeting Retrieval-Augmented Generation (RAG) systems. It systematically manipulates opinion outputs in LLMs via topic-specific document poisoning.

### 📂 Repository Structure

1. **Stage 1: Knowledge-Guided Attack**  
   Includes the core logic for injecting topic-specific misinformation by modifying documents based on LLM-inferred knowledge (`doc_know` generation).

2. **Stage 2: Adversarial Trigger Generation**  
   Optimizes minimal triggers to attach to `doc_know` for final poisoned documents. Includes formatting scripts for downstream poisoning tasks.

3. **RAG Pipeline**  
   Builds a full RAG system (retriever + database + LLM) and evaluates poisoning effects. Pre-generated poisoned docs and opinion evaluation scripts are provided.

4. **Data**  
   Contains:
   - `PROCON_data.json` (the base dataset)
   - Sample poisoned documents (`data/Topic-FlipRAG_society_CON_passges/`)
   - Generated adversarial examples for convenient testing

---

## 🚀 Quick Start

This project is **Colab-friendly**. You only need to replace paths in the Jupyter notebooks to point to the corresponding files in the `data/` directory. **OpenAI API** is required for Stage 1 and the RAG pipeline.

---

### 🔧 Colab Notebooks

1. **Stage 1 – Knowledge Attack**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gongyuyang-alt/Topic-FlipRAG-1/blob/main/Stage1_knowledge_guided_attack.ipynb)  
   ⮕ Replace `path_know = 'doc_path_from_stage_1_know_attack.json'` with  
   `data/know_attack_data_3_0.json`

2. **Stage 2 – Trigger Optimization**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gongyuyang-alt/Topic-FlipRAG-1/blob/main/Stage2_adversarial_trigger_generation.ipynb)  
   ⮕ Format and optimize triggers based on Stage 1 outputs.

3. **RAG Pipeline – Execution & Evaluation**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gongyuyang-alt/Topic-FlipRAG-1/blob/main/RAG_pipeline.ipynb)  
   ⮕ Replace `result_path` in `load_data()` with a file path from  
   `data/Topic-FlipRAG_society_CON_passges/` (we recommend using Google Drive for hosting large files)

---

## 💡 Note

To facilitate quick testing, we provide a subset of adversarial outputs in the `"Society & Culture"` domain.  
You can modify the code to load the full dataset (`PROCON_data.json`) for a complete evaluation.

---

## 📎 Citation

If you find this work useful, please cite:

```bibtex
@article{gong2024topicfliprag,
  title={Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models},
  author={Gong, Jian and Liu, Jiawei and Lu, Wei},
  journal={arXiv preprint arXiv:2502.01386},
  year={2024}
}

