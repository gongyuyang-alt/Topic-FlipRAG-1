# 🎯 Topic-FlipRAG: Topic-Oriented Adversarial Opinion Manipulation Attacks on RAG Models

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Core implementation of Paper:**  
[**Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**](https://arxiv.org/abs/2502.01386)



## 🧠 Overview

This repository contains the full implementation of **Topic-FlipRAG**, a novel black-box adversarial attack framework against Retrieval-Augmented Generation (RAG) systems. By leveraging general language knowledge and reverse-gradient signals, it optimizes a small number of poisoned documents to effectively flip the opinion stance of the RAG system across an entire set of topic-related queries.

### 📂 Repository Structure

1. **Stage1_knowledge_guided_attack.ipynb**  
   Includes the core implementation of the knowledge-guided attack, which leverages LLM-inferred general knowledge to perform multi-granularity document modifications (`doc_know` generation).

2. **Stage2_adversarial_trigger_generation.ipynb**  
   Optimizes minimal triggers to attach to `doc_know` for final poisoned documents. Includes formatting scripts for downstream poisoning tasks.

3. **RAG_pipeline.ipynb**  
   Builds a full RAG system (retriever + database + LLM) and evaluates poisoning effects. Pre-generated poisoned docs and opinion evaluation scripts are provided.

4. **Data**  
   - `PROCON_data.json`: The opinion dataset used in the paper.  
   - Example poisoned documents: `data/Topic-FlipRAG_society_CON_passges/` — used in `RAG_pipeline.ipynb`.  
   - Example `doc_know` file: `data/know_attack_data_3_0.json` — used in `Stage2_adversarial_trigger_generation.ipynb` to demonstrate the trigger generation process.


## 🚀 Quick Start

This project is **Colab-friendly**. You only need to replace paths in the Jupyter notebooks to point to the corresponding files in the `data/` directory. **OpenAI API** is required for Stage1_knowledge_guided_attack.ipynb and the RAG_pipeline.ipynb.

### 🔧 Colab Notebooks

1. **Stage 1 – Knowledge-guided Attack**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gongyuyang-alt/Topic-FlipRAG-1/blob/main/Stage1_knowledge_guided_attack.ipynb)
   💡 *Recommended GPU: T4*

3. **Stage 2 – Adversarial Trigger Generation**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gongyuyang-alt/Topic-FlipRAG-1/blob/main/Stage2_adversarial_trigger_generation.ipynb)  
   ➤ Optimizes adversarial triggers based on Stage 1 outputs.  
   ⮕ To skip Stage 1, directly set: (This is a pre-generated example for fast evaluation.)  
   ```python
   path_know = 'data/know_attack_data_3_0.json'
   ```
   💡 *Recommended GPU: T4*

4. **RAG Pipeline – Execution & Evaluation**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gongyuyang-alt/Topic-FlipRAG-1/blob/main/RAG_pipeline.ipynb)  
   ➤ Runs the full RAG system and evaluates poisoned document impact.  
   ⮕ To skip Stage 2, replace `result_path` in `load_data()` with a sample file from:  
   ```
   data/Topic-FlipRAG_society_CON_passges/
   ```
   (These are pre-generated adversarial examples in the *society* domain targeting the CON stance.)  
   💡 *Recommended GPU: A100*  
   🔁 *Use Google Drive for hosting large poisoned files if needed.*


## 💡 Note
To facilitate quick testing, we provide a subset of poisoned documents located in `data/Topic-FlipRAG_society_CON_passges/`, specifically targeting the `"Society & Culture"` domain with a CON (oppose) stance.  For full-scale evaluation, you can modify the code to load the entire dataset from `PROCON_data.json`.



## 📎 Citation

If you find this work useful, please cite:

```bibtex
@article{gong2025topic,
  title={Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models},
  author={Gong, Yuyang and Chen, Zhuo and Chen, Miaokun and Yu, Fengchang and Lu, Wei and Wang, Xiaofeng and Liu, Xiaozhong and Liu, Jiawei},
  journal={arXiv preprint arXiv:2502.01386},
  year={2025}
}

