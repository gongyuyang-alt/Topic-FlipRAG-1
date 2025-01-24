import numpy as np
from transformers import AutoModel, AutoTokenizer
# from sentence_transformers import SentenceTransformer, util
from numpy.linalg import norm
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import torch
from torch import nn
from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
import torch
import torch.nn as nn
from typing import List
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


class localEmbedding_sentence_dpr(nn.Module):
    def __init__(self, path: str = '', device: str = 'cuda', language_code: str = 'en_XX'):
        super().__init__()
        self.device = device
        # 加载指定的 DPR 模型
        self.model = SentenceTransformer(path, device=device)
        # 激活特定语言的适配器
        self.model[0].auto_model.set_default_language(language_code)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 替换掉文本中的换行符
        texts = [text.replace("\n", " ") for text in texts]
        # 使用模型编码，归一化嵌入
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings


class  localEmbedding_sentence_ance(nn.Module):
    def __init__(self, path: str = '', device: str = 'cuda'):
        super().__init__()
        # 加载指定的 SentenceTransformer 模型
        self.model = SentenceTransformer('/DR/ance', device=device)
        self.device = device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        # 替换掉文本中的换行符
        texts = [text.replace("\n", " ") for text in texts]
        # 使用模型编码，归一化嵌入
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings


class localEmbedding_sentence_contriever(nn.Module):
    def __init__(self, path: str = '', device: str = 'cuda'):
        super().__init__()
        # Load specified model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device)
        
    def mean_pooling(self, model_output, attention_mask):
        # Mean Pooling considering attention mask
        token_embeddings = model_output[0]  # First element is all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        return sum_embeddings / torch.clamp(sum_mask, min=1e-9)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        # Encode the text
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,  # Ensure consistent max_length
            return_tensors='pt'
        ).to(self.device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Perform mean pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        # Do not normalize embeddings
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # Convert embeddings to lists of floats
        return embeddings.cpu().numpy().tolist()
        
    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings













class localEmbedding(nn.Module):
    def __init__(self, path:str='', device:str=''):
        super().__init__()
        self.embedding = AutoModel.from_pretrained(path, add_pooling_layer = False, output_hidden_states=False)
        self.embedding.to(device)
        self.pool_type = 'cls'
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # self.tokenizer.to(device)
        self.decive = device
    
    def pooling(self, token_embeddings,input):#添加平均池化层，效果比使用CLS可能更好
        output_vectors = []
        #attention_mask
        attention_mask = input['attention_mask']
        #[B,L]------>[B,L,1]------>[B,L,768],矩阵的值是0或者1
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        #这里做矩阵点积，就是对元素相乘(序列中padding字符，通过乘以0给去掉了)[B,L,768]
        t = token_embeddings * input_mask_expanded
        #[B,768]
        sum_embeddings = torch.sum(t, 1)
 
        # [B,768],最大值为seq_len
        sum_mask = input_mask_expanded.sum(1)
        #限定每个元素的最小值是1e-9，保证分母不为0
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        #得到最后的具体embedding的每一个维度的值——元素相除
        output_vectors.append(sum_embeddings / sum_mask)
 
        output_vector = torch.cat(output_vectors, 1)
        return  output_vector
    
    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        """
        length_sorted_idx = np.argsort([-len(sen) for sen in texts])
        texts = [texts[idx] for idx in length_sorted_idx]
       """
        input_ids = self.tokenizer(texts, max_length=256, padding = "max_length", truncation=True, return_tensors='pt')#注意截断长度
        # input_ids.pop('attention_mask')
        input_ids = input_ids.to(self.decive)
        embeddings = self.embedding(**input_ids)#不能仅仅考虑input_ids，不然效果可能不好,对pad进行处理
        if self.pool_type == 'mean':
            token_embeddings = embeddings[0]
            embeddings = self.pooling(token_embeddings, input_ids)
        else:
            # print("CLS!")
            embeddings = embeddings[0][:, 0, :]
        
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)#规范化
        # embeddings = embeddings.last_hidden_state[:,0,:]
        """
        embeddings = self.embedding.encode(texts)"""
        # print("HF embed:", embeddings, embeddings.shape)
        
        return embeddings.tolist()
    

