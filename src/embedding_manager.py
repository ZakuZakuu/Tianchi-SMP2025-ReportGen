"""
多提供商Embedding支持
"""
import requests
import numpy as np
from typing import List, Union, Any
import logging

logger = logging.getLogger(__name__)

class BaseEmbedding:
    """Embedding基类"""
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        raise NotImplementedError
    
    def __call__(self, texts):
        return self.encode(texts)

class SiliconFlowEmbedding(BaseEmbedding):
    """SiliconFlow Embedding API"""
    
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        
        logger.info(f"初始化SiliconFlow Embedding: {model_name}")
        
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                embeddings_array = np.array(embeddings)
                
                if single_text:
                    return embeddings_array[0]
                return embeddings_array
            else:
                raise Exception(f"API请求失败: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"SiliconFlow Embedding API调用失败: {e}")
            raise

class LocalEmbedding(BaseEmbedding):
    """本地Sentence Transformers模型"""
    
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"初始化本地Embedding模型: {model_name}")
        except ImportError:
            raise ImportError("需要安装 sentence-transformers: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"加载本地模型失败: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.model.encode(texts)

class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding API"""
    
    def __init__(self, api_key: str, base_url: str, model_name: str):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            self.model_name = model_name
            logger.info(f"初始化OpenAI Embedding: {model_name}")
        except ImportError:
            raise ImportError("需要安装 openai: pip install openai")
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            embeddings_array = np.array(embeddings)
            
            if single_text:
                return embeddings_array[0]
            return embeddings_array
            
        except Exception as e:
            logger.error(f"OpenAI Embedding API调用失败: {e}")
            raise

class EmbeddingFactory:
    """Embedding工厂类"""
    
    @staticmethod
    def create_embedding(provider: str, model_name: str, api_key: str = None, base_url: str = None) -> BaseEmbedding:
        """创建embedding实例"""
        
        if provider == "siliconflow":
            if not api_key:
                raise ValueError("SiliconFlow需要API密钥")
            return SiliconFlowEmbedding(api_key, base_url, model_name)
        
        elif provider == "openai":
            if not api_key:
                raise ValueError("OpenAI需要API密钥")
            return OpenAIEmbedding(api_key, base_url, model_name)
        
        elif provider == "local":
            return LocalEmbedding(model_name)
        
        else:
            raise ValueError(f"不支持的embedding提供商: {provider}")

def get_embedding_function(provider: str, model_name: str, api_key: str = None, base_url: str = None):
    """获取embedding函数，兼容Chroma"""
    embedding_instance = EmbeddingFactory.create_embedding(provider, model_name, api_key, base_url)
    
    # 创建Chroma兼容的wrapper
    class ChromaEmbeddingWrapper:
        def __init__(self, embedding_instance, provider, model_name):
            self.embedding_instance = embedding_instance
            self._name = f"{provider}:{model_name}"  # 存储名称
        
        def name(self):  # Chroma需要这个方法
            return self._name
        
        def __call__(self, input):  # Chroma需要input参数名
            if isinstance(input, str):
                input = [input]
            
            embeddings = self.embedding_instance.encode(input)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            return embeddings.tolist()
    
    return ChromaEmbeddingWrapper(embedding_instance, provider, model_name)
