"""
RAG系统核心模块
"""
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from loguru import logger

from .config import config
from .embedding_manager import get_embedding_function

class RAGSystem:
    """RAG检索增强生成系统"""
    
    def __init__(self):
        self.db_path = Path(config.CHROMA_DB_PATH)
        self.db_path.mkdir(exist_ok=True)
        
        # 初始化Chroma客户端
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(allow_reset=True)
        )
        
        # 初始化embedding函数
        try:
            # 从配置中解析embedding提供商和模型
            embedding_provider = config.EMBEDDING_PROVIDER
            embedding_model = config.EMBEDDING_MODEL
            
            if ":" in embedding_model:
                provider, model_name = embedding_model.split(":", 1)
                embedding_provider = provider
            else:
                model_name = embedding_model
                
            # 准备embedding配置
            embedding_config = {
                "provider": embedding_provider,
                "model_name": model_name
            }
            
            # 根据provider添加相应的API配置
            if embedding_provider == "siliconflow":
                embedding_config.update({
                    "api_key": config.SILICONFLOW_API_KEY,
                    "base_url": config.SILICONFLOW_BASE_URL
                })
            elif embedding_provider == "openai":
                embedding_config.update({
                    "api_key": config.OPENAI_API_KEY,
                    "base_url": config.OPENAI_BASE_URL
                })
            
            self.embedding_function = get_embedding_function(**embedding_config)
            logger.info(f"已初始化embedding: {embedding_provider}:{model_name}")
            
        except Exception as e:
            logger.warning(f"初始化配置的embedding失败: {e}, 使用默认local模型")
            self.embedding_function = get_embedding_function("local", "all-MiniLM-L6-v2")
        
        # 为每个类别创建collection
        self.collections = {}
        self._init_collections()
    
    def _init_collections(self):
        """初始化各类别的collection"""
        for category_name, category_key in config.CATEGORIES.items():
            try:
                collection = self.chroma_client.get_or_create_collection(
                    name=f"smp_{category_key}",
                    embedding_function=self.embedding_function,
                    metadata={"category": category_name}
                )
                self.collections[category_name] = collection
                logger.info(f"已初始化collection: {category_name}")
            except Exception as e:
                logger.error(f"初始化collection失败 {category_name}: {e}")
    
    def add_documents(self, category: str, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """添加文档到指定类别的知识库"""
        if category not in self.collections:
            logger.error(f"未知类别: {category}")
            return False
        
        try:
            collection = self.collections[category]
            
            # 准备metadata
            if metadatas is None:
                metadatas = [{"source": "knowledge_base", "category": category}] * len(documents)
            
            # 过滤过长文档（限制在400字符以内，约512 tokens）
            filtered_docs = []
            filtered_metas = []
            for i, doc in enumerate(documents):
                if len(doc) <= 400:
                    filtered_docs.append(doc)
                    filtered_metas.append(metadatas[i] if i < len(metadatas) else metadatas[0])
                else:
                    # 截断过长文档
                    truncated = doc[:400]
                    filtered_docs.append(truncated)
                    filtered_metas.append(metadatas[i] if i < len(metadatas) else metadatas[0])
                    logger.warning(f"文档过长已截断: {len(doc)} -> 400字符")
            
            # 分批处理，每批最多16个文档（留出安全边界）
            batch_size = 16
            existing_count = collection.count()
            total_added = 0
            
            for i in range(0, len(filtered_docs), batch_size):
                batch_docs = filtered_docs[i:i+batch_size]
                batch_metas = filtered_metas[i:i+batch_size]
                
                # 生成文档ID
                batch_ids = [f"{config.CATEGORIES[category]}_{existing_count + total_added + j}" 
                           for j in range(len(batch_docs))]
                
                # 添加到collection
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                
                total_added += len(batch_docs)
                logger.debug(f"已添加批次 {i//batch_size + 1}: {len(batch_docs)} 个文档")
            
            logger.info(f"已添加 {total_added} 个文档到 {category}")
            return True
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    def retrieve(self, query: str, category: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """从指定类别检索相关文档"""
        if category not in self.collections:
            logger.error(f"未知类别: {category}")
            return []
        
        try:
            collection = self.collections[category]
            
            # 执行检索
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 格式化结果
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    retrieved_docs.append({
                        'text': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            logger.info(f"从 {category} 检索到 {len(retrieved_docs)} 个相关文档")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, int]:
        """获取各collection的统计信息"""
        stats = {}
        for category_name, collection in self.collections.items():
            try:
                stats[category_name] = collection.count()
            except:
                stats[category_name] = 0
        return stats
    
    def load_sample_data(self):
        """加载示例数据（用于测试）"""
        # 检查是否已有数据，避免重复加载
        stats = self.get_collection_stats()
        if any(count > 0 for count in stats.values()):
            logger.info("检测到已有数据，跳过示例数据加载")
            return
            
        sample_data = {
            "Cutting-Edge Tech & AI": [
                "人工智能技术在2025年呈现出前所未有的发展势头，特别是在大语言模型、计算机视觉和机器人技术领域。",
                "可解释AI(XAI)技术正成为AI系统部署的关键要求，特别是在医疗、金融等高风险领域。",
                "数字孪生技术在智慧城市建设中发挥着重要作用，能够实现城市系统的实时监控和预测。"
            ],
            "Business Models & Market Dynamics": [
                "平台经济模式在全球范围内继续演进，但监管政策的收紧对其发展带来新的挑战。",
                "订阅经济模式正在从软件行业扩展到更多传统行业，改变着消费者的消费习惯。",
                "ESG投资理念日益成为企业战略规划和投资决策的重要考量因素。"
            ],
            # 其他类别的示例数据...
        }
        
        for category, docs in sample_data.items():
            if category in self.collections:
                self.add_documents(category, docs)
        
        logger.info("已加载示例数据到知识库")

# 全局RAG系统实例
rag_system = RAGSystem()
