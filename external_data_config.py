#!/usr/bin/env python3
"""
外部数据源配置和缓存管理系统
"""
import json
import time
import hashlib
import requests
import feedparser
from pathlib import Path
from typing import List, Dict, Any, Optional
from external_data_preprocessor import ExternalDataPreprocessor
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_system import rag_system
from src.config import config
from loguru import logger

class DataCacheManager:
    """数据缓存管理器"""
    
    def __init__(self, cache_dir: str = "external_data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 为每个类别创建缓存目录
        self.categories = [
            "Cutting-Edge Tech & AI",
            "Business Models & Market Dynamics", 
            "Sustainability & Environmental Governance",
            "Social Change & Cultural Trends",
            "Life Sciences & Public Health",
            "Global Affairs & Future Governance"
        ]
        
        for category in self.categories:
            category_dir = self.cache_dir / self._safe_filename(category)
            category_dir.mkdir(exist_ok=True)
    
    def _safe_filename(self, text: str) -> str:
        """转换为安全的文件名"""
        return text.replace(" & ", "_and_").replace(" ", "_").lower()
    
    def _get_cache_key(self, source: str, query: str, category: str) -> str:
        """生成缓存键"""
        content = f"{source}_{query}_{category}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_cache_valid(self, cache_file: Path, max_age_hours: int = 24) -> bool:
        """检查缓存是否仍然有效"""
        if not cache_file.exists():
            return False
        
        cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return datetime.now() - cache_time < timedelta(hours=max_age_hours)
    
    def save_cache(self, source: str, query: str, category: str, data: List[Dict]) -> Path:
        """保存数据到缓存"""
        cache_key = self._get_cache_key(source, query, category)
        category_dir = self.cache_dir / self._safe_filename(category)
        cache_file = category_dir / f"{source}_{cache_key}.json"
        
        cache_data = {
            "source": source,
            "query": query,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已缓存 {len(data)} 条数据到 {cache_file}")
        return cache_file
    
    def load_cache(self, source: str, query: str, category: str) -> Optional[List[Dict]]:
        """从缓存加载数据"""
        cache_key = self._get_cache_key(source, query, category)
        category_dir = self.cache_dir / self._safe_filename(category)
        cache_file = category_dir / f"{source}_{cache_key}.json"
        
        if self.is_cache_valid(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                logger.info(f"从缓存加载 {len(cache_data['data'])} 条数据")
                return cache_data['data']
            except Exception as e:
                logger.error(f"加载缓存失败: {e}")
        
        return None

class ArxivDataSource:
    """ArXiv学术论文数据源"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.cache_manager = DataCacheManager()
    
    def _build_query(self, keywords: List[str], category: str) -> str:
        """构建ArXiv查询字符串"""
        # 根据类别调整查询
        category_queries = {
            "Cutting-Edge Tech & AI": "cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL",
            "Life Sciences & Public Health": "cat:q-bio.* OR cat:physics.med-ph",
            "Sustainability & Environmental Governance": "cat:physics.ao-ph OR cat:eess.SP",
        }
        
        base_query = category_queries.get(category, "")
        keyword_query = " OR ".join([f'all:"{kw}"' for kw in keywords])
        
        if base_query and keyword_query:
            return f"({base_query}) AND ({keyword_query})"
        elif base_query:
            return base_query
        else:
            return keyword_query
    
    def fetch(self, keywords: List[str], category: str, limit: int = 20) -> List[Dict[str, Any]]:
        """获取ArXiv论文数据"""
        query_key = "_".join(keywords)
        
        # 尝试从缓存加载
        cached_data = self.cache_manager.load_cache("arxiv", query_key, category)
        if cached_data:
            return cached_data
        
        try:
            query = self._build_query(keywords, category)
            params = {
                'search_query': query,
                'start': 0,
                'max_results': limit,
                'sortBy': 'lastUpdatedDate',
                'sortOrder': 'descending'
            }
            
            url = f"{self.base_url}?{urlencode(params)}"
            logger.info(f"获取ArXiv数据: {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # 解析XML响应
            root = ET.fromstring(response.content)
            entries = []
            
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
                summary_elem = entry.find('.//{http://www.w3.org/2005/Atom}summary')
                published_elem = entry.find('.//{http://www.w3.org/2005/Atom}published')
                
                if title_elem is not None and summary_elem is not None:
                    entries.append({
                        "title": title_elem.text.strip(),
                        "content": summary_elem.text.strip(),
                        "source": "ArXiv",
                        "date": published_elem.text[:10] if published_elem is not None else "",
                        "type": "academic_paper"
                    })
            
            # 缓存结果
            self.cache_manager.save_cache("arxiv", query_key, category, entries)
            logger.info(f"获取到 {len(entries)} 篇ArXiv论文")
            
            return entries
            
        except Exception as e:
            logger.error(f"获取ArXiv数据失败: {e}")
            return []

class NewsAPISource:
    """新闻API数据源（示例 - 需要API密钥）"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.cache_manager = DataCacheManager()
    
    def fetch(self, category: str, keywords: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """获取新闻数据"""
        if not self.api_key:
            logger.warning("NewsAPI密钥未配置，跳过新闻数据获取")
            return []
        
        query_key = "_".join(keywords)
        
        # 尝试从缓存加载
        cached_data = self.cache_manager.load_cache("news", query_key, category)
        if cached_data:
            return cached_data
        
        try:
            query = " OR ".join(keywords)
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': limit,
                'apiKey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            entries = []
            
            for article in data.get('articles', []):
                if article.get('title') and article.get('description'):
                    entries.append({
                        "title": article['title'],
                        "content": article['description'] + (article.get('content', '') or ''),
                        "source": article.get('source', {}).get('name', 'News'),
                        "date": article.get('publishedAt', '')[:10],
                        "type": "news_article"
                    })
            
            # 缓存结果
            self.cache_manager.save_cache("news", query_key, category, entries)
            logger.info(f"获取到 {len(entries)} 条新闻")
            
            return entries
            
        except Exception as e:
            logger.error(f"获取新闻数据失败: {e}")
            return []

class RSSFeedSource:
    """RSS源数据获取"""
    
    def __init__(self):
        self.cache_manager = DataCacheManager()
        
        # 各类别的RSS源（使用可用的源）
        self.category_feeds = {
            "Cutting-Edge Tech & AI": [
                "https://feeds.feedburner.com/TechCrunch",
                "https://feeds.feedburner.com/venturebeat/SZYF"
            ],
            "Business Models & Market Dynamics": [
                "https://feeds.feedburner.com/TechCrunch",
                "https://feeds.feedburner.com/venturebeat/SZYF"
            ],
            "Sustainability & Environmental Governance": [
                "https://feeds.feedburner.com/TechCrunch",
                "https://feeds.feedburner.com/venturebeat/SZYF"
            ],
            "Social Change & Cultural Trends": [
                "https://feeds.feedburner.com/TechCrunch",
                "https://feeds.feedburner.com/venturebeat/SZYF"
            ],
            "Life Sciences & Public Health": [
                "https://feeds.feedburner.com/TechCrunch", 
                "https://feeds.feedburner.com/venturebeat/SZYF"
            ],
            "Global Affairs & Future Governance": [
                "https://feeds.feedburner.com/TechCrunch",
                "https://feeds.feedburner.com/venturebeat/SZYF"
            ]
        }
    
    def fetch(self, category: str, keywords: List[str], limit: int = 15) -> List[Dict[str, Any]]:
        """获取RSS数据"""
        query_key = "_".join(keywords)
        
        # 尝试从缓存加载
        cached_data = self.cache_manager.load_cache("rss", query_key, category)
        if cached_data:
            return cached_data
        
        entries = []
        feeds = self.category_feeds.get(category, [])
        
        for feed_url in feeds:
            try:
                logger.info(f"获取RSS数据: {feed_url}")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:limit//len(feeds)]:
                    # 检查是否包含关键词
                    text_content = (entry.get('title', '') + ' ' + 
                                  entry.get('summary', '') + ' ' + 
                                  entry.get('description', '')).lower()
                    
                    if any(kw.lower() in text_content for kw in keywords):
                        entries.append({
                            "title": entry.get('title', ''),
                            "content": entry.get('summary', '') or entry.get('description', ''),
                            "source": feed.feed.get('title', 'RSS Feed'),
                            "date": entry.get('published', '')[:10] if 'published' in entry else '',
                            "type": "rss_article"
                        })
                
            except Exception as e:
                logger.error(f"获取RSS数据失败 {feed_url}: {e}")
        
        # 缓存结果
        if entries:
            self.cache_manager.save_cache("rss", query_key, category, entries)
        
        logger.info(f"获取到 {len(entries)} 条RSS数据")
        return entries

class ExternalDataManager:
    """外部数据管理器"""
    
    def __init__(self):
        self.arxiv_source = ArxivDataSource()
        self.news_source = NewsAPISource(getattr(config, 'NEWS_API_KEY', None))
        self.rss_source = RSSFeedSource()
        self.preprocessor = ExternalDataPreprocessor()  # 添加预处理器
        
        # 各类别的关键词配置（扩展了更多通用关键词）
        self.category_keywords = {
            "Cutting-Edge Tech & AI": [
                "artificial intelligence", "machine learning", "deep learning", 
                "neural networks", "AI technology", "computer vision", "NLP",
                "generative AI", "large language model", "robotics", "AI", 
                "technology", "tech", "innovation", "digital", "algorithm",
                "automation", "software", "computing", "data science"
            ],
            "Business Models & Market Dynamics": [
                "business model", "digital transformation", "e-commerce", 
                "fintech", "market dynamics", "startup", "venture capital",
                "platform economy", "sharing economy", "subscription model",
                "business", "market", "economy", "finance", "investment", 
                "commerce", "company", "corporate", "industry", "enterprise"
            ],
            "Sustainability & Environmental Governance": [
                "sustainability", "climate change", "renewable energy", 
                "carbon neutral", "ESG", "green technology", "environmental policy",
                "circular economy", "clean energy", "carbon footprint",
                "environment", "climate", "energy", "green", "sustainable",
                "carbon", "emission", "renewable", "eco", "conservation"
            ],
            "Social Change & Cultural Trends": [
                "social media", "digital culture", "remote work", 
                "urbanization", "demographic change", "social innovation",
                "cultural trends", "lifestyle change", "digital society",
                "social", "culture", "society", "community", "lifestyle",
                "trend", "behavior", "demographic", "youth", "family"
            ],
            "Life Sciences & Public Health": [
                "public health", "biotechnology", "medical technology", 
                "healthcare", "telemedicine", "precision medicine",
                "biotech innovation", "health policy", "medical research",
                "health", "medical", "medicine", "hospital", "treatment",
                "disease", "pandemic", "vaccine", "drug", "therapy"
            ],
            "Global Affairs & Future Governance": [
                "international relations", "global governance", "geopolitics", 
                "trade policy", "cybersecurity", "digital governance",
                "international cooperation", "global challenges", "diplomacy",
                "global", "international", "politics", "policy", "government",
                "governance", "security", "trade", "diplomatic", "world"
            ]
        }
    
    def cache_all_categories(self, limit_per_source: int = 15):
        """为所有类别缓存外部数据"""
        logger.info("开始缓存所有类别的外部数据...")
        
        for category in self.category_keywords:
            logger.info(f"正在处理类别: {category}")
            self.cache_category_data(category, limit_per_source)
            time.sleep(1)  # 避免请求过于频繁
        
        logger.info("所有类别数据缓存完成")
    
    def cache_category_data(self, category: str, limit_per_source: int = 15):
        """缓存指定类别的数据"""
        keywords = self.category_keywords.get(category, [])
        all_data = []
        
        # 从ArXiv获取学术数据
        try:
            arxiv_data = self.arxiv_source.fetch(keywords[:5], category, limit_per_source)
            all_data.extend(arxiv_data)
        except Exception as e:
            logger.error(f"ArXiv数据获取失败: {e}")
        
        # 从RSS获取新闻数据
        try:
            rss_data = self.rss_source.fetch(category, keywords[:5], limit_per_source)
            all_data.extend(rss_data)
        except Exception as e:
            logger.error(f"RSS数据获取失败: {e}")
        
        # 从NewsAPI获取数据（如果有API密钥）
        try:
            news_data = self.news_source.fetch(category, keywords[:3], limit_per_source)
            all_data.extend(news_data)
        except Exception as e:
            logger.error(f"NewsAPI数据获取失败: {e}")
        
        # 预处理数据（HTML清理和智能总结）
        if all_data:
            try:
                logger.info(f"开始预处理 {len(all_data)} 条数据...")
                all_data = self.preprocessor.preprocess_batch(all_data)
                
                # 显示处理统计
                stats = self.preprocessor.get_processing_stats(all_data)
                logger.info(f"预处理完成: {stats}")
                
            except Exception as e:
                logger.error(f"数据预处理失败: {e}")
        
        logger.info(f"类别 {category} 共获取 {len(all_data)} 条数据")
        return all_data
    
    def vectorize_cached_data(self):
        """将缓存的数据向量化并添加到RAG系统"""
        logger.info("开始向量化缓存数据...")
        
        cache_dir = Path("external_data_cache")
        if not cache_dir.exists():
            logger.warning("缓存目录不存在，请先运行数据缓存")
            return
        
        for category_dir in cache_dir.iterdir():
            if category_dir.is_dir():
                # 正确转换目录名到类别名
                dir_name = category_dir.name
                if dir_name == "cutting-edge_tech_and_ai":
                    category = "Cutting-Edge Tech & AI"
                elif dir_name == "business_models_and_market_dynamics":
                    category = "Business Models & Market Dynamics"
                elif dir_name == "sustainability_and_environmental_governance":
                    category = "Sustainability & Environmental Governance"
                elif dir_name == "social_change_and_cultural_trends":
                    category = "Social Change & Cultural Trends"
                elif dir_name == "life_sciences_and_public_health":
                    category = "Life Sciences & Public Health"
                elif dir_name == "global_affairs_and_future_governance":
                    category = "Global Affairs & Future Governance"
                else:
                    continue
                
                logger.info(f"向量化类别: {category}")
                documents = []
                
                # 读取该类别下的所有缓存文件
                for cache_file in category_dir.glob("*.json"):
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                        
                        for item in cache_data.get('data', []):
                            # 格式化为文档块
                            doc_text = f"""标题: {item.get('title', '')}
来源: {item.get('source', '')} ({item.get('date', '')})
类型: {item.get('type', '')}
内容: {item.get('content', '')}"""
                            documents.append(doc_text)
                    
                    except Exception as e:
                        logger.error(f"读取缓存文件失败 {cache_file}: {e}")
                
                # 添加到RAG系统
                if documents:
                    # 分块处理（每块最大1000字符）
                    chunked_docs = self._chunk_documents(documents)
                    rag_system.add_documents(category, chunked_docs)
                    logger.info(f"已向量化 {len(chunked_docs)} 个文档块到类别 {category}")
        
        logger.info("数据向量化完成")
    
    def _chunk_documents(self, documents: List[str], max_chunk_size: int = 1000) -> List[str]:
        """将文档分块"""
        chunks = []
        
        for doc in documents:
            if len(doc) <= max_chunk_size:
                chunks.append(doc)
            else:
                # 按段落分割
                paragraphs = doc.split('\n')
                current_chunk = ""
                
                for para in paragraphs:
                    if len(current_chunk) + len(para) <= max_chunk_size:
                        current_chunk += para + "\n"
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = para + "\n"
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
        
        return chunks

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="外部数据源管理")
    parser.add_argument("--action", choices=["cache", "vectorize", "both", "stats"], required=True)
    parser.add_argument("--category", help="指定类别")
    parser.add_argument("--limit", type=int, default=15, help="每个数据源的限制数量")
    
    args = parser.parse_args()
    
    manager = ExternalDataManager()
    
    if args.action == "cache":
        if args.category:
            manager.cache_category_data(args.category, args.limit)
        else:
            manager.cache_all_categories(args.limit)
    
    elif args.action == "vectorize":
        manager.vectorize_cached_data()
    
    elif args.action == "both":
        if args.category:
            manager.cache_category_data(args.category, args.limit)
        else:
            manager.cache_all_categories(args.limit)
        time.sleep(2)
        manager.vectorize_cached_data()
    
    elif args.action == "stats":
        # 显示缓存统计
        cache_dir = Path("external_data_cache")
        if cache_dir.exists():
            total_files = 0
            total_items = 0
            
            print("=== 外部数据缓存统计 ===")
            for category_dir in cache_dir.iterdir():
                if category_dir.is_dir():
                    files = list(category_dir.glob("*.json"))
                    items = 0
                    
                    for file in files:
                        try:
                            with open(file, 'r') as f:
                                data = json.load(f)
                                items += len(data.get('data', []))
                        except:
                            pass
                    
                    category_name = category_dir.name.replace("_and_", " & ").replace("_", " ").title()
                    print(f"{category_name}: {len(files)} 个缓存文件, {items} 条数据")
                    
                    total_files += len(files)
                    total_items += items
            
            print(f"\n总计: {total_files} 个文件, {total_items} 条数据")
        else:
            print("❌ 缓存目录不存在")

if __name__ == "__main__":
    main()
