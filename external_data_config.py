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
from src.multi_provider_model import MultiProviderModelManager
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
    
    def __init__(self, cache_manager: DataCacheManager = None):
        self.base_url = "http://export.arxiv.org/api/query"
        self.cache_manager = cache_manager or DataCacheManager()
    
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
    
    def fetch(self, keywords: List[str], category: str, limit: int = 20, save_individual_cache: bool = True) -> List[Dict[str, Any]]:
        """获取ArXiv论文数据"""
        query_key = "_".join(keywords)
        
        # 尝试从缓存加载（仅在允许单独缓存时）
        if save_individual_cache:
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
            
            # 缓存结果（仅在允许单独缓存时）
            if save_individual_cache:
                self.cache_manager.save_cache("arxiv", query_key, category, entries)
            logger.info(f"获取到 {len(entries)} 篇ArXiv论文")
            
            return entries
            
        except Exception as e:
            logger.error(f"获取ArXiv数据失败: {e}")
            return []

class NewsAPISource:
    """Event Registry API数据源"""
    
    def __init__(self, api_key: Optional[str] = None, cache_manager: DataCacheManager = None):
        self.api_key = api_key
        self.base_url = "http://eventregistry.org/api/v1/article/getArticles"
        self.cache_manager = cache_manager or DataCacheManager()
    
    def fetch(self, category: str, keywords: List[str], limit: int = 20, save_individual_cache: bool = True) -> List[Dict[str, Any]]:
        """获取新闻数据"""
        if not self.api_key:
            logger.warning("Event Registry API密钥未配置，跳过新闻数据获取")
            return []
        
        query_key = "_".join(keywords)
        
        # 尝试从缓存加载（仅在允许单独缓存时）
        if save_individual_cache:
            cached_data = self.cache_manager.load_cache("news", query_key, category)
            if cached_data:
                return cached_data
        
        try:
            # 构建Event Registry API请求体
            request_body = {
                "action": "getArticles",
                "keyword": keywords,
                "keywordOper": "or",  # 使用OR操作符
                "lang": ["eng"],  # 只要英文文章
                "articlesPage": 1,
                "articlesCount": min(limit, 100),  # Event Registry最多100篇
                "articlesSortBy": "date",
                "articlesSortByAsc": False,
                "articleBodyLen": -1,  # 获取完整文章内容
                "dataType": ["news"],
                "forceMaxDataTimeWindow": 31,  # 限制最近31天
                "resultType": "articles",
                "apiKey": self.api_key
            }
            
            response = requests.post(self.base_url, json=request_body, timeout=30, headers={
                'User-Agent': 'Tianchi-SMP2025-ReportGen/1.0',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            })
            response.raise_for_status()
            
            data = response.json()
            entries = []
            
            # Event Registry API 返回格式
            articles = data.get('articles', {}).get('results', [])
            
            for article in articles:
                if article.get('title') and article.get('body'):
                    entries.append({
                        "title": article['title'],
                        "content": article['body'],
                        "source": article.get('source', {}).get('title', 'Event Registry'),
                        "date": article.get('date', ''),
                        "type": "news_article",
                        "url": article.get('url', ''),
                        "lang": article.get('lang', 'eng')
                    })
            
            # 缓存结果（仅在允许单独缓存时）
            if save_individual_cache:
                self.cache_manager.save_cache("news", query_key, category, entries)
            logger.info(f"从Event Registry获取到 {len(entries)} 条新闻")
            
            return entries
            
        except Exception as e:
            logger.error(f"获取新闻数据失败: {e}")
            return []

class RSSFeedSource:
    """RSS源数据获取"""
    
    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager or DataCacheManager()
        
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
    
    def fetch(self, category: str, keywords: List[str], limit: int = 15, save_individual_cache: bool = True) -> List[Dict[str, Any]]:
        """获取RSS数据"""
        query_key = "_".join(keywords)
        
        # 尝试从缓存加载（仅在允许单独缓存时）
        if save_individual_cache:
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
                    # 检查是否包含关键词 - RSS使用更宽松的匹配逻辑
                    text_content = (entry.get('title', '') + ' ' + 
                                  entry.get('summary', '') + ' ' + 
                                  entry.get('description', '')).lower()
                    
                    # RSS特殊的匹配逻辑：更宽松，包含通用术语
                    keyword_matches = 0
                    matched_keywords = []
                    
                    # 为RSS创建更通用的关键词
                    rss_keywords = []
                    for kw in keywords:
                        kw_lower = kw.lower().strip()
                        if len(kw_lower) < 3:
                            continue
                        
                        # 添加原关键词
                        rss_keywords.append(kw_lower)
                        
                        # 对于复合关键词，也添加单独的词
                        if ' ' in kw_lower:
                            for part in kw_lower.split():
                                if len(part) >= 3 and part not in ['and', 'the', 'for', 'with', 'from']:
                                    rss_keywords.append(part)
                    
                    # 添加一些通用的相关词汇（基于类别）
                    if category == "Cutting-Edge Tech & AI":
                        rss_keywords.extend(['ai', 'artificial intelligence', 'machine learning', 'technology', 'innovation'])
                    elif category == "Business Models & Market Dynamics":
                        rss_keywords.extend(['business', 'market', 'economy', 'finance', 'investment'])
                    elif category == "Sustainability & Environmental Governance":
                        rss_keywords.extend(['sustainability', 'environment', 'green', 'climate', 'renewable'])
                    
                    # 去重
                    rss_keywords = list(set(rss_keywords))
                    
                    for kw in rss_keywords:
                        if kw in text_content:
                            if len(kw) > 6:  # 长关键词权重更高
                                keyword_matches += 2
                            else:
                                keyword_matches += 1
                            matched_keywords.append(kw)
                    
                    # RSS使用更宽松的阈值：只要有匹配就接受
                    if keyword_matches >= 1:
                        entries.append({
                            "title": entry.get('title', ''),
                            "content": entry.get('summary', '') or entry.get('description', ''),
                            "source": feed.feed.get('title', 'RSS Feed'),
                            "date": entry.get('published', '')[:10] if 'published' in entry else '',
                            "type": "rss_article"
                        })
                
            except Exception as e:
                logger.error(f"获取RSS数据失败 {feed_url}: {e}")
        
        # 缓存结果（仅在允许单独缓存时）
        if entries and save_individual_cache:
            self.cache_manager.save_cache("rss", query_key, category, entries)
        
        logger.info(f"获取到 {len(entries)} 条RSS数据")
        return entries

class ExternalDataManager:
    """外部数据管理器"""
    
    def __init__(self):
        # 首先初始化缓存管理器
        self.cache_manager = DataCacheManager()
        
        # 然后初始化各个数据源，传入缓存管理器
        self.arxiv_source = ArxivDataSource(self.cache_manager)
        self.news_source = NewsAPISource(getattr(config, 'NEWS_API_KEY', None), self.cache_manager)
        self.rss_source = RSSFeedSource(self.cache_manager)
        self.preprocessor = ExternalDataPreprocessor()  # 添加预处理器
        
        # 初始化关键词提取模型（使用便宜的总结模型）
        self.keyword_extractor = self._init_summary_model()
        logger.info("初始化关键词提取模型（总结模型）")
        
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
    
    def _init_summary_model(self):
        """初始化总结模型用于关键词提取"""
        # 创建一个独立的模型管理器，专门用于总结任务
        from src.multi_provider_model import MultiProviderModelManager
        
        # 临时修改配置使用总结模型
        original_primary = config.PRIMARY_MODEL
        config.PRIMARY_MODEL = config.SUMMARY_MODEL
        
        manager = MultiProviderModelManager()
        
        # 恢复原配置
        config.PRIMARY_MODEL = original_primary
        
        return manager
    
    def extract_keywords_batch(self, questions_data: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """批量提取多个题目的关键词"""
        
        if not questions_data:
            return {}
        
        # 构建批量提取prompt
        batch_prompt = """请为以下比赛题目批量提取关键词。对每个题目提取3-6个最重要的英文关键词，用于国际数据源检索。

严格要求:
1. 关键词必须是英文，便于国际数据源检索
2. 必须是核心技术术语、概念名词或专业术语
3. 严禁使用过于通用的词汇（如"report", "analysis", "study", "technology", "system"等）
4. 严禁使用单字母缩写（如单独的"AI"），应使用完整术语（如"artificial intelligence", "explainable AI"）
5. 优先选择多词组合的专业术语，确保检索精确性
6. 关键词应该能够有效区分相关和不相关的内容

示例：
- 好的关键词："explainable AI", "model interpretability", "algorithmic transparency"
- 不好的关键词："AI", "technology", "report", "analysis"

请按以下格式返回，每行一个题目的结果：
ID1: keyword1, keyword2, keyword3, ...
ID2: keyword1, keyword2, keyword3, ...

题目列表:
"""
        
        # 添加题目到prompt
        for item in questions_data:
            batch_prompt += f"{item['id']}: [{item['type']}] {item['question']}\n"
        
        batch_prompt += "\n请开始提取关键词："
        
        try:
            logger.info(f"批量提取 {len(questions_data)} 个题目的关键词...")
            
            # 使用总结模型进行批量提取
            response = self.keyword_extractor.generate_with_retry(
                prompt=batch_prompt,
                temperature=0.3,
                max_tokens=1500  # 增加token限制以支持20个题目的批量处理
            )
            
            logger.info(f"批量关键词提取完成，使用总结模型")
            
            # 解析批量响应
            results = {}
            lines = response.strip().split('\n')
            
            for line in lines:
                if ':' in line and any(item['id'] in line for item in questions_data):
                    try:
                        # 找到对应的ID
                        for item in questions_data:
                            if line.startswith(item['id'] + ':'):
                                # 提取关键词部分
                                keywords_text = line.split(':', 1)[1].strip()
                                keywords = [kw.strip().strip('"\'') for kw in keywords_text.split(',')]
                                
                                # 过滤和清理关键词
                                filtered_keywords = []
                                for kw in keywords:
                                    kw = kw.strip()
                                    if 2 <= len(kw) <= 30 and kw.lower() not in ['report', 'analysis', 'study', 'research']:
                                        filtered_keywords.append(kw)
                                
                                results[item['id']] = filtered_keywords[:8]
                                break
                    except Exception as e:
                        logger.warning(f"解析关键词行失败: {line}, 错误: {e}")
                        continue
            
            logger.info(f"批量提取成功，获得 {len(results)} 个题目的关键词")
            return results
            
        except Exception as e:
            logger.error(f"批量关键词提取失败: {e}")
            # 备用方案：使用默认关键词
            fallback_results = {}
            for item in questions_data:
                fallback_results[item['id']] = self.category_keywords.get(item['type'], [])[:5]
            return fallback_results

    def extract_keywords_from_question(self, question: str, category: str) -> List[str]:
        """使用LLM从赛题中提取关键词"""
        
        extraction_prompt = f"""
请从以下比赛题目中提取用于数据检索的关键词。

题目类别: {category}
题目内容: {question}

要求:
1. 提取3-8个最重要的关键词
2. 关键词应该是英文，便于国际数据源检索
3. 包含核心技术术语、概念名词
4. 避免过于通用的词汇（如"report", "analysis"等）
5. 优先选择可能在学术论文和新闻中出现的专业术语

请只返回关键词列表，用逗号分隔，不要其他内容。

示例格式: artificial intelligence, machine learning, explainable AI, XAI, interpretability
"""
        
        try:
            logger.info(f"正在为题目提取关键词: {question[:50]}...")
            
            # 使用总结模型进行关键词提取
            response = self.keyword_extractor.generate_with_retry(
                prompt=extraction_prompt,
                temperature=0.3,  # 较低温度确保一致性
                max_tokens=100
            )
            
            logger.info(f"关键词提取完成，使用总结模型")
            
            # 解析关键词
            keywords_text = response.strip()
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            
            # 过滤和清理关键词
            filtered_keywords = []
            for kw in keywords:
                # 移除引号和特殊字符
                kw = kw.strip('"\'').strip()
                # 跳过过短或过长的关键词
                if 2 <= len(kw) <= 30 and kw.lower() not in ['report', 'analysis', 'study', 'research']:
                    filtered_keywords.append(kw)
            
            logger.info(f"提取到关键词: {filtered_keywords}")
            return filtered_keywords[:8]  # 最多8个关键词
            
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            # 备用方案：使用默认类别关键词
            return self.category_keywords.get(category, [])[:5]
    
    def cache_data_for_competition_questions(self, competition_file: str = "data/preliminary.json", 
                                           limit_per_source: int = 10):
        """基于比赛题目的智能数据缓存"""
        logger.info("开始基于比赛题目的智能数据缓存...")
        
        # 加载比赛数据
        competition_path = Path(competition_file)
        if not competition_path.exists():
            logger.error(f"比赛数据文件不存在: {competition_file}")
            return
        
        with open(competition_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        logger.info(f"加载了 {len(questions)} 个比赛题目")
        
        # 按类别组织题目
        questions_by_category = {}
        for q in questions:
            category = q["type"]
            if category not in questions_by_category:
                questions_by_category[category] = []
            questions_by_category[category].append(q)
        
        # 为每个类别的题目提取关键词并缓存数据
        for category, category_questions in questions_by_category.items():
            logger.info(f"处理类别: {category} ({len(category_questions)} 个题目)")
            
            # 收集该类别所有题目的关键词
            all_category_keywords = set(self.category_keywords.get(category, []))
            
            # 准备批量处理的题目（每个类别处理前20个题目）
            batch_questions = category_questions[:20]
            
            # 批量提取关键词
            batch_keywords_results = self.extract_keywords_batch(batch_questions)
            
            # 合并所有关键词
            for question_data in batch_questions:
                question_id = question_data["id"]
                question_keywords = batch_keywords_results.get(question_id, [])
                all_category_keywords.update(question_keywords)
                
                logger.info(f"题目 {question_id} 关键词: {question_keywords}")
            
            # 类别间短暂休息
            time.sleep(1)
            
            # 使用增强的关键词列表缓存数据
            enhanced_keywords = list(all_category_keywords)
            logger.info(f"类别 {category} 增强关键词列表 ({len(enhanced_keywords)} 个): {enhanced_keywords[:10]}...")
            
            # 缓存该类别的数据
            self.cache_category_data_with_keywords(category, enhanced_keywords, limit_per_source)
            
            # 类别间休息
            time.sleep(2)
        
        logger.info("基于比赛题目的智能数据缓存完成")
    
    def cache_data_for_competition_questions_batched(self, competition_file: str = "data/preliminary.json", 
                                                    batch_size: int = 20, limit_per_question: int = 5):
        """基于比赛题目的分批智能数据缓存 - 累积保存模式"""
        logger.info("开始基于比赛题目的分批智能数据缓存...")
        logger.info(f"配置: 批次大小={batch_size}, 每题目数据量={limit_per_question}")
        
        # 加载比赛数据
        competition_path = Path(competition_file)
        if not competition_path.exists():
            logger.error(f"比赛数据文件不存在: {competition_file}")
            return
        
        with open(competition_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        logger.info(f"加载了 {len(questions)} 个比赛题目")
        
        # 按类别组织题目
        questions_by_category = {}
        for q in questions:
            category = q["type"]
            if category not in questions_by_category:
                questions_by_category[category] = []
            questions_by_category[category].append(q)
        
        total_stats = {
            "total_questions_processed": 0,
            "total_articles_cached": 0,
            "categories_processed": 0
        }
        
        # 处理每个类别
        for category, category_questions in questions_by_category.items():
            logger.info(f"处理类别: {category} ({len(category_questions)} 个题目)")
            category_stats = {
                "questions_processed": 0,
                "arxiv_articles": 0,
                "news_articles": 0,
                "newsapi_articles": 0,
                "batches_processed": 0
            }
            
            # 分批处理该类别的题目
            for i in range(0, len(category_questions), batch_size):
                batch_questions = category_questions[i:i+batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(category_questions) + batch_size - 1) // batch_size
                
                logger.info(f"处理批次 {batch_num}/{total_batches}: {len(batch_questions)} 个题目")
                
                # 批量提取关键词
                batch_keywords_results = self.extract_keywords_batch(batch_questions)
                
                # 为每个题目获取数据并累积保存
                for question_data in batch_questions:
                    question_id = question_data["id"]
                    question = question_data["question"]
                    question_keywords = batch_keywords_results.get(question_id, [])
                    
                    if question_keywords:
                        logger.info(f"为题目 {question_id} 获取数据，关键词: {question_keywords[:3]}...")
                        
                        # 为单个题目获取数据
                        question_stats = self._fetch_and_accumulate_data_for_question(
                            question_keywords, category, question_id, limit_per_question
                        )
                        
                        category_stats["arxiv_articles"] += question_stats["arxiv_count"]
                        category_stats["news_articles"] += question_stats["news_count"]
                        category_stats["newsapi_articles"] += question_stats["newsapi_count"]
                        category_stats["questions_processed"] += 1
                    
                    # 短暂休息避免API限制
                    time.sleep(0.5)
                
                category_stats["batches_processed"] += 1
                
                # 批次间休息
                time.sleep(2)
                logger.info(f"批次 {batch_num} 完成")
            
            # 类别统计
            logger.info(f"类别 {category} 完成:")
            logger.info(f"  处理题目数: {category_stats['questions_processed']}")
            logger.info(f"  获取论文数: {category_stats['arxiv_articles']}")
            logger.info(f"  获取新闻数: {category_stats['news_articles']}")
            logger.info(f"  获取NewsAPI数: {category_stats['newsapi_articles']}")
            logger.info(f"  处理批次数: {category_stats['batches_processed']}")
            
            total_stats["total_questions_processed"] += category_stats["questions_processed"]
            total_stats["total_articles_cached"] += category_stats["arxiv_articles"] + category_stats["news_articles"] + category_stats["newsapi_articles"]
            total_stats["categories_processed"] += 1
            
            # 类别间休息
            time.sleep(3)
        
        # 总体统计
        logger.info("分批智能数据缓存完成!")
        logger.info(f"总统计:")
        logger.info(f"  处理类别数: {total_stats['categories_processed']}")
        logger.info(f"  处理题目数: {total_stats['total_questions_processed']}")
        logger.info(f"  缓存文章数: {total_stats['total_articles_cached']}")
    
    def _fetch_and_accumulate_data_for_question(self, keywords: List[str], category: str, 
                                               question_id: str, limit_per_question: int) -> Dict[str, int]:
        """为单个题目获取数据并累积保存到缓存"""
        stats = {"arxiv_count": 0, "news_count": 0, "newsapi_count": 0}
        
        if not keywords:
            return stats
        
        # 选择关键词（使用更多关键词以获得更好的覆盖）
        selected_keywords = keywords[:min(8, len(keywords))]
        
        # 从ArXiv获取数据
        try:
            logger.debug(f"从ArXiv获取数据，关键词: {selected_keywords[:5]}")
            arxiv_data = self.arxiv_source.fetch(selected_keywords[:5], category, limit_per_question, save_individual_cache=False)
            if arxiv_data:
                # 累积保存到缓存
                self._append_to_category_cache(arxiv_data, category, "arxiv", question_id)
                stats["arxiv_count"] = len(arxiv_data)
                logger.debug(f"题目 {question_id} 获取到 {len(arxiv_data)} 篇ArXiv论文")
        except Exception as e:
            logger.error(f"ArXiv数据获取失败 (题目 {question_id}): {e}")
        
        # 从NewsAPI获取数据
        try:
            logger.debug(f"从NewsAPI获取数据，关键词: {selected_keywords[:3]}")
            newsapi_data = self.news_source.fetch(category, selected_keywords[:3], limit_per_question, save_individual_cache=False)
            if newsapi_data:
                # 累积保存到缓存
                self._append_to_category_cache(newsapi_data, category, "newsapi", question_id)
                stats["newsapi_count"] = len(newsapi_data)
                logger.debug(f"题目 {question_id} 获取到 {len(newsapi_data)} 篇NewsAPI新闻")
        except Exception as e:
            logger.error(f"NewsAPI数据获取失败 (题目 {question_id}): {e}")
        
        # 从RSS获取数据
        try:
            logger.debug(f"从RSS获取数据，关键词: {selected_keywords[:5]}")
            rss_data = self.rss_source.fetch(category, selected_keywords[:5], limit_per_question, save_individual_cache=False)
            if rss_data:
                # 累积保存到缓存
                self._append_to_category_cache(rss_data, category, "rss", question_id)
                stats["news_count"] = len(rss_data)
                logger.debug(f"题目 {question_id} 获取到 {len(rss_data)} 篇新闻文章")
        except Exception as e:
            logger.error(f"RSS数据获取失败 (题目 {question_id}): {e}")
        
        return stats
    
    def _append_to_category_cache(self, new_data: List[Dict], category: str, source_type: str, question_id: str):
        """将新数据追加到类别缓存文件中"""
        if not new_data:
            return
        
        # 生成缓存文件路径
        safe_category = self.cache_manager._safe_filename(category)
        cache_dir = self.cache_manager.cache_dir / safe_category
        cache_dir.mkdir(exist_ok=True)
        
        # 使用source_type_accumulated作为文件名
        cache_file = cache_dir / f"{source_type}_accumulated.json"
        
        # 读取现有数据
        existing_data = []
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.warning(f"读取现有缓存失败: {e}")
                existing_data = []
        
        # 创建现有数据的标题集合，用于去重
        existing_titles = set()
        for item in existing_data:
            if 'title' in item and item['title']:
                existing_titles.add(item['title'].strip().lower())
        
        # 为新数据添加题目ID标记，并进行去重
        new_unique_data = []
        for item in new_data:
            item["source_question_id"] = question_id
            item["cached_timestamp"] = time.time()
            
            # 检查是否重复（通过标题去重）
            item_title = item.get('title', '').strip().lower()
            if item_title and item_title not in existing_titles:
                new_unique_data.append(item)
                existing_titles.add(item_title)  # 更新已存在标题集合
            else:
                logger.debug(f"跳过重复内容: {item.get('title', 'Unknown')[:50]}...")
        
        # 合并数据（只添加去重后的新数据）
        existing_data.extend(new_unique_data)
        
        # 保存合并后的数据
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            duplicates_skipped = len(new_data) - len(new_unique_data)
            logger.debug(f"累积保存 {len(new_unique_data)} 条 {source_type} 数据到 {cache_file}" + 
                        (f"，跳过 {duplicates_skipped} 条重复数据" if duplicates_skipped > 0 else ""))
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def deduplicate_existing_cache(self, category: str = None):
        """去重现有缓存文件中的重复数据"""
        if category:
            categories = [category]
        else:
            # 获取所有类别
            categories = [
                "Cutting-Edge Tech & AI",
                "Business Models & Market Dynamics", 
                "Sustainability & Environmental Governance",
                "Social Change & Cultural Trends",
                "Life Sciences & Public Health",
                "Global Affairs & Future Governance"
            ]
        
        for cat in categories:
            safe_category = self.cache_manager._safe_filename(cat)
            cache_dir = self.cache_manager.cache_dir / safe_category
            
            if not cache_dir.exists():
                continue
                
            # 处理每种数据源类型
            for source_type in ["arxiv", "rss", "newsapi"]:
                cache_file = cache_dir / f"{source_type}_accumulated.json"
                
                if not cache_file.exists():
                    continue
                    
                try:
                    # 读取现有数据
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    original_count = len(data)
                    if original_count == 0:
                        continue
                    
                    # 去重（保留第一次出现的）
                    seen_titles = set()
                    unique_data = []
                    
                    for item in data:
                        title = item.get('title', '').strip().lower()
                        if title and title not in seen_titles:
                            unique_data.append(item)
                            seen_titles.add(title)
                        elif not title:
                            # 保留没有标题的项目（可能是错误数据但保险起见）
                            unique_data.append(item)
                    
                    duplicates_removed = original_count - len(unique_data)
                    
                    if duplicates_removed > 0:
                        # 保存去重后的数据
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(unique_data, f, ensure_ascii=False, indent=2)
                        
                        logger.info(f"类别 {cat} - {source_type}: 去重前 {original_count} 条，去重后 {len(unique_data)} 条，删除 {duplicates_removed} 条重复数据")
                    else:
                        logger.info(f"类别 {cat} - {source_type}: 无重复数据，共 {original_count} 条")
                        
                except Exception as e:
                    logger.error(f"去重缓存文件失败 {cache_file}: {e}")

    def cache_category_data_with_keywords(self, category: str, keywords: List[str], limit_per_source: int = 15):
        """使用指定关键词缓存类别数据"""
        all_data = []
        
        # 选择最重要的关键词
        selected_keywords = keywords[:8]  # 使用前8个关键词
        
        # 从ArXiv获取学术数据
        try:
            logger.info(f"从ArXiv获取 {category} 数据，关键词: {selected_keywords[:5]}")
            arxiv_data = self.arxiv_source.fetch(selected_keywords[:5], category, limit_per_source)
            all_data.extend(arxiv_data)
            logger.info(f"获取到 {len(arxiv_data)} 篇ArXiv论文")
        except Exception as e:
            logger.error(f"ArXiv数据获取失败: {e}")
        
        # 从RSS获取新闻数据
        try:
            logger.info(f"从RSS获取 {category} 数据，关键词: {selected_keywords[:5]}")
            rss_data = self.rss_source.fetch(category, selected_keywords[:5], limit_per_source)
            all_data.extend(rss_data)
            logger.info(f"获取到 {len(rss_data)} 篇新闻文章")
        except Exception as e:
            logger.error(f"RSS数据获取失败: {e}")
        
        logger.info(f"类别 {category} 总计获取 {len(all_data)} 条数据")
        return all_data

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
        
        logger.info(f"类别 {category} 共获取 {len(all_data)} 条原始数据（已保存到缓存）")
        return all_data
    
    def vectorize_cached_data(self):
        """将缓存的数据向量化并添加到RAG系统
        
        优先使用预处理后的缓存数据，如果不存在则使用原始缓存数据
        """
        logger.info("开始向量化缓存数据...")
        
        # 优先使用预处理后的缓存
        processed_cache_dir = Path("external_data_cache_processed")
        raw_cache_dir = Path("external_data_cache")
        
        # 确定使用哪个缓存目录
        if processed_cache_dir.exists() and any(processed_cache_dir.iterdir()):
            cache_dir = processed_cache_dir
            logger.info("使用预处理后的缓存数据进行向量化")
        elif raw_cache_dir.exists():
            cache_dir = raw_cache_dir
            logger.info("使用原始缓存数据进行向量化（建议先运行预处理）")
        else:
            logger.warning("未找到任何缓存数据")
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
                        
                        # 处理两种数据结构：list 或 dict
                        if isinstance(cache_data, list):
                            items = cache_data
                        else:
                            items = cache_data.get('data', [])
                        
                        for item in items:
                            # 智能选择内容：优先使用预处理后的数据
                            content = self._get_best_content_for_vectorization(item)
                            
                            # 格式化为文档块
                            doc_text = f"""标题: {item.get('title', '')}
来源: {item.get('source', '')} ({item.get('date', '')})
类型: {item.get('type', '')}
内容: {content}"""
                            
                            # 准备元数据
                            metadata = {
                                'category': category,
                                'title': item.get('title', ''),
                                'source': item.get('source', ''),
                                'date': item.get('date', ''),
                                'type': item.get('type', ''),
                                'source_file': str(cache_file.name)
                            }
                            
                            documents.append({'text': doc_text, 'metadata': metadata})
                    
                    except Exception as e:
                        logger.error(f"读取缓存文件失败 {cache_file}: {e}")
                
                # 添加到RAG系统
                if documents:
                    # 分块处理（每块最大1000字符）并保持元数据
                    chunked_docs_with_meta = self._chunk_documents_with_metadata(documents)
                    rag_system.add_documents_with_metadata(category, chunked_docs_with_meta)
                    logger.info(f"已向量化 {len(chunked_docs_with_meta)} 个文档块到类别 {category}")
        
        logger.info("数据向量化完成")
    
    def _get_best_content_for_vectorization(self, item: Dict[str, Any]) -> str:
        """智能选择最适合向量化的内容
        
        优先级：
        1. 预处理后的内容（已总结的高质量内容）
        2. 清理后的内容（去除HTML但未总结）
        3. 原始内容（兜底方案）
        """
        # 优先级1: 预处理后的内容（已总结）
        if item.get('processed', False) and 'content' in item:
            processed_content = item['content']
            # 验证预处理后的内容质量
            if processed_content and len(processed_content.strip()) > 50:
                logger.debug(f"使用预处理内容，长度: {len(processed_content)}")
                return processed_content
        
        # 优先级2: 清理后但未总结的内容
        if 'cleaned_content' in item:
            cleaned_content = item['cleaned_content']
            if cleaned_content and len(cleaned_content.strip()) > 50:
                logger.debug(f"使用清理后内容，长度: {len(cleaned_content)}")
                return cleaned_content
        
        # 优先级3: 原始内容（兜底）
        original_content = item.get('content', '')
        if original_content:
            logger.debug(f"使用原始内容，长度: {len(original_content)}")
            return original_content
        
        logger.warning("所有内容选项都为空")
        return ""
    
    def reprocess_cached_data(self):
        """重新预处理现有的缓存数据
        
        这个方法会：
        1. 扫描所有原始缓存文件
        2. 进行预处理
        3. 保存到独立的预处理文件夹
        4. 保持原始数据不变
        """
        logger.info("开始重新预处理现有缓存数据...")
        
        raw_cache_dir = Path("external_data_cache")
        processed_cache_dir = Path("external_data_cache_processed")
        
        if not raw_cache_dir.exists():
            logger.warning("原始缓存目录不存在")
            return
        
        # 创建预处理缓存目录
        processed_cache_dir.mkdir(exist_ok=True)
        logger.info(f"预处理数据将保存到: {processed_cache_dir}")
        
        total_files = 0
        total_items = 0
        processed_items = 0
        
        for category_dir in raw_cache_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name.replace('_', ' ').replace('-', ' ').title()
                logger.info(f"处理类别: {category_name}")
                
                # 创建对应的预处理类别目录
                processed_category_dir = processed_cache_dir / category_dir.name
                processed_category_dir.mkdir(exist_ok=True)
                
                for cache_file in category_dir.glob("*.json"):
                    total_files += 1
                    logger.info(f"  处理文件: {cache_file.name}")
                    
                    # 定义预处理后的文件路径
                    processed_file = processed_category_dir / cache_file.name
                    
                    # 检查预处理文件是否已存在且较新
                    if processed_file.exists():
                        raw_mtime = cache_file.stat().st_mtime
                        processed_mtime = processed_file.stat().st_mtime
                        if processed_mtime >= raw_mtime:
                            logger.info(f"    预处理文件已存在且是最新的，跳过")
                            continue
                    
                    try:
                        # 读取原始缓存文件
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            raw_cache_data = json.load(f)

                        # 兼容两种结构：list 或 dict
                        if isinstance(raw_cache_data, list):
                            original_data = raw_cache_data
                            cache_meta = {}
                        elif isinstance(raw_cache_data, dict):
                            original_data = raw_cache_data.get('data', [])
                            cache_meta = raw_cache_data.copy()
                        else:
                            logger.error(f"未知缓存文件结构: {cache_file}")
                            original_data = []
                            cache_meta = {}

                        total_items += len(original_data)
                        logger.info(f"    原始数据: {len(original_data)}条")

                        # 预处理所有数据
                        if original_data:
                            logger.info(f"    开始预处理 {len(original_data)} 条数据...")
                            processed_data = self.preprocessor.preprocess_batch(original_data)
                            processed_items += len(processed_data)

                            # 创建预处理后的缓存数据结构
                            processed_cache_data = cache_meta if isinstance(cache_meta, dict) else {}
                            processed_cache_data['data'] = processed_data
                            processed_cache_data['processing_info'] = {
                                'processed_at': time.time(),
                                'processed_count': len(processed_data),
                                'original_file': str(cache_file),
                                'preprocessor_config': self.preprocessor.get_stats()
                            }

                            # 保存预处理后的缓存文件
                            with open(processed_file, 'w', encoding='utf-8') as f:
                                json.dump(processed_cache_data, f, ensure_ascii=False, indent=2)

                            logger.info(f"    预处理完成，已保存到: {processed_file}")
                        else:
                            logger.info(f"    无数据需要处理")

                    except Exception as e:
                        logger.error(f"处理缓存文件失败 {cache_file}: {e}")
        
        logger.info(f"缓存数据重新预处理完成！")
        logger.info(f"  处理文件: {total_files}个")
        logger.info(f"  总数据项: {total_items}条")
        logger.info(f"  新处理项: {processed_items}条")
    
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
    
    def _chunk_documents_with_metadata(self, documents: List[Dict], max_chunk_size: int = 1000) -> List[Dict]:
        """将文档分块并保持元数据"""
        chunked_docs = []
        
        for doc_data in documents:
            doc_text = doc_data['text']
            metadata = doc_data['metadata']
            
            if len(doc_text) <= max_chunk_size:
                chunked_docs.append({'text': doc_text, 'metadata': metadata})
            else:
                # 按段落分割
                paragraphs = doc_text.split('\n')
                current_chunk = ""
                chunk_index = 0
                
                for para in paragraphs:
                    if len(current_chunk) + len(para) <= max_chunk_size:
                        current_chunk += para + "\n"
                    else:
                        if current_chunk.strip():
                            # 为每个分块创建元数据副本，并添加分块信息
                            chunk_metadata = metadata.copy()
                            chunk_metadata['chunk_index'] = chunk_index
                            chunked_docs.append({
                                'text': current_chunk.strip(), 
                                'metadata': chunk_metadata
                            })
                            chunk_index += 1
                        current_chunk = para + "\n"
                
                if current_chunk.strip():
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_index'] = chunk_index
                    chunked_docs.append({
                        'text': current_chunk.strip(), 
                        'metadata': chunk_metadata
                    })
        
        return chunked_docs

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="外部数据源管理")
    parser.add_argument("--action", choices=["cache", "process", "vectorize", "all", "stats", "reprocess", "smart-cache", "dedupe"], 
                       required=True, help="""
操作类型:
  cache: 获取原始数据并保存到external_data_cache（不进行预处理）
  process: 处理external_data_cache中的原始数据，保存到external_data_cache_processed
  vectorize: 将处理后的数据向量化并添加到RAG系统
  all: 完整流程：缓存数据 -> 预处理数据 -> 向量化
  smart-cache: 基于比赛题目的智能关键词缓存（推荐）
  stats: 显示缓存统计信息
  reprocess: 重新处理所有缓存数据
""")
    parser.add_argument("--category", help="指定类别")
    parser.add_argument("--limit", type=int, default=15, help="每个题目获取的数据条数")
    parser.add_argument("--batch", type=int, default=20, help="每批次处理的题目数量")
    
    args = parser.parse_args()
    
    manager = ExternalDataManager()
    
    if args.action == "cache":
        if args.category:
            manager.cache_category_data(args.category, args.limit)
        else:
            manager.cache_all_categories(args.limit)
    
    elif args.action == "smart-cache":
        # 智能缓存：基于比赛题目提取关键词
        print("=== 🧠 智能缓存模式：基于比赛题目提取关键词 ===")
        print(f"批次大小: {args.batch}, 每题目数据量: {args.limit}")
        manager.cache_data_for_competition_questions_batched(
            batch_size=args.batch, 
            limit_per_question=args.limit
        )
    
    elif args.action == "dedupe":
        # 去重现有缓存数据
        print("=== 🧹 缓存数据去重 ===")
        if args.category:
            print(f"去重类别: {args.category}")
            manager.deduplicate_existing_cache(args.category)
        else:
            print("去重所有类别")
            manager.deduplicate_existing_cache()
        print("去重完成!")
    
    elif args.action == "vectorize":
        manager.vectorize_cached_data()
    
    elif args.action == "all":
        # 完整流程：cache -> process -> vectorize
        print("=== 步骤 1/3: 缓存原始数据 ===")
        if args.category:
            manager.cache_category_data(args.category, args.limit)
        else:
            manager.cache_all_categories(args.limit)
        
        print("\n=== 步骤 2/3: 处理原始缓存数据 ===")
        time.sleep(1)
        manager.reprocess_cached_data()
        
        print("\n=== 步骤 3/3: 向量化处理后的数据 ===")
        time.sleep(1)
        manager.vectorize_cached_data()
        
        print("\n✅ 完整流程执行完毕！")
    
    elif args.action == "process":
        # 处理原始缓存数据（预处理并保存到 processed 文件夹）
        manager.reprocess_cached_data()
    
    elif args.action == "reprocess":
        # 重新预处理现有缓存数据
        manager.reprocess_cached_data()
    
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
