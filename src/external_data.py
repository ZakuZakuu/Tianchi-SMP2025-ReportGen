"""
外部数据获取模块 - 为后续扩展预留接口
"""
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from loguru import logger

class BaseDataSource(ABC):
    """数据源基类"""
    
    @abstractmethod
    def fetch(self, query: str, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """获取数据"""
        pass

class MockDataSource(BaseDataSource):
    """模拟数据源 - 用于测试和开发阶段"""
    
    def __init__(self):
        self.mock_data = {
            "Cutting-Edge Tech & AI": [
                {
                    "title": "2025年AI技术发展报告",
                    "content": "人工智能技术在2025年继续快速发展，大模型技术日趋成熟...",
                    "source": "Tech Research Institute",
                    "date": "2025-01-15"
                },
                {
                    "title": "可解释AI的最新进展",
                    "content": "可解释人工智能(XAI)在医疗、金融等关键领域的应用越来越重要...",
                    "source": "AI Journal",
                    "date": "2025-02-10"
                }
            ],
            "Business Models & Market Dynamics": [
                {
                    "title": "2025年全球市场动态分析",
                    "content": "全球经济在后疫情时代呈现新的发展特征，数字化转型加速...",
                    "source": "Economic Times",
                    "date": "2025-01-20"
                }
            ],
            # 其他类别的模拟数据...
        }
    
    def fetch(self, query: str, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """获取模拟数据"""
        logger.info(f"模拟数据源查询: {query}, 类别: {category}")
        
        if category and category in self.mock_data:
            return self.mock_data[category][:limit]
        
        # 如果没有指定类别，返回所有相关数据
        all_data = []
        for cat_data in self.mock_data.values():
            all_data.extend(cat_data)
        
        return all_data[:limit]

# TODO: 后续实现的真实数据源
class NewsAPISource(BaseDataSource):
    """新闻API数据源 - 待实现"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
    
    def fetch(self, query: str, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        # TODO: 实现新闻API调用
        logger.info("新闻API数据源 - 待实现")
        return []

class ArxivSource(BaseDataSource):
    """ArXiv学术论文数据源 - 待实现"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
    
    def fetch(self, query: str, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        # TODO: 实现ArXiv API调用
        logger.info("ArXiv数据源 - 待实现")
        return []

class FinancialDataSource(BaseDataSource):
    """金融数据源 - 待实现"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def fetch(self, query: str, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        # TODO: 实现金融数据API调用
        logger.info("金融数据源 - 待实现")
        return []

class ExternalDataManager:
    """外部数据管理器"""
    
    def __init__(self):
        self.data_sources = {
            "mock": MockDataSource(),
            # TODO: 后续添加真实数据源
            # "news": NewsAPISource(api_key),
            # "arxiv": ArxivSource(),
            # "financial": FinancialDataSource(api_key)
        }
        
        # 根据类别映射到合适的数据源
        self.category_mapping = {
            "Cutting-Edge Tech & AI": ["mock", "arxiv"],
            "Business Models & Market Dynamics": ["mock", "news", "financial"],
            "Sustainability & Environmental Governance": ["mock", "news"],
            "Social Change & Cultural Trends": ["mock", "news"],
            "Life Sciences & Public Health": ["mock", "arxiv"],
            "Global Affairs & Future Governance": ["mock", "news"]
        }
    
    def fetch_data(self, query: str, category: str, limit: int = 15) -> List[Dict[str, Any]]:
        """根据查询和类别获取外部数据"""
        logger.info(f"获取外部数据: {query}, 类别: {category}")
        
        # 获取该类别对应的数据源
        source_names = self.category_mapping.get(category, ["mock"])
        
        all_data = []
        for source_name in source_names:
            if source_name in self.data_sources:
                try:
                    source = self.data_sources[source_name]
                    data = source.fetch(query, category, limit // len(source_names))
                    all_data.extend(data)
                except Exception as e:
                    logger.error(f"数据源 {source_name} 获取失败: {e}")
        
        logger.info(f"共获取到 {len(all_data)} 条外部数据")
        return all_data[:limit]
    
    def add_data_source(self, name: str, source: BaseDataSource):
        """添加新的数据源"""
        self.data_sources[name] = source
        logger.info(f"已添加数据源: {name}")
    
    def format_external_data(self, data_list: List[Dict[str, Any]]) -> str:
        """格式化外部数据为文本"""
        if not data_list:
            return "暂无相关外部数据。"
        
        formatted_parts = []
        for i, item in enumerate(data_list[:10], 1):  # 最多使用10条数据
            title = item.get("title", "无标题")
            content = item.get("content", "")[:300]  # 限制内容长度
            source = item.get("source", "未知来源")
            date = item.get("date", "")
            
            formatted_parts.append(f"""
{i}. 【{title}】
来源：{source} {date}
内容：{content}...
""")
        
        return "\n".join(formatted_parts)

# 全局外部数据管理器实例
external_data_manager = ExternalDataManager()
