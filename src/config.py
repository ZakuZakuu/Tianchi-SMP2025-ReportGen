"""
配置管理模块
"""
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Anthropic配置
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    
    # SiliconFlow配置
    SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "sk-dtnqrtjfhadmpltdpclywhkxjcstclztuemifpeddhfeappf")
    SILICONFLOW_BASE_URL: str = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    
    # 阿里百炼配置
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_BASE_URL: str = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # 智谱AI配置
    ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "")
    ZHIPU_BASE_URL: str = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    
    # 模型配置 (格式: provider:model_name)
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "dashscope:qwen3-235b-a22b")
    SECONDARY_MODEL: str = os.getenv("SECONDARY_MODEL", "anthropic:claude-3-5-sonnet-20241022")
    BACKUP_MODEL: str = os.getenv("BACKUP_MODEL", "siliconflow:Qwen/Qwen2.5-72B-Instruct")
    
    # 服务商优先级
    PROVIDER_PRIORITY: str = os.getenv("PROVIDER_PRIORITY", "openai,siliconflow,dashscope,zhipu,anthropic")
    
    # 服务商配置映射
    @property
    def PROVIDER_CONFIGS(self) -> Dict[str, Dict[str, str]]:
        return {
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "base_url": self.OPENAI_BASE_URL,
                "type": "openai"
            },
            "anthropic": {
                "api_key": self.ANTHROPIC_API_KEY,
                "base_url": self.ANTHROPIC_BASE_URL,
                "type": "anthropic"
            },
            "siliconflow": {
                "api_key": self.SILICONFLOW_API_KEY,
                "base_url": self.SILICONFLOW_BASE_URL,
                "type": "openai"  # 兼容OpenAI格式
            },
            "dashscope": {
                "api_key": self.DASHSCOPE_API_KEY,
                "base_url": self.DASHSCOPE_BASE_URL,
                "type": "openai"  # 兼容OpenAI格式
            },
            "zhipu": {
                "api_key": self.ZHIPU_API_KEY,
                "base_url": self.ZHIPU_BASE_URL,
                "type": "openai"  # 兼容OpenAI格式
            }
        }
    
    @property
    def PROVIDER_PRIORITY_LIST(self) -> List[str]:
        return [p.strip() for p in self.PROVIDER_PRIORITY.split(",")]
    
    # RAG配置
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")  # local, siliconflow, openai
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))
    
    # 生成配置
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "180"))
    WORD_COUNT_TOLERANCE: float = float(os.getenv("WORD_COUNT_TOLERANCE", "0.03"))
    
    # 字数控制配置（简化版）
    WORD_COUNT_ACCEPTABLE_MIN_RATIO: float = float(os.getenv("WORD_COUNT_ACCEPTABLE_MIN_RATIO", "0.90"))
    WORD_COUNT_ACCEPTABLE_MAX_RATIO: float = float(os.getenv("WORD_COUNT_ACCEPTABLE_MAX_RATIO", "1.10"))
    WORD_COUNT_EXPANDABLE_MIN_RATIO: float = float(os.getenv("WORD_COUNT_EXPANDABLE_MIN_RATIO", "0.60"))
    WORD_COUNT_EXPANSION_DISCOUNT: float = float(os.getenv("WORD_COUNT_EXPANSION_DISCOUNT", "0.92"))
    WORD_COUNT_TRUNCATION_THRESHOLD: float = float(os.getenv("WORD_COUNT_TRUNCATION_THRESHOLD", "1.25"))
    WORD_COUNT_GENERATION_BOOST: float = float(os.getenv("WORD_COUNT_GENERATION_BOOST", "1.08"))
    
    # 主题类别映射
    CATEGORIES = {
        "Cutting-Edge Tech & AI": "tech_ai",
        "Business Models & Market Dynamics": "business_market", 
        "Sustainability & Environmental Governance": "sustainability_env",
        "Social Change & Cultural Trends": "social_cultural",
        "Life Sciences & Public Health": "life_health",
        "Global Affairs & Future Governance": "global_affairs"
    }

# 全局配置实例
config = Config()
