"""
模块导入统一管理
"""

# 使用标准库的logging替代loguru，避免依赖问题
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 创建logger实例供其他模块使用
logger = logging.getLogger(__name__)
