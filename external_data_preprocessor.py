"""
- 外部数据预处理模块
- HTML清理: 去除网页标签和格式
- 智能总结: 使用大模型压缩长文本
- 批量处理: 高效处理大量数据
- 缓存机制: 避免重复获取
"""

import re
import os
import time
from typing import Dict, Any, List, Optional
from loguru import logger
from dotenv import load_dotenv

# 导入模型相关类
from src.multi_provider_model import OpenAICompatibleModel
from src.config import config

# 加载环境变量
load_dotenv()

class ExternalDataPreprocessor:
    """外部数据预处理器"""
    
    def __init__(self):
        # 从环境变量和统一配置读取配置
        self.summary_model = os.getenv('SUMMARY_MODEL', 'siliconflow:Qwen/Qwen2.5-7B-Instruct')
        # 使用统一配置的文档长度限制，确保与向量化过程一致
        self.summary_max_length = min(int(os.getenv('SUMMARY_MAX_LENGTH', '1000')), config.MAX_DOCUMENT_LENGTH)
        self.summary_min_length = int(os.getenv('SUMMARY_MIN_LENGTH', '1500'))
        self.html_cleanup_enabled = os.getenv('HTML_CLEANUP_ENABLED', 'true').lower() == 'true'
        self.summary_enabled = bool(self.summary_model and self.summary_model.strip())
        
        # 延迟初始化ModelManager，使用简单的API调用
        self._summary_client = None
        
        # HTML清理正则表达式
        self.html_patterns = [
            r'<script[^>]*>.*?</script>',  # 移除script标签及内容
            r'<style[^>]*>.*?</style>',   # 移除style标签及内容
            r'<[^>]+>',                   # 移除所有HTML标签
            r'&nbsp;|&amp;|&lt;|&gt;|&quot;|&#\d+;',  # 移除HTML实体
            r'\s+',                       # 压缩多个空白字符
        ]
        
        # 总结提示词模板
        self.summary_prompt_template = """
Please summarize the following text content. Requirements:
1. Preserve core information and main points
2. Remove marketing language and redundant content
3. Keep within {max_length} characters
4. Maintain objective and accurate tone
5. Preserve specific data, numbers, and dates if present
6. Output in English only

Original content:
{content}

Summary:
"""
    
    def _get_summary_client(self):
        """延迟初始化总结客户端"""
        if self._summary_client is None:
            try:
                provider, model_name = self.summary_model.split(':', 1)
                
                # 使用OpenAI兼容的模型类，支持dashscope和siliconflow
                if provider in ['dashscope', 'siliconflow']:
                    self._summary_client = OpenAICompatibleModel(provider, model_name)
                else:
                    logger.error(f"不支持的总结模型提供商: {provider}")
                    return None
                    
                logger.info(f"已初始化总结模型: {self.summary_model}")
                
            except Exception as e:
                logger.error(f"初始化总结模型失败: {e}")
                return None
                
        return self._summary_client

    def clean_html(self, text: str) -> str:
        """清理HTML标签和格式"""
        if not self.html_cleanup_enabled or not text:
            return text
            
        try:
            cleaned_text = text
            
            # 应用所有清理规则
            for pattern in self.html_patterns:
                cleaned_text = re.sub(pattern, ' ', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
            
            # 清理多余的空白字符
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            cleaned_text = cleaned_text.strip()
            
            logger.debug(f"HTML清理完成，原长度: {len(text)}, 清理后长度: {len(cleaned_text)}")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"HTML清理失败: {e}")
            return text

    def should_summarize(self, content: str) -> bool:
        """判断是否需要进行总结"""
        return len(content) > self.summary_min_length

    def summarize_content(self, content: str, title: str = "") -> str:
        """使用大模型总结内容"""
        if not content:
            return content
            
        try:
            # 获取总结客户端
            client = self._get_summary_client()
            if client is None:
                logger.warning("总结客户端未初始化，使用原文")
                return content
            
            # 准备提示词
            prompt = self.summary_prompt_template.format(
                content=content,
                max_length=self.summary_max_length
            )
            
            # 如果有标题，添加到上下文中
            if title:
                prompt = f"文章标题：{title}\n\n" + prompt
            
            # 调用总结模型
            response = client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=self.summary_max_length + 100
            )
            
            # generate方法返回(response, metadata)，我们只需要response
            if isinstance(response, tuple):
                response_text = response[0]
            else:
                response_text = response
            
            if response_text and response_text.strip():
                logger.info(f"内容总结完成，原长度: {len(content)}, 总结后长度: {len(response_text)}")
                return response_text.strip()
            else:
                logger.warning("总结模型返回空内容，使用原文")
                return content
                
        except Exception as e:
            logger.error(f"内容总结失败: {e}")
            return content

    def preprocess_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """预处理单个数据项"""
        try:
            # 创建副本避免修改原数据
            processed_item = item.copy()
            
            title = item.get('title', '')
            content = item.get('content', '')
            
            # 第一步：HTML清理
            if content:
                cleaned_content = self.clean_html(content)
                processed_item['content'] = cleaned_content
                processed_item['original_content'] = content  # 保留原始内容
                
                # 第二步：智能总结（如果需要）
                if self.should_summarize(cleaned_content):
                    summarized_content = self.summarize_content(cleaned_content, title)
                    processed_item['content'] = summarized_content
                    processed_item['cleaned_content'] = cleaned_content  # 保留清理后的原文
                
            # 添加处理标记
            processed_item['processed'] = True
            processed_item['processed_at'] = time.time()
            
            logger.info(f"数据项预处理完成: {title[:50]}...")
            return processed_item
            
        except Exception as e:
            logger.error(f"数据项预处理失败: {e}")
            return item

    def preprocess_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量预处理数据"""
        logger.info(f"开始批量预处理 {len(items)} 个数据项")
        
        processed_items = []
        for i, item in enumerate(items):
            try:
                processed_item = self.preprocess_item(item)
                processed_items.append(processed_item)
                
                # 避免API调用过于频繁
                if i > 0 and i % 3 == 0:
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"处理第 {i+1} 个数据项时失败: {e}")
                processed_items.append(item)  # 使用原始数据
        
        logger.info(f"批量预处理完成，成功处理 {len(processed_items)} 个数据项")
        return processed_items

    def get_processing_stats(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取处理统计信息"""
        total_items = len(items)
        processed_items = sum(1 for item in items if item.get('processed', False))
        html_cleaned_items = sum(1 for item in items if 'original_content' in item)
        summarized_items = sum(1 for item in items if 'cleaned_content' in item)
        
        return {
            'total_items': total_items,
            'processed_items': processed_items,
            'html_cleaned_items': html_cleaned_items,
            'summarized_items': summarized_items,
            'processing_rate': processed_items / total_items if total_items > 0 else 0
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取预处理器状态统计"""
        return {
            'html_cleanup_enabled': self.html_cleanup_enabled,
            'summary_enabled': self.summary_enabled,
            'summary_model': self.summary_model,
            'summary_max_length': self.summary_max_length,
            'summary_min_length': self.summary_min_length,
            'client_initialized': self._summary_client is not None
        }

def test_preprocessor():
    """测试预处理器功能"""
    preprocessor = ExternalDataPreprocessor()
    
    # 测试数据
    test_item = {
        'title': 'Test Article',
        'content': '<p>This is a <b>test</b> article with <a href="#">HTML</a> content.</p>' * 20,
        'source': 'Test Source',
        'type': 'test_article'
    }
    
    print("原始内容长度:", len(test_item['content']))
    
    # 测试预处理
    processed_item = preprocessor.preprocess_item(test_item)
    
    print("处理后内容长度:", len(processed_item['content']))
    print("处理后内容:", processed_item['content'][:200] + "...")
    
    # 测试批量处理
    test_items = [test_item] * 3
    processed_items = preprocessor.preprocess_batch(test_items)
    
    # 显示统计信息
    stats = preprocessor.get_processing_stats(processed_items)
    print("处理统计:", stats)

if __name__ == "__main__":
    test_preprocessor()
