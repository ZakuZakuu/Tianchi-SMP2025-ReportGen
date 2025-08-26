"""
多服务商模型管理器
"""
import time
import json
import requests
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
import openai
import anthropic
import logging

from .config import config

logger = logging.getLogger(__name__)

class TokenUsage:
    """Token使用统计"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.requests = 0
    
    def add_usage(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += (input_tokens + output_tokens)
        self.requests += 1
    
    def get_summary(self) -> dict:
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'requests': self.requests
        }

class BaseModel(ABC):
    """大模型基类"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Tuple[str, dict]:
        """生成文本并返回token使用信息"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

class DashScopeNativeModel(BaseModel):
    """DashScope原生API模型封装（用于支持qwen3-235b-a22b等特殊模型）"""
    
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        
        provider_config = config.PROVIDER_CONFIGS.get(provider)
        if not provider_config:
            raise ValueError(f"未知的服务商: {provider}")
        
        if not provider_config["api_key"]:
            raise ValueError(f"{provider} API密钥未配置")
        
        self.api_key = provider_config["api_key"]
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        
        # 检查是否是需要特殊处理的模型
        self.is_thinking_model = self._is_thinking_model(model_name)
        
        logger.info(f"已初始化 {provider} 原生API模型: {model_name}")
        if self.is_thinking_model:
            logger.info(f"检测到思维链模型，支持 enable_thinking 参数")
    
    def _is_thinking_model(self, model_name: str) -> bool:
        """检查是否是需要enable_thinking参数的模型"""
        thinking_models = [
            "qwen3-235b-a22b",
            "qwen-max-0428",
            "qwen2.5-max"
        ]
        return any(thinking_model in model_name.lower() for thinking_model in thinking_models)
    
    def generate(self, prompt: str, **kwargs) -> Tuple[str, dict]:
        """使用DashScope原生API生成文本"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # 构建请求参数
            parameters = {
                'result_format': 'message',
                'temperature': kwargs.get("temperature", config.DEFAULT_TEMPERATURE),
                'max_tokens': kwargs.get("max_tokens", 4000)
            }
            
            # 如果是思维链模型，设置enable_thinking参数
            if self.is_thinking_model:
                # 对于非streaming调用，enable_thinking必须为False
                is_streaming = kwargs.get("stream", False)
                parameters['enable_thinking'] = is_streaming
                if is_streaming:
                    parameters['incremental_output'] = True
                    logger.info(f"启用流式思维链模式")
                else:
                    logger.info(f"非流式调用，禁用思维链: enable_thinking=False")
            
            payload = {
                'model': self.model_name,
                'input': {
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ]
                },
                'parameters': parameters
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=config.REQUEST_TIMEOUT  # 使用配置文件中的超时设置
            )
            
            if response.status_code != 200:
                raise Exception(f"API请求失败: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # 检查响应格式
            if 'output' not in result or 'choices' not in result['output']:
                raise Exception(f"响应格式异常: {result}")
            
            content = result['output']['choices'][0]['message']['content']
            
            # 提取token使用信息
            usage = result.get('usage', {})
            token_info = {
                'input_tokens': usage.get('input_tokens', 0),
                'output_tokens': usage.get('output_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
            
            # 如果是思维链模型，检查是否有推理过程
            if self.is_thinking_model:
                reasoning_content = result['output']['choices'][0]['message'].get('reasoning_content', '')
                if reasoning_content:
                    logger.info(f"模型推理过程长度: {len(reasoning_content)} 字符")
            
            return content, token_info
            
        except Exception as e:
            logger.error(f"DashScope原生API生成失败: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        # 简单估算
        return len(text.split()) * 1.3


class OpenAICompatibleModel(BaseModel):
    """OpenAI兼容格式的模型封装（支持SiliconFlow、阿里百炼等）"""
    
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        
        provider_config = config.PROVIDER_CONFIGS.get(provider)
        if not provider_config:
            raise ValueError(f"未知的服务商: {provider}")
        
        if not provider_config["api_key"]:
            raise ValueError(f"{provider} API密钥未配置")
        
        self.client = openai.OpenAI(
            api_key=provider_config["api_key"],
            base_url=provider_config["base_url"],
            timeout=config.REQUEST_TIMEOUT  # 使用配置文件中的超时设置
        )
        
        # 检查是否是需要特殊处理的模型
        self.is_thinking_model = self._is_thinking_model(model_name)
        
        logger.info(f"已初始化 {provider} 模型: {model_name}")
        if self.is_thinking_model:
            logger.info(f"检测到思维链模型，将启用 enable_thinking 参数")
    
    def _is_thinking_model(self, model_name: str) -> bool:
        """检查是否是需要enable_thinking参数的模型"""
        thinking_models = [
            "qwen3-235b-a22b",
            "qwen-max-0428",  # 其他可能需要的模型
            "qwen2.5-max"     # 未来可能的模型
        ]
        return any(thinking_model in model_name.lower() for thinking_model in thinking_models)
        
    def generate(self, prompt: str, **kwargs) -> Tuple[str, dict]:
        """生成文本并返回token使用信息"""
        try:
            # 构建基础参数
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", config.DEFAULT_TEMPERATURE),
                "max_tokens": kwargs.get("max_tokens", 4000)
            }
            
            # 如果是思维链模型，根据是否streaming来设置enable_thinking
            if self.is_thinking_model:
                is_streaming = kwargs.get("stream", False)
                if is_streaming:
                    # 流式调用可以启用enable_thinking
                    params["enable_thinking"] = kwargs.get("enable_thinking", True)
                    params["stream"] = True
                    logger.info(f"启用流式思维链模式: enable_thinking={params['enable_thinking']}")
                else:
                    # 非流式调用必须禁用enable_thinking
                    params["enable_thinking"] = False
                    logger.info(f"非流式调用，禁用思维链: enable_thinking=False")
            
            response = self.client.chat.completions.create(**params)
            
            # 处理流式响应
            if kwargs.get("stream", False):
                content_parts = []
                input_tokens = 0
                output_tokens = 0
                
                for chunk in response:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            content_parts.append(delta.content)
                    
                    # 处理token使用信息（通常在最后一个chunk中）
                    if hasattr(chunk, 'usage') and chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens
                
                content = ''.join(content_parts)
                token_info = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens
                }
            else:
                # 处理非流式响应
                usage = response.usage
                token_info = {
                    'input_tokens': usage.prompt_tokens if usage else 0,
                    'output_tokens': usage.completion_tokens if usage else 0,
                    'total_tokens': usage.total_tokens if usage else 0
                }
                content = response.choices[0].message.content
            
            return content, token_info
            
        except Exception as e:
            logger.error(f"{self.provider} 生成失败: {e}")
            
            # 如果是思维链模型的参数错误，尝试简化参数重试
            if self.is_thinking_model and ("enable_thinking" in str(e) or "parameter" in str(e).lower()):
                logger.warning("思维链参数可能有问题，尝试使用最简参数重试")
                try:
                    simple_params = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": kwargs.get("temperature", config.DEFAULT_TEMPERATURE),
                        "max_tokens": kwargs.get("max_tokens", 4000)
                    }
                    response = self.client.chat.completions.create(**simple_params)
                    
                    usage = response.usage
                    token_info = {
                        'input_tokens': usage.prompt_tokens if usage else 0,
                        'output_tokens': usage.completion_tokens if usage else 0,
                        'total_tokens': usage.total_tokens if usage else 0
                    }
                    
                    return response.choices[0].message.content, token_info
                except Exception as retry_e:
                    logger.error(f"简化参数重试也失败: {retry_e}")
                    raise retry_e
            raise
    
    def count_tokens(self, text: str) -> int:
        # 简单估算
        return len(text.split()) * 1.3

class AnthropicModel(BaseModel):
    """Anthropic Claude模型封装"""
    
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        
        provider_config = config.PROVIDER_CONFIGS.get(provider)
        if not provider_config or not provider_config["api_key"]:
            raise ValueError(f"{provider} API密钥未配置")
        
        # Anthropic可能不支持自定义base_url，使用官方SDK
        self.client = anthropic.Anthropic(
            api_key=provider_config["api_key"],
            timeout=config.REQUEST_TIMEOUT  # 使用配置文件中的超时设置
        )
        
        logger.info(f"已初始化 {provider} 模型: {model_name}")
        
    def generate(self, prompt: str, **kwargs) -> Tuple[str, dict]:
        """生成文本并返回token使用信息"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 4000),
                temperature=kwargs.get("temperature", config.DEFAULT_TEMPERATURE),
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Anthropic也提供token使用信息
            usage = response.usage
            token_info = {
                'input_tokens': usage.input_tokens if usage else 0,
                'output_tokens': usage.output_tokens if usage else 0,
                'total_tokens': (usage.input_tokens + usage.output_tokens) if usage else 0
            }
            
            return response.content[0].text, token_info
        except Exception as e:
            logger.error(f"{self.provider} 生成失败: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        return len(text.split()) * 1.2

class MultiProviderModelManager:
    """多服务商模型管理器"""
    
    def __init__(self):
        self.models = {}
        self.available_providers = []
        self.token_usage = TokenUsage()  # 添加token统计
        self._init_models()
        
    def _parse_model_config(self, model_config: str) -> Tuple[str, str]:
        """解析模型配置 格式: provider:model_name"""
        if ":" not in model_config:
            # 兼容旧格式，默认为openai
            return "openai", model_config
        
        provider, model_name = model_config.split(":", 1)
        return provider.strip(), model_name.strip()
    
    def _create_model(self, provider: str, model_name: str) -> Optional[BaseModel]:
        """创建模型实例"""
        try:
            provider_config = config.PROVIDER_CONFIGS.get(provider)
            if not provider_config:
                logger.warning(f"未知的服务商: {provider}")
                return None
            
            if not provider_config["api_key"]:
                logger.warning(f"{provider} API密钥未配置，跳过")
                return None
            
            # 特殊模型使用原生API
            if provider == "dashscope" and "qwen3-235b-a22b" in model_name:
                return DashScopeNativeModel(provider, model_name)
            
            # 根据类型创建不同的模型实例
            if provider_config["type"] == "anthropic":
                return AnthropicModel(provider, model_name)
            else:  # openai兼容格式
                return OpenAICompatibleModel(provider, model_name)
                
        except Exception as e:
            logger.error(f"创建模型失败 {provider}:{model_name} - {e}")
            return None
    
    def _init_models(self):
        """初始化模型"""
        model_configs = {
            "primary": config.PRIMARY_MODEL,
            "secondary": config.SECONDARY_MODEL,
            "backup": config.BACKUP_MODEL
        }
        
        for role, model_config in model_configs.items():
            provider, model_name = self._parse_model_config(model_config)
            model = self._create_model(provider, model_name)
            
            if model:
                self.models[role] = {
                    "model": model,
                    "provider": provider,
                    "model_name": model_name
                }
                
                if provider not in self.available_providers:
                    self.available_providers.append(provider)
                
                logger.info(f"已初始化{role}模型: {provider}:{model_name}")
            else:
                logger.warning(f"跳过{role}模型: {model_config}")
        
        # 如果没有可用模型，尝试按优先级自动初始化
        if not self.models:
            self._auto_init_fallback_models()
    
    def _auto_init_fallback_models(self):
        """自动初始化备用模型"""
        logger.info("没有可用模型，尝试自动初始化...")
        
        # 常见的模型映射
        fallback_models = {
            "openai": ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            "siliconflow": [
                "Qwen/Qwen2.5-72B-Instruct", 
                "deepseek-ai/DeepSeek-V2.5",
                "meta-llama/Meta-Llama-3.1-8B-Instruct"
            ],
            "dashscope": ["qwen-max", "qwen-plus", "qwen-turbo"],
            "zhipu": ["glm-4", "glm-4-plus", "glm-3-turbo"],
            "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
        }
        
        for provider in config.PROVIDER_PRIORITY_LIST:
            if provider in fallback_models:
                for model_name in fallback_models[provider]:
                    model = self._create_model(provider, model_name)
                    if model:
                        if "primary" not in self.models:
                            self.models["primary"] = {
                                "model": model,
                                "provider": provider,
                                "model_name": model_name
                            }
                        elif "backup" not in self.models:
                            self.models["backup"] = {
                                "model": model,
                                "provider": provider,
                                "model_name": model_name
                            }
                        
                        if provider not in self.available_providers:
                            self.available_providers.append(provider)
                        
                        logger.info(f"自动初始化模型: {provider}:{model_name}")
                        break
            
            # 如果已经有主要模型和备用模型，停止初始化
            if "primary" in self.models and "backup" in self.models:
                break
    
    def select_model(self, category: str, word_limit: int) -> str:
        """根据类别和字数要求选择最优模型"""
        # 科学类话题优先使用Claude
        science_categories = [
            "Life Sciences & Public Health",
            "Sustainability & Environmental Governance"
        ]
        
        if category in science_categories and "secondary" in self.models:
            secondary_provider = self.models["secondary"]["provider"]
            if secondary_provider == "anthropic":
                return "secondary"
        
        # 默认使用主力模型
        if "primary" in self.models:
            return "primary"
        else:
            return "backup"
    
    def generate_with_retry(self, prompt: str, model_key: str = "primary", **kwargs) -> str:
        """带重试机制的生成，并记录token使用"""
        # 定义重试顺序
        retry_order = [model_key]
        for key in ["primary", "secondary", "backup"]:
            if key != model_key and key in self.models:
                retry_order.append(key)
        
        for attempt in range(config.MAX_RETRIES):
            for current_key in retry_order:
                if current_key not in self.models:
                    continue
                
                try:
                    model_info = self.models[current_key]
                    model = model_info["model"]
                    
                    logger.info(f"尝试使用 {model_info['provider']}:{model_info['model_name']}")
                    
                    # 调用返回tuple的generate方法
                    result, token_info = model.generate(prompt, **kwargs)
                    
                    # 记录token使用
                    self.token_usage.add_usage(
                        token_info['input_tokens'], 
                        token_info['output_tokens']
                    )
                    
                    logger.info(f"Token使用: 输入{token_info['input_tokens']}, 输出{token_info['output_tokens']}")
                    
                    if result and len(result.strip()) > 50:
                        logger.info(f"生成成功，使用模型: {model_info['provider']}:{model_info['model_name']}")
                        return result
                    else:
                        raise ValueError("生成内容质量不足")
                        
                except Exception as e:
                    logger.warning(f"模型 {current_key} 生成失败: {e}")
                    
                    if attempt < config.MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)  # 指数退避
                    continue
        
        raise Exception("所有模型重试均失败")
    
    def dual_model_validation(self, prompt: str, category: str) -> str:
        """双模型验证生成"""
        results = []
        
        # 获取两个不同的模型
        model_keys = []
        if "primary" in self.models:
            model_keys.append("primary")
        if "secondary" in self.models:
            model_keys.append("secondary")
        elif "backup" in self.models:
            model_keys.append("backup")
        
        if len(model_keys) < 2:
            # 如果只有一个模型，直接使用单模型生成
            return self.generate_with_retry(prompt, model_keys[0] if model_keys else "primary")
        
        # 尝试用两个不同模型生成
        for model_key in model_keys[:2]:
            try:
                result = self.generate_with_retry(prompt, model_key)
                model_info = self.models[model_key]
                results.append({
                    "model_key": model_key,
                    "provider": model_info["provider"],
                    "model_name": model_info["model_name"],
                    "content": result,
                    "word_count": len(result.split())
                })
            except Exception as e:
                logger.error(f"双模型验证失败 {model_key}: {e}")
        
        if not results:
            raise Exception("双模型验证完全失败")
        
        # 简单选择第一个结果（后续可以加入质量评估）
        selected = results[0]
        logger.info(f"双模型验证完成，选择: {selected['provider']}:{selected['model_name']}")
        return selected["content"]
    
    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        status = {
            "available_models": {},
            "available_providers": self.available_providers,
            "total_models": len(self.models)
        }
        
        for role, model_info in self.models.items():
            status["available_models"][role] = {
                "provider": model_info["provider"],
                "model_name": model_info["model_name"]
            }
        
        return status
    
    def switch_model(self, role: str, provider: str, model_name: str) -> bool:
        """动态切换模型"""
        try:
            new_model = self._create_model(provider, model_name)
            if new_model:
                self.models[role] = {
                    "model": new_model,
                    "provider": provider,
                    "model_name": model_name
                }
                logger.info(f"已切换{role}模型到: {provider}:{model_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"切换模型失败: {e}")
            return False
    
    def get_token_usage(self) -> dict:
        """获取token使用统计"""
        return self.token_usage.get_summary()
    
    def reset_token_usage(self):
        """重置token使用统计"""
        self.token_usage.reset()

# 全局模型管理器实例
model_manager = MultiProviderModelManager()
