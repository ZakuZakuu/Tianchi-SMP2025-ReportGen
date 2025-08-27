"""
自适应字数增益模块
动态调整生成字数的增益系数，实现精确的字数控制
"""
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger
from datetime import datetime


class AdaptiveWordGainManager:
    """自适应字数增益管理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.history_file = self.data_dir / "adaptive_word_gain_history.json"
        
        # 配置参数
        self.alpha = float(os.getenv('ADAPTIVE_GAIN_ALPHA', '0.3'))  # 学习率
        self.min_gain = float(os.getenv('ADAPTIVE_GAIN_MIN', '1.0'))  # 最小增益系数
        self.max_gain = float(os.getenv('ADAPTIVE_GAIN_MAX', '1.35'))  # 最大增益系数
        self.default_gain = float(os.getenv('ADAPTIVE_GAIN_DEFAULT', '1.12'))  # 默认增益系数
        self.enabled = os.getenv('ADAPTIVE_GAIN_ENABLED', 'true').lower() == 'true'
        
        # 新增配置参数
        self.satisfaction_min_ratio = float(os.getenv('ADAPTIVE_GAIN_SATISFACTION_MIN', '0.95'))  # 满意区间下限
        self.satisfaction_max_ratio = float(os.getenv('ADAPTIVE_GAIN_SATISFACTION_MAX', '1.05'))  # 满意区间上限
        self.max_adjustment_ratio = float(os.getenv('ADAPTIVE_GAIN_MAX_ADJUSTMENT', '0.1'))  # 单次最大调整比例
        
        # 分类别的增益历史
        self.category_gains = self._load_history()
        
        logger.info(f"自适应字数增益管理器初始化完成 - 启用状态: {self.enabled}")
        logger.info(f"参数配置 - α: {self.alpha}, 范围: [{self.min_gain}, {self.max_gain}], 默认: {self.default_gain}")
        logger.info(f"满意区间: [{self.satisfaction_min_ratio:.2f}, {self.satisfaction_max_ratio:.2f}], 最大调整: {self.max_adjustment_ratio:.1%}")
    
    def _load_history(self) -> Dict[str, Dict[str, Any]]:
        """加载历史增益数据"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"已加载 {len(data)} 个类别的增益历史")
                return data
        except Exception as e:
            logger.warning(f"加载增益历史失败: {e}")
        
        # 返回默认结构
        return {}
    
    def _save_history(self):
        """保存增益历史到文件"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.category_gains, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存增益历史失败: {e}")
    
    def _get_category_key(self, category: str) -> str:
        """规范化类别键名"""
        return category.lower().replace(' ', '_').replace('&', 'and')
    
    def _calculate_ratio(self, target_words: int, actual_words: int) -> float:
        """
        计算字数比率 (目标字数/实际字数)
        
        Args:
            target_words: 目标字数
            actual_words: 实际生成字数
            
        Returns:
            字数比率，带边界保护
        """
        if actual_words <= 0:
            logger.warning(f"实际字数无效: {actual_words}，使用默认比率")
            return 1.0
        
        ratio = target_words / actual_words
        
        # 边界保护：防止极端比率
        min_ratio = 0.7  # 最小比率
        max_ratio = 1.6  # 最大比率
        
        if ratio < min_ratio:
            logger.warning(f"字数比率过小 {ratio:.3f}，调整为 {min_ratio}")
            ratio = min_ratio
        elif ratio > max_ratio:
            logger.warning(f"字数比率过大 {ratio:.3f}，调整为 {max_ratio}")
            ratio = max_ratio
        
        return ratio
    
    def _apply_proportional_adjustment(self, target_words: int, actual_words: int, previous_gain: float) -> float:
        """
        应用比例调整算法计算新的增益系数
        
        公式: new_gain = previous_gain * (1 + α * ((target_words/actual_words) - 1))
        
        特性:
        - 满意区间内不调整
        - 单次调整幅度有上限
        - 比例调整更平滑
        
        Args:
            target_words: 目标字数
            actual_words: 实际生成字数
            previous_gain: 上一次的增益系数
            
        Returns:
            新的增益系数
        """
        if actual_words <= 0:
            logger.warning(f"实际字数无效: {actual_words}，返回原增益系数")
            return previous_gain
        
        # 计算准确率
        accuracy_ratio = actual_words / target_words
        
        # 检查是否在满意区间内
        if self.satisfaction_min_ratio <= accuracy_ratio <= self.satisfaction_max_ratio:
            logger.debug(f"字数准确率 {accuracy_ratio:.3f} 在满意区间内，不调整增益系数")
            return previous_gain
        
        # 计算调整因子
        adjustment_factor = (target_words / actual_words) - 1
        
        # 限制调整幅度
        max_adjustment = self.max_adjustment_ratio
        if abs(adjustment_factor * self.alpha) > max_adjustment:
            adjustment_factor = max_adjustment / self.alpha * (1 if adjustment_factor > 0 else -1)
            logger.debug(f"调整幅度超限，限制为: {adjustment_factor:.4f}")
        
        # 应用比例调整
        new_gain = previous_gain * (1 + self.alpha * adjustment_factor)
        
        # 边界保护
        new_gain = max(self.min_gain, min(new_gain, self.max_gain))
        
        return new_gain
    
    def get_gain_coefficient(self, category: str, question_id: str = None) -> float:
        """
        获取指定类别的当前增益系数
        
        Args:
            category: 题目类别
            question_id: 题目ID (可选，用于日志)
            
        Returns:
            增益系数
        """
        if not self.enabled:
            return self.default_gain
        
        category_key = self._get_category_key(category)
        
        if category_key not in self.category_gains:
            # 初始化新类别
            self.category_gains[category_key] = {
                'current_gain': self.default_gain,
                'total_samples': 0,
                'last_updated': datetime.now().isoformat(),
                'category_name': category
            }
            logger.info(f"初始化类别 '{category}' 的增益系数: {self.default_gain}")
        
        current_gain = self.category_gains[category_key]['current_gain']
        
        if question_id:
            logger.debug(f"题目 {question_id} (类别: {category}) 获取增益系数: {current_gain:.4f}")
        else:
            logger.debug(f"类别 '{category}' 当前增益系数: {current_gain:.4f}")
        
        return current_gain
    
    def update_gain_coefficient(self, category: str, target_words: int, actual_words: int, 
                              question_id: str = None) -> float:
        """
        基于生成结果更新增益系数
        
        Args:
            category: 题目类别
            target_words: 目标字数
            actual_words: 实际生成字数
            question_id: 题目ID (可选，用于日志)
            
        Returns:
            更新后的增益系数
        """
        if not self.enabled:
            return self.default_gain
        
        try:
            category_key = self._get_category_key(category)
            
            # 确保类别存在
            if category_key not in self.category_gains:
                self.category_gains[category_key] = {
                    'current_gain': self.default_gain,
                    'total_samples': 0,
                    'last_updated': datetime.now().isoformat(),
                    'category_name': category
                }
            
            # 获取上一次的增益系数
            previous_gain = self.category_gains[category_key]['current_gain']
            
            # 计算新的增益系数
            new_gain = self._apply_proportional_adjustment(target_words, actual_words, previous_gain)
            
            # 计算准确率和比率（用于日志）
            accuracy_ratio = actual_words / target_words if target_words > 0 else 0
            
            # 更新记录
            self.category_gains[category_key].update({
                'current_gain': new_gain,
                'total_samples': self.category_gains[category_key]['total_samples'] + 1,
                'last_updated': datetime.now().isoformat(),
                'last_accuracy_ratio': accuracy_ratio,
                'last_target_words': target_words,
                'last_actual_words': actual_words
            })
            
            # 保存到文件
            self._save_history()
            
            # 记录日志
            accuracy = (actual_words / target_words * 100) if target_words > 0 else 0
            log_msg = (f"更新增益系数 - 类别: {category}")
            if question_id:
                log_msg += f", 题目: {question_id}"
            log_msg += f", 字数: {actual_words}/{target_words} ({accuracy:.1f}%)"
            log_msg += f", 准确率: {accuracy_ratio:.4f}, 增益: {previous_gain:.4f} → {new_gain:.4f}"
            
            logger.info(log_msg)
            
            return new_gain
            
        except Exception as e:
            logger.error(f"更新增益系数失败: {e}")
            return self.default_gain
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取增益系数统计信息"""
        stats = {
            'enabled': self.enabled,
            'total_categories': len(self.category_gains),
            'parameters': {
                'alpha': self.alpha,
                'min_gain': self.min_gain,
                'max_gain': self.max_gain,
                'default_gain': self.default_gain
            },
            'categories': {}
        }
        
        for category_key, data in self.category_gains.items():
            stats['categories'][data.get('category_name', category_key)] = {
                'current_gain': data['current_gain'],
                'total_samples': data['total_samples'],
                'last_updated': data['last_updated'],
                'last_accuracy': (data.get('last_accuracy_ratio', 0) * 100) if data.get('last_accuracy_ratio') else 0
            }
        
        return stats
    
    def reset_category(self, category: str):
        """重置指定类别的增益系数为默认值"""
        category_key = self._get_category_key(category)
        if category_key in self.category_gains:
            self.category_gains[category_key]['current_gain'] = self.default_gain
            self.category_gains[category_key]['total_samples'] = 0
            self.category_gains[category_key]['last_updated'] = datetime.now().isoformat()
            self._save_history()
            logger.info(f"已重置类别 '{category}' 的增益系数为默认值: {self.default_gain}")
    
    def reset_all(self):
        """重置所有类别的增益系数"""
        for category_key in self.category_gains:
            self.category_gains[category_key]['current_gain'] = self.default_gain
            self.category_gains[category_key]['total_samples'] = 0
            self.category_gains[category_key]['last_updated'] = datetime.now().isoformat()
        self._save_history()
        logger.info(f"已重置所有类别的增益系数为默认值: {self.default_gain}")


# 全局实例
adaptive_word_gain_manager = AdaptiveWordGainManager()
