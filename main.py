"""
主入口文件
"""
import sys
import json
from pathlib import Path
import argparse
import logging

# 配置简单的日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入核心模块
from src.config import config
from src.rag_system import rag_system
from src.multi_provider_model import model_manager
from src.report_generator import report_generator

def init_system():
    """初始化系统"""
    logger.info("正在初始化系统...")
    
    # 检查API密钥
    available_providers = []
    for provider, provider_config in config.PROVIDER_CONFIGS.items():
        if provider_config["api_key"]:
            available_providers.append(provider)
    
    if not available_providers:
        logger.error("请在.env文件中配置至少一个服务商的API密钥")
        logger.info("支持的服务商: openai, anthropic, siliconflow, dashscope, zhipu")
        return False
    
    logger.info(f"检测到可用的服务商: {', '.join(available_providers)}")
    
    # 初始化知识库（加载示例数据）
    try:
        rag_system.load_sample_data()
        logger.info("知识库初始化完成")
    except Exception as e:
        logger.error(f"知识库初始化失败: {e}")
        return False
    
    # 检查模型状态
    if not model_manager.models:
        logger.error("没有可用的模型，请检查API配置")
        return False
    
    logger.info("系统初始化完成")
    return True

def test_single_generation(question_id=None):
    """测试单个报告生成"""
    logger.info("开始测试单个报告生成...")
    
    # 重置token统计
    model_manager.reset_token_usage()
    
    # 加载问题数据
    data_path = Path("data/preliminary.json")
    if not data_path.exists():
        logger.error("找不到题目数据文件")
        return False
    
    with open(data_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # 选择测试问题
    if question_id:
        # 根据ID查找问题
        target_question = None
        for q in questions:
            if q["id"] == f"smp25_pre_{question_id:03d}":
                target_question = q
                break
        if not target_question:
            logger.error(f"找不到问题ID: smp25_pre_{question_id:03d}")
            return False
    else:
        # 默认使用第一个问题
        target_question = questions[0]
    
    test_question = target_question["question"]
    test_category = target_question["type"]
    test_word_limit = target_question["word_limit"]
    
    logger.info(f"测试问题: {target_question['id']}")
    logger.info(f"题目: {test_question}")
    logger.info(f"类别: {test_category}")
    logger.info(f"字数限制: {test_word_limit}")
    
    try:
        result = report_generator.generate_report(test_question, test_category, test_word_limit)
        
        word_count = report_generator.count_words(result)
        accuracy = (word_count / test_word_limit) * 100
        
        # 获取token使用统计
        token_stats = model_manager.get_token_usage()
        
        logger.info(f"测试完成！生成报告字数: {word_count}/{test_word_limit} ({accuracy:.1f}%)")
        logger.info(f"Token使用统计:")
        logger.info(f"  - 输入Token: {token_stats['input_tokens']}")
        logger.info(f"  - 输出Token: {token_stats['output_tokens']}")
        logger.info(f"  - 总Token: {token_stats['total_tokens']}")
        logger.info(f"  - API请求数: {token_stats['requests']}")
        logger.info(f"报告预览: {result[:200]}...")
        
        return True
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False

def run_batch_generation(input_file: str, output_file: str):
    """运行批量生成"""
    logger.info(f"开始批量生成: {input_file} -> {output_file}")
    
    if not Path(input_file).exists():
        logger.error(f"输入文件不存在: {input_file}")
        return False
    
    try:
        final_output = report_generator.batch_generate(input_file, output_file)
        logger.info(f"批量生成完成，结果保存至: {final_output}")
        return final_output
    except Exception as e:
        logger.error(f"批量生成失败: {e}")
        return False

def run_post_processing(input_file: str, output_file: str = None):
    """运行字数后处理"""
    logger.info(f"开始字数后处理: {input_file}")
    
    if not Path(input_file).exists():
        logger.error(f"输入文件不存在: {input_file}")
        return False
    
    try:
        processed_output = report_generator.post_process_word_count(input_file, output_file)
        logger.info(f"字数后处理完成，结果保存至: {processed_output}")
        return processed_output
    except Exception as e:
        logger.error(f"字数后处理失败: {e}")
        return False

def show_stats():
    """显示系统统计信息"""
    logger.info("=== 系统状态 ===")
    
    # 模型状态
    model_status = model_manager.get_status()
    logger.info("可用模型:")
    for role, model_info in model_status["available_models"].items():
        logger.info(f"  {role}: {model_info['provider']}:{model_info['model_name']}")
    
    logger.info(f"可用服务商: {', '.join(model_status['available_providers'])}")
    
    # 知识库状态
    kb_stats = rag_system.get_collection_stats()
    logger.info("知识库统计:")
    for category, count in kb_stats.items():
        logger.info(f"  {category}: {count} 个文档")
    
    # 配置信息
    logger.info(f"主力模型配置: {config.PRIMARY_MODEL}")
    logger.info(f"辅助模型配置: {config.SECONDARY_MODEL}")
    logger.info(f"备用模型配置: {config.BACKUP_MODEL}")
    logger.info(f"字数容忍度: {config.WORD_COUNT_TOLERANCE}")
    logger.info(f"服务商优先级: {config.PROVIDER_PRIORITY}")
    
    # 自适应字数增益统计
    gain_stats = report_generator.adaptive_gain.get_statistics()
    logger.info("自适应字数增益统计:")
    logger.info(f"  启用状态: {gain_stats['enabled']}")
    logger.info(f"  管理类别数: {gain_stats['total_categories']}")
    logger.info(f"  参数配置: α={gain_stats['parameters']['alpha']}, 范围=[{gain_stats['parameters']['min_gain']:.2f}, {gain_stats['parameters']['max_gain']:.2f}]")
    
    if gain_stats['categories']:
        logger.info("  各类别增益系数:")
        for category, data in gain_stats['categories'].items():
            logger.info(f"    {category}: {data['current_gain']:.4f} (样本数: {data['total_samples']}, 上次准确率: {data['last_accuracy']:.1f}%)")
    else:
        logger.info("  暂无类别数据")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SMP 2025 报告生成系统")
    parser.add_argument("--mode", choices=["init", "test", "batch", "post-process", "stats"], 
                       default="init", help="运行模式")
    parser.add_argument("--input", help="输入文件路径")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--question_id", type=int, help="测试特定问题ID (1-xxx)")
    parser.add_argument("--post-process", action="store_true", 
                       help="批量生成后自动进行字数后处理")
    
    args = parser.parse_args()
    
    if args.mode == "init":
        logger.info("初始化模式")
        if init_system():
            logger.info("系统初始化成功，可以开始使用")
            logger.info("使用 --mode test 测试单个生成")
            logger.info("使用 --mode test --question_id 2 测试特定问题")
            logger.info("使用 --mode batch --input <输入文件> --output <输出文件> 进行批量生成")
            logger.info("使用 --mode batch --post-process 进行批量生成并自动后处理")
            logger.info("使用 --mode post-process --input <文件> 对已有结果进行字数后处理")
        else:
            logger.error("系统初始化失败")
            sys.exit(1)
    
    elif args.mode == "test":
        logger.info("测试模式")
        if not init_system():
            sys.exit(1)
        if test_single_generation(args.question_id):
            logger.info("测试成功")
        else:
            logger.error("测试失败")
            sys.exit(1)
    
    elif args.mode == "batch":
        if not args.input or not args.output:
            logger.error("批量模式需要指定 --input 和 --output 参数")
            sys.exit(1)
        
        logger.info("批量生成模式")
        if not init_system():
            sys.exit(1)
        
        batch_result = run_batch_generation(args.input, args.output)
        if batch_result:
            logger.info("批量生成成功")
            
            # 检查是否需要自动后处理
            if getattr(args, 'post_process', False):
                logger.info("开始自动字数后处理...")
                post_result = run_post_processing(batch_result)
                if post_result:
                    logger.info("自动后处理完成")
                else:
                    logger.error("自动后处理失败")
                    sys.exit(1)
        else:
            logger.error("批量生成失败")
            sys.exit(1)
    
    elif args.mode == "post-process":
        if not args.input:
            logger.error("后处理模式需要指定 --input 参数")
            sys.exit(1)
        
        logger.info("字数后处理模式")
        if not init_system():
            sys.exit(1)
        
        if run_post_processing(args.input, args.output):
            logger.info("字数后处理成功")
        else:
            logger.error("字数后处理失败")
            sys.exit(1)
    
    elif args.mode == "stats":
        logger.info("统计模式")
        if init_system():
            show_stats()
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()
