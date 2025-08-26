"""
报告生成核心模块
"""
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger

from .config import config
from .rag_system import rag_system
from .multi_provider_model import model_manager
from .prompt_templates import prompt_templates
from .external_data import external_data_manager

class ReportGenerator:
    """报告生成器 - 核心业务逻辑"""
    
    def __init__(self):
        self.rag = rag_system
        self.model_manager = model_manager
        self.prompts = prompt_templates
        self.external_data = external_data_manager
    
    def count_words(self, text: str) -> int:
        """
        按赛题要求统计单词数：以空格为分隔符，不包含标点符号
        赛题要求："以空格为分隔符进行统计的单词数量，不包含标点符号"
        """
        # 按空格分割
        words = text.split()
        # 过滤掉纯标点符号的"单词"，保留包含字母数字的单词
        valid_words = []
        for word in words:
            # 移除单词两端的标点符号，检查是否还有内容
            clean_word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
            if clean_word:  # 如果移除标点后还有内容，则算作一个单词
                valid_words.append(clean_word)
        return len(valid_words)
    
    def clean_report_artifacts(self, text: str) -> str:
        """
        强力清理报告中的格式化标记和不需要的文本
        移除所有结构性标题和重复内容，保留有意义的内容标题
        """
        # 移除字数标记，如 (210 words), (785 words) 等
        text = re.sub(r'\(\s*\d+\s*words?\s*\)', '', text)
        
        # 移除常见的结构性标题（无论在行首还是行中）
        structural_patterns = [
            # 英文结构性标题
            r'(?:^|\s)##?\s*Executive Summary\s*(?:\n|$)',
            r'(?:^|\s)##?\s*Main Analysis\s*(?:\n|$)', 
            r'(?:^|\s)##?\s*Main Content\s*(?:\n|$)',
            r'(?:^|\s)##?\s*Conclusion\s*(?:\n|$)',
            r'(?:^|\s)##?\s*Conclusion & Recommendations\s*(?:\n|$)',
            r'(?:^|\s)##?\s*Recommendations\s*(?:\n|$)',
            # 处理出现在句子中的情况
            r'\.\s*Executive Summary\s*[\.:]?',
            r'\.\s*Main Analysis\s*[\.:]?',
            r'\.\s*Conclusion\s*[\.:]?',
            r'\.\s*Conclusion & Recommendations\s*[\.:]?',
            # 移除明显的扩展标记
            r'\.\s*Additionally,\s*',
            r'\s*Additionally,\s*',
            # 中文结构性标题
            r'(?:^|\s)##?\s*执行摘要\s*(?:\n|$)',
            r'(?:^|\s)##?\s*主要分析\s*(?:\n|$)',
            r'(?:^|\s)##?\s*结论\s*(?:\n|$)',
            r'(?:^|\s)##?\s*建议\s*(?:\n|$)',
        ]
        
        for pattern in structural_patterns:
            text = re.sub(pattern, ' ', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # 检测并移除大段重复内容（评论员提到的核心问题）
        # 将文本分割为句子
        sentences = re.split(r'[.!?]+\s+', text)
        cleaned_sentences = []
        sentence_fingerprints = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # 跳过太短的句子
                cleaned_sentences.append(sentence)
                continue
            
            # 创建句子指纹（去除数字和标点，保留核心词汇）
            fingerprint = re.sub(r'[^\w\s]', '', sentence.lower())
            fingerprint = re.sub(r'\d+', '', fingerprint)
            fingerprint = ' '.join(fingerprint.split()[:8])  # 取前8个词作为指纹
            
            if fingerprint in sentence_fingerprints:
                # 检测到重复句子，跳过
                continue
            
            sentence_fingerprints.add(fingerprint)
            cleaned_sentences.append(sentence)
        
        # 重新组合文本
        text = '. '.join(cleaned_sentences)
        text = re.sub(r'\.\s*\.', '.', text)  # 移除多余的句号
        
        # 移除多余的空行和空格
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # 将多个空行合并为两个空行
        text = re.sub(r'[ \t]+', ' ', text)  # 将多个空格合并为一个空格
        text = text.strip()  # 移除首尾空白
        
        return text
    
    def assess_complexity(self, question: str, word_limit: int) -> float:
        """评估题目复杂度"""
        complexity_score = 0.0
        
        # 字数要求越高，复杂度越高
        if word_limit > 1100:
            complexity_score += 0.3
        elif word_limit > 1000:
            complexity_score += 0.2
        
        # 包含特定关键词增加复杂度
        complex_keywords = [
            "global", "comprehensive", "analysis", "assessment", 
            "impact", "future", "challenges", "governance",
            "全球", "综合", "评估", "影响", "未来", "挑战", "治理"
        ]
        
        question_lower = question.lower()
        for keyword in complex_keywords:
            if keyword in question_lower:
                complexity_score += 0.1
        
        return min(complexity_score, 1.0)
    
    def select_generation_strategy(self, question: str, word_limit: int) -> str:
        """选择生成策略"""
        complexity = self.assess_complexity(question, word_limit)
        
        if complexity > 0.7 and word_limit > 1000:
            return "dual_validation"
        elif word_limit > 1000:
            return "multi_step"
        else:
            return "single_shot"
    
    def generate_single_shot(self, question: str, category: str, word_limit: int) -> str:
        """单次生成策略"""
        logger.info(f"使用单次生成策略: {question[:50]}...")
        
        # 1. RAG检索
        local_context = self.rag.retrieve(question, category, top_k=8)
        local_text = "\n\n".join([doc['text'] for doc in local_context])
        
        # 2. 外部数据获取
        external_data = self.external_data.fetch_data(question, category, limit=5)
        external_text = self.external_data.format_external_data(external_data)
        
        # 3. 构建完整context
        full_context = f"""
本地知识库检索结果：
{local_text}

外部数据补充：
{external_text}
"""
        
        # 4. 选择模型
        model_key = self.model_manager.select_model(category, word_limit)
        
        # 5. 生成报告（应用反向折扣策略）
        boosted_word_limit = int(word_limit * config.WORD_COUNT_GENERATION_BOOST)
        logger.info(f"原始字数目标: {word_limit}, 折扣后生成目标: {boosted_word_limit}")
        
        generation_prompt = self.prompts.get_generation_prompt(
            question, full_context[:config.MAX_CONTEXT_LENGTH], 
            "", boosted_word_limit, category
        )
        
        report = self.model_manager.generate_with_retry(generation_prompt, model_key)
        
        # 6. 字数优化
        optimized_report = self.optimize_word_count_simple(report, word_limit)
        
        return optimized_report
    
    def generate_multi_step(self, question: str, category: str, word_limit: int) -> str:
        """多步骤生成策略"""
        logger.info(f"使用多步骤生成策略: {question[:50]}...")
        
        # Step 1: 题目分析
        analysis_prompt = self.prompts.get_analysis_prompt(question, category)
        model_key = self.model_manager.select_model(category, word_limit)
        analysis = self.model_manager.generate_with_retry(analysis_prompt, model_key)
        
        # Step 2: 信息收集
        local_context = self.rag.retrieve(question, category, top_k=12)
        local_text = "\n\n".join([doc['text'] for doc in local_context])
        
        external_data = self.external_data.fetch_data(question, category, limit=8)
        external_text = self.external_data.format_external_data(external_data)
        
        full_context = f"""
题目分析：
{analysis}

本地知识库：
{local_text}

外部数据：
{external_text}
"""
        
        # Step 3: 大纲生成
        outline_prompt = self.prompts.get_outline_prompt(
            question, full_context[:config.MAX_CONTEXT_LENGTH], 
            word_limit, category
        )
        outline = self.model_manager.generate_with_retry(outline_prompt, model_key)
        
        # Step 4: 报告生成（应用反向折扣策略）
        boosted_word_limit = int(word_limit * config.WORD_COUNT_GENERATION_BOOST)
        logger.info(f"多步策略 - 原始字数目标: {word_limit}, 折扣后生成目标: {boosted_word_limit}")
        
        generation_prompt = self.prompts.get_generation_prompt(
            question, full_context[:config.MAX_CONTEXT_LENGTH], 
            outline, boosted_word_limit, category
        )
        report = self.model_manager.generate_with_retry(generation_prompt, model_key)
        
        # Step 5: 字数优化
        optimized_report = self.optimize_word_count_simple(report, word_limit)
        
        return optimized_report
    
    def generate_dual_validation(self, question: str, category: str, word_limit: int) -> str:
        """双模型验证生成策略"""
        logger.info(f"使用双模型验证策略: {question[:50]}...")
        
        # 信息收集（与多步骤相同）
        local_context = self.rag.retrieve(question, category, top_k=15)
        local_text = "\n\n".join([doc['text'] for doc in local_context])
        
        external_data = self.external_data.fetch_data(question, category, limit=10)
        external_text = self.external_data.format_external_data(external_data)
        
        full_context = f"""
本地知识库：
{local_text}

外部数据：
{external_text}
"""
        
        # 双模型生成（应用反向折扣策略）
        boosted_word_limit = int(word_limit * config.WORD_COUNT_GENERATION_BOOST)
        logger.info(f"双模型策略 - 原始字数目标: {word_limit}, 折扣后生成目标: {boosted_word_limit}")
        
        generation_prompt = self.prompts.get_generation_prompt(
            question, full_context[:config.MAX_CONTEXT_LENGTH], 
            "", boosted_word_limit, category
        )
        
        report = self.model_manager.dual_model_validation(generation_prompt, category)
        
        # 字数优化 - 使用简化策略
        optimized_report = self.optimize_word_count_simple(report, word_limit)
        
        return optimized_report
    
    def optimize_word_count_simple(self, content: str, target_words: int) -> str:
        """智能字数控制 - 简化的三档分级制"""
        current_words = self.count_words(content)
        
        logger.info(f"字数优化: 当前{current_words}词, 目标{target_words}词")
        
        # 简化的三档分级制
        acceptable_min = int(target_words * config.WORD_COUNT_ACCEPTABLE_MIN_RATIO)  # 90%
        acceptable_max = int(target_words * config.WORD_COUNT_ACCEPTABLE_MAX_RATIO)  # 110%
        expandable_min = int(target_words * config.WORD_COUNT_EXPANDABLE_MIN_RATIO)  # 60%
        
        # 1. 完全可接受范围：90%-110%
        if acceptable_min <= current_words <= acceptable_max:
            logger.info("字数在完全可接受范围内")
            return content
        
        # 2. 字数超出110%，执行截断
        if current_words > acceptable_max:
            logger.info(f"字数超出可接受范围，执行截断到{target_words}词")
            words = content.split()
            truncated_content = ' '.join(words[:target_words])
            
            # 确保以句号结束
            if not truncated_content.endswith('.'):
                last_period = truncated_content.rfind('.')
                if last_period > len(truncated_content) * 0.8:
                    truncated_content = truncated_content[:last_period + 1]
                else:
                    truncated_content += '.'
            
            final_words = self.count_words(truncated_content)
            logger.info(f"截断后实际字数: {final_words}词")
            return truncated_content
        
        # 3. 字数不足90%，判断是否可扩写
        if current_words < acceptable_min:
            # 3a. 字数过低（<60%），建议重新生成
            if current_words < expandable_min:
                shortage = target_words - current_words
                shortage_ratio = shortage / target_words
                logger.warning(f"字数严重不足{shortage}词（{shortage_ratio*100:.1f}%），建议重新生成")
                return content
            
            # 3b. 字数适度不足（60%-90%），进行智能扩写
            else:
                shortage = target_words - current_words
                logger.info(f"字数不足{shortage}词，进行智能扩写")
                return self._intelligent_expand(content, target_words)
        
        return content
    
    def _intelligent_expand(self, content: str, target_words: int) -> str:
        """使用辅助模型进行智能扩写"""
        current_words = self.count_words(content)
        needed_words = target_words - current_words
        
        # 使用配置文件中的打折系数，避免过度扩写
        conservative_needed = int(needed_words * config.WORD_COUNT_EXPANSION_DISCOUNT)
        
        logger.info(f"使用辅助模型扩写: 当前{current_words}词, 需要增加约{conservative_needed}词 (原需求{needed_words}词，已打折)")
        
        # 构建扩写提示（优化后的英文版本）
        expand_prompt = f"""Please expand the following analytical report by approximately {conservative_needed} words to enhance its depth and comprehensiveness.

Current word count: {current_words}
Target expansion: ~{conservative_needed} words
Maximum acceptable total: {int(target_words * 1.05)} words

Original Report:
{content}

Expansion Guidelines:
1. Maintain the core arguments and logical structure of the original text
2. Add specific case studies, quantitative data, or deeper analytical insights
3. Avoid repetition of existing points and perspectives
4. Ensure professional tone with innovative and forward-thinking language
5. Integrate new content seamlessly with the existing writing style
6. Avoid generic transition words like "Additionally", "Furthermore", "Moreover"
7. Focus on substantive content rather than filler text
8. Keep expansion concise and purposeful - quality over quantity

Output the complete expanded report without any additional explanations or meta-commentary:"""

        try:
            # 使用辅助模型进行扩写
            expanded_content = self.model_manager.generate_with_retry(
                expand_prompt, "secondary"
            )
            
            # 清理扩写结果
            expanded_content = self.clean_report_artifacts(expanded_content)
            
            # 检查扩写效果
            new_word_count = self.count_words(expanded_content)
            
            # 验证扩写质量
            if new_word_count < current_words:
                logger.warning("扩写后字数反而减少，使用原文")
                return content
            
            # 使用配置文件中的截断阈值
            if new_word_count > target_words * config.WORD_COUNT_TRUNCATION_THRESHOLD:
                logger.warning(f"扩写过度({new_word_count}词 > {target_words * config.WORD_COUNT_TRUNCATION_THRESHOLD:.0f}词)，截断到目标长度")
                words = expanded_content.split()
                truncated_content = ' '.join(words[:target_words])
                if not truncated_content.endswith('.'):
                    last_period = truncated_content.rfind('.')
                    if last_period > len(truncated_content) * 0.8:
                        truncated_content = truncated_content[:last_period + 1]
                    else:
                        truncated_content += '.'
                expanded_content = truncated_content
                new_word_count = self.count_words(expanded_content)
            
            logger.info(f"扩写完成: {current_words} -> {new_word_count}词 ({new_word_count/target_words*100:.1f}%)")
            return expanded_content
            
        except Exception as e:
            logger.error(f"扩写失败: {e}")
            logger.info("扩写失败，返回原文")
            return content
    
    def generate_report(self, question: str, category: str, word_limit: int) -> str:
        """主生成接口"""
        try:
            logger.info(f"开始生成报告: {question[:50]}... (类别: {category}, 字数: {word_limit})")
            
            # 选择生成策略
            strategy = self.select_generation_strategy(question, word_limit)
            logger.info(f"选择生成策略: {strategy}")
            
            # 根据策略生成报告
            if strategy == "single_shot":
                report = self.generate_single_shot(question, category, word_limit)
            elif strategy == "multi_step":
                report = self.generate_multi_step(question, category, word_limit)
            else:  # dual_validation
                report = self.generate_dual_validation(question, category, word_limit)
            
            # 清理格式标记和不需要的文本
            report = self.clean_report_artifacts(report)
            
            # 最终质量检查
            final_word_count = self.count_words(report)
            logger.info(f"报告生成完成，最终字数: {final_word_count}")
            
            return report
            
        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            raise

    def batch_generate(self, questions_file: str, output_file: str) -> None:
        """批量生成报告"""
        from datetime import datetime
        import os
        
        logger.info(f"开始批量生成，输入文件: {questions_file}")
        
        # 重置token统计
        self.model_manager.reset_token_usage()
        
        # 加载题目
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        # 生成带时间戳的输出文件名
        if not output_file.startswith('/'):
            # 相对路径，添加时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(output_file).stem
            ext = Path(output_file).suffix
            output_dir = Path(output_file).parent
            
            # 确保output目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamped_filename = f"{base_name}_{timestamp}{ext}"
            final_output_file = output_dir / timestamped_filename
        else:
            final_output_file = Path(output_file)
        
        logger.info(f"输出文件: {final_output_file}")
        
        results = []
        total_questions = len(questions)
        successful_count = 0
        
        logger.info(f"总共需要处理 {total_questions} 个题目")
        
        for i, item in enumerate(questions, 1):
            try:
                logger.info(f"正在处理第 {i}/{total_questions} 题: {item['id']}")
                
                question_id = item['id']
                question = item['question']
                category = item['type']
                word_limit = item['word_limit']
                
                # 生成报告
                answer = self.generate_report(question, category, word_limit)
                word_count = self.count_words(answer)
                accuracy = (word_count / word_limit) * 100
                
                # 构建结果
                result = {
                    "id": question_id,
                    "question": question,
                    "type": category,
                    "word_limit": word_limit,
                    "answer": answer
                }
                
                results.append(result)
                successful_count += 1
                
                # 获取当前token使用情况
                token_stats = self.model_manager.get_token_usage()
                
                logger.info(f"第 {i} 题完成！字数: {word_count}/{word_limit} ({accuracy:.1f}%)")
                logger.info(f"累计Token使用: 输入{token_stats['input_tokens']}, 输出{token_stats['output_tokens']}, 请求{token_stats['requests']}次")
                
                # 每5题保存一次（防止意外中断丢失进度）
                if i % 5 == 0:
                    # 创建backup子目录
                    backup_dir = final_output_file.parent / "backup"
                    backup_dir.mkdir(exist_ok=True)
                    
                    # 生成备份文件名
                    backup_file = backup_dir / f"{final_output_file.stem}_backup_{i}{final_output_file.suffix}"
                    
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    logger.info(f"已保存进度备份: {backup_file}")
                
            except Exception as e:
                logger.error(f"第 {i} 题生成失败: {e}")
                # 添加空结果，避免数组错位
                results.append({
                    "id": item.get('id', f'error_{i}'),
                    "question": item.get('question', ''),
                    "type": item.get('type', ''),
                    "word_limit": item.get('word_limit', 0),
                    "answer": f"生成失败: {str(e)}"
                })
        
        # 保存最终结果
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 生成统计报告
        final_token_stats = self.model_manager.get_token_usage()
        logger.info(f"=== 批量生成完成统计 ===")
        logger.info(f"成功生成: {successful_count}/{total_questions} 题")
        logger.info(f"总Token消耗: {final_token_stats['total_tokens']}")
        logger.info(f"输入Token: {final_token_stats['input_tokens']}")
        logger.info(f"输出Token: {final_token_stats['output_tokens']}")
        logger.info(f"API请求次数: {final_token_stats['requests']}")
        logger.info(f"结果保存至: {final_output_file}")
        
        return str(final_output_file)  # 返回实际使用的文件名
    
    def post_process_word_count(self, input_file: str, output_file: str = None) -> str:
        """后处理：优化字数不达标的报告"""
        logger.info("开始执行字数后处理流程")
        
        # 读取原始结果
        with open(input_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 统计需要处理的题目
        need_expansion = []
        need_truncation = []
        acceptable = []
        
        for i, result in enumerate(results):
            current_words = self.count_words(result['answer'])
            target_words = result['word_limit']
            accuracy = current_words / target_words
            
            if accuracy < 0.80:  # 字数不足80%
                need_expansion.append((i, result, current_words, target_words))
            elif accuracy > 1.15:  # 字数超出15%
                need_truncation.append((i, result, current_words, target_words))
            else:
                acceptable.append((i, result, current_words, target_words))
        
        logger.info(f"字数分析: 需扩展{len(need_expansion)}题, 需截断{len(need_truncation)}题, 合格{len(acceptable)}题")
        
        # 处理需要扩展的题目
        expansion_count = 0
        for i, result, current_words, target_words in need_expansion:
            logger.info(f"扩展题目{i+1} ({result['id']}): {current_words}/{target_words}词 ({current_words/target_words*100:.1f}%)")
            
            try:
                expanded_answer = self._intelligent_expand(result['answer'], target_words)
                new_word_count = self.count_words(expanded_answer)
                
                if new_word_count > current_words:
                    results[i]['answer'] = expanded_answer
                    expansion_count += 1
                    logger.info(f"扩展成功: {current_words} -> {new_word_count}词 ({new_word_count/target_words*100:.1f}%)")
                else:
                    logger.warning(f"扩展失败，保持原文")
                    
            except Exception as e:
                logger.error(f"扩展题目{i+1}时出错: {e}")
        
        # 处理需要截断的题目
        truncation_count = 0
        for i, result, current_words, target_words in need_truncation:
            logger.info(f"截断题目{i+1} ({result['id']}): {current_words}/{target_words}词 ({current_words/target_words*100:.1f}%)")
            
            try:
                # 使用优化后的截断逻辑
                words = result['answer'].split()
                truncated_content = ' '.join(words[:target_words])
                
                # 确保以句号结束
                if not truncated_content.endswith('.'):
                    last_period = truncated_content.rfind('.')
                    if last_period > len(truncated_content) * 0.8:
                        truncated_content = truncated_content[:last_period + 1]
                    else:
                        truncated_content += '.'
                
                new_word_count = self.count_words(truncated_content)
                results[i]['answer'] = truncated_content
                truncation_count += 1
                logger.info(f"截断完成: {current_words} -> {new_word_count}词 ({new_word_count/target_words*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"截断题目{i+1}时出错: {e}")
        
        # 保存后处理结果
        if output_file is None:
            # 在原文件名基础上添加后缀
            base_name = Path(input_file).stem
            output_file = Path(input_file).parent / f"{base_name}_POST_PROCESSED.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 生成后处理统计
        final_stats = {"需扩展": len(need_expansion), "需截断": len(need_truncation), "合格": len(acceptable)}
        process_stats = {"成功扩展": expansion_count, "成功截断": truncation_count}
        
        logger.info(f"=== 字数后处理完成 ===")
        logger.info(f"原始统计: {final_stats}")
        logger.info(f"处理结果: {process_stats}")
        logger.info(f"结果保存至: {output_file}")
        
        # 最终字数统计
        final_word_stats = []
        for result in results:
            word_count = self.count_words(result['answer'])
            accuracy = word_count / result['word_limit']
            final_word_stats.append(accuracy)
        
        avg_accuracy = sum(final_word_stats) / len(final_word_stats) * 100
        good_count = sum(1 for acc in final_word_stats if 0.80 <= acc <= 1.15)
        
        logger.info(f"最终字数质量: 平均准确率{avg_accuracy:.1f}%, 合格率{good_count}/{len(results)}题 ({good_count/len(results)*100:.1f}%)")
        
        return str(output_file)

# 全局报告生成器实例
report_generator = ReportGenerator()
