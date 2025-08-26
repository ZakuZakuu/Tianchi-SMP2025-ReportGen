"""
提示词模板管理
"""
from typing import Dict, Any
from .config import config

class PromptTemplates:
    """提示词模板管理器"""
    
    @staticmethod
    def get_analysis_prompt(question: str, category: str) -> str:
        """Question analysis prompt"""
        return f"""
Please conduct an in-depth analysis of the following report topic requirements and key points:

Topic: {question}
Category: {category}

Please provide:
1. Core keywords and concepts
2. Main analytical dimensions
3. Key content focus areas
4. Suggested logical structure
5. Critical aspects requiring special attention

Requirements: The analysis should be thorough and comprehensive, providing clear guidance for subsequent report writing.
"""

    @staticmethod
    def get_outline_prompt(question: str, context: str, word_limit: int, category: str) -> str:
        """Outline generation prompt - optimized word allocation"""
        # Calculate word allocation
        exec_summary_words = int(word_limit * 0.18)
        main_content_words = int(word_limit * 0.72)
        conclusion_words = int(word_limit * 0.10)
        
        return f"""
Based on the following information, create a precise word allocation outline for the report "{question}":

Report Category: {category}
Total Word Requirement: {word_limit} words (strict limit)

Word Allocation Plan:
1. Executive Summary: {exec_summary_words} words
2. Main Analysis: {main_content_words} words
3. Conclusion & Recommendations: {conclusion_words} words

Reference Materials:
{context}

Please provide a detailed outline including:
1. Executive Summary key points (within {exec_summary_words} words)
   - Core problem overview
   - Main findings
   - Key recommendations

2. Main Analysis structure (within {main_content_words} words)
   - Divide into 2-3 major sections
   - Word allocation for each section
   - Core arguments and supporting evidence

3. Conclusion & Recommendations (within {conclusion_words} words)
   - Summary of key points
   - Strategic recommendations
   - Future outlook

Requirements:
- Ensure precise word count control within {word_limit} words total
- Each section should be substantial but not exceed allocated word count
- Clear logical structure with highlighted focus areas
- Write entirely in English
"""

    @staticmethod
    def get_category_template(category: str) -> Dict[str, Any]:
        """获取分类专用模板"""
        
        templates = {
            "Cutting-Edge Tech & AI": {
                "structure": [
                    "Executive Summary (15%)",
                    "Technology Landscape & Current State (25%)", 
                    "Applications & Market Impact (25%)",
                    "Technical Challenges & Limitations (20%)",
                    "Future Outlook & Strategic Recommendations (15%)"
                ],
                "focus_areas": [
                    "technical principles and innovations",
                    "application scenarios and market impact", 
                    "technical challenges and limitations",
                    "development trends and future outlook",
                    "strategic recommendations and policy implications"
                ],
                "tone": "professional, forward-looking, objective analysis"
            },
            
            "Business Models & Market Dynamics": {
                "structure": [
                    "Executive Summary (15%)",
                    "Market Context & Background (20%)",
                    "Current Business Model Analysis (30%)", 
                    "Market Dynamics & Competitive Landscape (20%)",
                    "Strategic Implications & Future Outlook (15%)"
                ],
                "focus_areas": [
                    "market background and business environment",
                    "core logic of business models",
                    "market dynamics and competitive landscape", 
                    "value creation and profit models",
                    "strategic implications and development prospects"
                ],
                "tone": "business-savvy, data-driven, strategic orientation"
            },
            
            "Sustainability & Environmental Governance": {
                "structure": [
                    "Executive Summary (15%)",
                    "Environmental Context & Challenges (25%)",
                    "Current Governance Framework (25%)",
                    "Policy Analysis & Implementation (20%)", 
                    "Future Directions & Recommendations (15%)"
                ],
                "focus_areas": [
                    "environmental status and challenges",
                    "governance framework and policy systems",
                    "implementation effectiveness and case studies",
                    "international cooperation and best practices",
                    "future directions and policy recommendations"
                ],
                "tone": "scientifically rigorous, policy-oriented, global perspective"
            },
            
            "Social Change & Cultural Trends": {
                "structure": [
                    "Executive Summary (15%)",
                    "Social Context & Cultural Background (25%)",
                    "Current Trends & Phenomena Analysis (30%)",
                    "Driving Forces & Impact Assessment (15%)",
                    "Future Implications & Societal Impact (15%)"
                ],
                "focus_areas": [
                    "social background and cultural environment",
                    "current trends and phenomena analysis",
                    "driving factors and impact mechanisms",
                    "social groups and generational differences",
                    "future implications and social significance"
                ],
                "tone": "humanistic care, deep insights, social responsibility"
            },
            
            "Life Sciences & Public Health": {
                "structure": [
                    "Executive Summary (15%)",
                    "Scientific Background & Current State (25%)",
                    "Research Findings & Clinical Evidence (25%)",
                    "Public Health Implications (20%)",
                    "Future Research & Policy Recommendations (15%)"
                ],
                "focus_areas": [
                    "scientific principles and research background",
                    "latest research findings and evidence",
                    "clinical applications and efficacy assessment",
                    "public health policies and impact",
                    "future research directions and recommendations"
                ],
                "tone": "scientifically rigorous, evidence-based medicine, public interest"
            },
            
            "Global Affairs & Future Governance": {
                "structure": [
                    "Executive Summary (15%)",
                    "Global Context & Geopolitical Landscape (25%)",
                    "Current Governance Mechanisms (25%)",
                    "Challenges & Emerging Issues (20%)",
                    "Future Governance Models & Recommendations (15%)"
                ],
                "focus_areas": [
                    "geopolitics and international relations",
                    "governance mechanisms and institutional frameworks",
                    "global challenges and emerging issues",
                    "multilateral cooperation and coordination mechanisms",
                    "future governance models and reform directions"
                ],
                "tone": "international perspective, policy analysis, strategic thinking"
            }
        }
        
        return templates.get(category, templates["Cutting-Edge Tech & AI"])

    @staticmethod
    def get_generation_prompt(question: str, context: str, outline: str, 
                            word_limit: int, category: str) -> str:
        """主生成prompt - 优化字数控制"""
        template_info = PromptTemplates.get_category_template(category)
        
        # 计算各段落字数分配
        exec_summary_words = int(word_limit * 0.18)  # 18%给执行摘要
        main_content_words = int(word_limit * 0.72)  # 72%给主体内容
        conclusion_words = int(word_limit * 0.10)    # 10%给结论
        
        return f"""
As a senior expert and thought leader in {category}, please write a high-quality, innovative professional analysis report based on the following information.

【IMPORTANT: WRITE ENTIRELY IN ENGLISH】
All content must be written in English only, regardless of the language of the question or reference materials.

【STRICT WORD COUNT CONTROL】
Total word target: {word_limit} words (must be within 90-110% range for quality)
- Executive Summary: {exec_summary_words} words
- Main Content: {main_content_words} words  
- Conclusion: {conclusion_words} words

【REPORT TITLE】
{question}

【REFERENCE OUTLINE】
{outline}

【REFERENCE MATERIALS】
{context}

【INNOVATION & ORIGINALITY REQUIREMENTS】
1. AVOID generic viewpoints - provide unique perspectives and fresh insights
2. Challenge conventional wisdom where appropriate
3. Propose novel frameworks, models, or analytical approaches
4. Identify emerging trends not commonly discussed
5. Offer forward-thinking predictions and recommendations
6. Integrate interdisciplinary perspectives where relevant

【CONTENT QUALITY STANDARDS】
1. NO repetitive content - each paragraph must add new value
2. Use specific, recent examples and data (2024-2025 preferred)
3. Provide contrarian or nuanced viewpoints alongside mainstream consensus
4. Include unexpected connections between concepts
5. Offer actionable, innovative recommendations

【STRUCTURE REQUIREMENTS】
Please write strictly according to the following format and word count requirements:

## Executive Summary
[Provide a compelling executive summary within {exec_summary_words} words, including your unique angle on the issue, key innovative insights, and transformative recommendations]

## Main Analysis
[Provide original, multi-dimensional analysis within {main_content_words} words, including:
- Current technological status and development trends (with unique perspective)
- Application scenarios and market impact (emphasize emerging, unconventional uses)
- Challenges and limitations faced (identify overlooked issues)
- Solutions and innovation directions (propose novel approaches)]

## Conclusion
[Provide visionary summary and innovative recommendations within {conclusion_words} words]

【CRITICAL QUALITY CHECKS】
1. Ensure NO sentence or paragraph repetition
2. Each section must offer distinct, non-overlapping insights
3. Avoid clichéd recommendations like "invest in training" or "establish frameworks"
4. Provide specific, actionable, and innovative solutions
5. Writing style: {template_info['tone']} with innovative edge
6. MANDATORY: Write entirely in English - no Chinese text allowed

Please start writing the complete report content now, focusing on originality and fresh perspectives:
"""

    @staticmethod
    def get_word_optimization_prompt(content: str, target_words: int, current_words: int) -> str:
        """Word count optimization prompt"""
        if current_words > target_words:
            action = "compress"
            instruction = f"Please compress the following report from {current_words} words to approximately {target_words} words, preserving core information and key arguments"
        else:
            action = "expand"
            instruction = f"Please expand the following report from {current_words} words to approximately {target_words} words, adding supporting details and in-depth analysis"
        
        return f"""
{instruction}.

Requirements:
1. Maintain the report's core structure and main arguments
2. Preserve professionalism and logical flow
3. {'Remove redundant information and streamline expression' if action == 'compress' else 'Add specific examples, data support, and detailed analysis'}
4. Ensure final word count is within {target_words}±{int(target_words * 0.02)} words range
5. Write entirely in English

Original Report:
{content}

Please provide the complete optimized report directly:
"""

# 全局提示词模板实例
prompt_templates = PromptTemplates()
