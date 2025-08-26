# 🏆 Tianchi SMP 2025 - AI报告生成系统

## 📖 项目简介

这是一个用于天池SMP 2025比赛的AI驱动报告生成系统，采用多提供商模型架构、RAG检索增强生成和智能字数控制技术，实现了高质量的自动化报告生成。

### 🎯 核心特性

- **🤖 多提供商模型支持**: 支持OpenAI、Anthropic、SiliconFlow、阿里百炼等多个模型提供商
- **🔍 RAG检索增强**: 基于ChromaDB的向量数据库，支持多类别知识检索
- **📊 智能字数控制**: 1.12反向折扣策略，实现94.8%平均字数准确率
- **🔄 自动降级机制**: 主模型失效时自动切换到备用模型
- **📈 外部数据集成**: 支持ArXiv、RSS、NewsAPI等多种数据源
- **⚡ 批量处理**: 高效的并行生成和后处理流程

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/ZakuZakuu/Tianchi-SMP2025-ReportGen.git
cd Tianchi-SMP2025-ReportGen

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

### 2. API密钥配置

在 `.env` 文件中配置你的API密钥：

```env
# 至少需要配置一个模型提供商
DASHSCOPE_API_KEY=your_dashscope_api_key_here
SILICONFLOW_API_KEY=your_siliconflow_api_key_here

# 可选：其他提供商
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. 基本使用

```bash
# 单题测试
python main.py --mode test --question_id "1" --input data/test_sample.json

# 批量生成
python main.py --mode batch --input data/test_sample.json --output output/results.json --post-process

# 更新外部数据
python external_data_config.py --action both --category "Cutting-Edge Tech & AI"
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 天池平台提供的比赛环境
- 各大模型提供商的API支持
- 开源社区的技术支持

**⭐ 如果这个项目对你有帮助，请给我们一个星标！**
