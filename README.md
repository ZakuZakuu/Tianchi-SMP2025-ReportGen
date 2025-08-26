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

### 🏅 比赛成绩

- **排名**: 前十名 (Top 10)
- **字数准确率**: 94.8%平均准确率
- **成功率**: 100%生成成功率

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

## 📁 项目结构

```
Tianchi-SMP2025-ReportGen/
├── main.py                    # 主入口文件
├── external_data_config.py    # 外部数据管理
├── external_data_preprocessor.py # 数据预处理
├── .env.example              # 环境变量模板
├── requirements.txt          # 依赖列表
├── src/                      # 核心模块
│   ├── config.py            # 配置管理
│   ├── multi_provider_model.py # 多提供商模型
│   ├── rag_system.py        # RAG检索系统
│   ├── report_generator.py  # 报告生成核心
│   ├── prompt_templates.py  # 提示词模板
│   └── external_data.py     # 外部数据接口
├── data/                    # 数据文件
│   ├── preliminary.json     # 比赛数据
│   └── test_sample.json     # 测试数据
└── output/                  # 输出目录
```

## 🎛️ 核心技术

### 1.12反向折扣策略
```python
# 解决主模型系统性字数不足问题
WORD_COUNT_GENERATION_BOOST=1.12
```

### 多层字数优化
- **90%-110%**: 完全可接受范围
- **60%-90%**: 智能扩写
- **<60%**: 建议重新生成

### 自动降级机制
```
主模型(qwen3-235b) → 辅助模型(qwen-flash) → 备用模型(qwen-plus)
```

## 📊 性能指标

| 指标 | 值 |
|------|------|
| 平均字数准确率 | 94.8% |
| 生成成功率 | 100% |
| 平均处理时间 | 2-3分钟/题 |
| 模型降级成功率 | 100% |

## 🔧 高级配置

### 外部数据源
```bash
# 查看缓存统计
python external_data_config.py --action stats

# 更新特定类别数据
python external_data_config.py --action both --category "Business Models & Market Dynamics" --limit 10
```

### 字数控制微调
```env
# 可根据需要调整字数控制参数
WORD_COUNT_GENERATION_BOOST=1.12  # 主生成提升系数
WORD_COUNT_EXPANSION_DISCOUNT=0.92 # 扩写打折系数
```

## 🛠️ 开发指南

### 添加新模型提供商
1. 在 `src/multi_provider_model.py` 中添加新的提供商类
2. 在 `.env` 中配置相应的API密钥
3. 更新 `PROVIDER_PRIORITY` 设置

### 自定义提示词
编辑 `src/prompt_templates.py` 中的模板内容

### 扩展外部数据源
在 `external_data_config.py` 中添加新的数据源配置

## 📈 监控和调试

```bash
# 查看详细日志
python main.py --mode test --question_id "1" --input data/test_sample.json --verbose

# 检查RAG系统状态
python -c "from src.rag_system import RAGSystem; rag = RAGSystem(); print(rag.get_collection_stats())"
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 天池平台提供的比赛环境
- 各大模型提供商的API支持
- 开源社区的技术支持

## 📞 联系我们

如有问题或建议，请通过以下方式联系：

- 创建 Issue
- 发起 Discussion

---

**⭐ 如果这个项目对你有帮助，请给我们一个星标！**
