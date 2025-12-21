# Invalid Results（协议不合规输出说明）

本目录用于存放在 **cross_model_evaluation（v2）** 实验中，
**未满足 EVAL_PROTOCOL_v2 评测协议要求** 的模型输出结果。

这些结果 **不参与定量统计与结论汇总**，
仅用于分析模型在结构化评测任务中的失败模式（failure modes）。

---

## 一、什么样的结果会被归类为 Invalid

凡满足以下任一条件的输出，均被判定为 *Invalid*：

1. **未按三段式结构输出**
   - 缺失或混淆以下任一部分：
     - 事实快照（Fact Snapshot）
     - ChatGPT 联网搜索指令
     - Gemini 深度挖掘指令

2. **违反事实快照约束**
   - 快照超过 50 字
   - 在快照中出现分析、解释、建议等内容

3. **评测证据不合规**
   - evidence 字段未逐字引用原始输出
   - 出现概括性表述、判断性语言或省略（如“无漂移”“完全失败”等）
   - 无法在原始输出中精确定位对应文本

4. **任务漂移（Prompt Drift）**
   - 输出退化为：
     - 提示词诊断 / 优化建议
     - 元层面讨论（如“这个问题本身是误解”）
     - 普通问答或长篇分析文章
   - 而非执行研究型评测模板

5. **JSON / 协议层面违规**
   - 非法 JSON
   - 输出中混入非 JSON 文本
   - evidence 与评分不一致（如 evidence 为空但得分 > 0）

---

## 二、Invalid ≠ 无价值

需要强调的是：

> **Invalid 结果并不代表模型能力低下，  
而是反映模型在“强结构化评测协议”下的遵循失败。**

因此，本目录的主要研究价值在于：

- 分析不同模型在以下方面的系统性差异：
  - 显式 vs 隐式指令触发
  - 结构化模板的稳健性
  - 长提示词、弱提示词、冲突提示词下的漂移模式
- 为后续 Prompt Drift 机制分析提供定性材料

---

## 三、目录结构说明

```txt
invalid_results/
├── main_method/
│   └── judge_*_v2.json
├── supporting_method/
│   ├── chatgpt.json
│   ├── claude.json
│   └── gemini.json
├── EVAL_PROTOCOL_v2.md
└── PROMPT_MANIFEST_v2.md
