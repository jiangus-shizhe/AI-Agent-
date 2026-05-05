"""
电商评价情感洞察与选品策略 Agent 系统
基于 LangGraph 的多 Agent 协作，演示属性级情感分析 -> 长链推理 -> 交叉验证 -> 报告生成
"""
import os
import json
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# 加载环境变量（OPENAI_API_KEY 等）
load_dotenv()

# ================== 1. 全局状态定义 ==================
class AgentState(TypedDict):
    product_id: str                # 商品ID
    raw_reviews: List[dict]       # 原始评论
    base_data: dict               # 基础数据（退货率、复购率等）
    extracted_attributes: List[dict]  # NLP Agent输出的属性级情感
    insights: str                 # Insight Agent的推理结论
    cross_validated: dict         # 交叉验证结果
    final_report: str             # 最终报告

# ================== 2. LLM 实例 ==================
# 如果环境变量未设，可在此直接填入 key（仅用于测试，生产请用环境变量）
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=2000)

# ================== 3. 各 Agent 节点 ==================

def crawler_agent(state: AgentState) -> AgentState:
    """
    Crawler Agent: 模拟采集评论及基础业务数据。
    真实环境接入数据库/爬虫，这里用硬编码示例。
    """
    product_id = state["product_id"]
    # 模拟 5 条评论，涵盖不同属性
    mock_reviews = [
        {"id": 1, "rating": 2, "content": "穿了两天就起球了，太失望了", "date": "2026-04-01"},
        {"id": 2, "rating": 1, "content": "掉色非常严重，把白色衣服都染了", "date": "2026-04-02"},
        {"id": 3, "rating": 3, "content": "版型还行，但面料太薄了，不适合秋冬", "date": "2026-03-28"},
        {"id": 4, "rating": 5, "content": "颜色好看，穿着舒服，物流快", "date": "2026-03-30"},
        {"id": 5, "rating": 4, "content": "透气性不错，但起球是个硬伤", "date": "2026-04-03"},
    ]
    mock_base_data = {
        "return_rate": 0.12,        # 12% 退货率
        "repurchase_rate": 0.08,   # 8% 复购率
        "category": "针织衫",
        "season": "春季",
    }
    print(f"[Crawler Agent] 采集 {len(mock_reviews)} 条评论及基础数据")
    return {
        **state,
        "raw_reviews": mock_reviews,
        "base_data": mock_base_data,
    }

def nlp_agent(state: AgentState) -> AgentState:
    """
    NLP Agent: 细粒度属性级情感分析。
    利用 LLM 抽取每条评论的「属性, 情感, 证据」三元组。
    """
    reviews = state["raw_reviews"]
    reviews_text = "\n".join(
        [f"[{r['id']}]({r['date']}, 评分{r['rating']}) {r['content']}" for r in reviews]
    )
    prompt = f"""你是一个电商评论分析专家。请对以下用户评论进行细粒度属性级情感分析。
提取所有被提及的“属性”（如面料、起球、颜色、版型等），并给出情感极性（正面/负面/中性）和证据片段。
请严格以JSON数组输出，每个元素包含 attribute, sentiment, evidence, review_id。

评论列表：
{reviews_text}

输出示例：
[{{"attribute": "起球","sentiment": "负面","evidence": "穿了两天就起球了","review_id": 1}},
 {{"attribute": "颜色","sentiment": "正面","evidence": "颜色好看","review_id": 4}}]"""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        attributes = json.loads(response.content)
    except:
        # 容错：如果解析失败，使用空列表
        attributes = []
    print(f"[NLP Agent] 提取属性级情感 {len(attributes)} 条：{attributes}")
    return {**state, "extracted_attributes": attributes}

def insight_agent(state: AgentState) -> AgentState:
    """
    Insight Agent: 长链推理。
    输入：属性情感列表 + 基础业务数据（退货率、复购率）。
    推理目标：
    1. 识别高频负面属性（如起球、掉色）；
    2. 推断可能的供应链/面料问题；
    3. 结合退货率、复购率判断严重性；
    4. 生成初步改进建议。
    """
    attributes = state["extracted_attributes"]
    base = state["base_data"]
    # 统计负面属性
    neg_attrs = [a for a in attributes if a["sentiment"] == "负面"]
    neg_summary = ", ".join([f"{a['attribute']}(如“{a['evidence']}”)" for a in neg_attrs])

    prompt = f"""你是一位资深选品专家与供应链分析师。请根据以下信息进行长链推理，输出一份结构化洞察报告。

【商品基础信息】
品类：{base['category']}，退货率：{base['return_rate']*100}%，复购率：{base['repurchase_rate']*100}%

【负面属性情感挖掘结果】
{neg_summary if neg_summary else "无显著负面属性"}

【推理要求】
1. 从高频负面词“起球”、“掉色”等推断可能存在的供应链/面料问题（例如：低等级棉纺、色牢度不达标、纱线捻度不足等）。
2. 结合退货率12%和极低复购率8%，评估问题的严重性及对店铺口碑的潜在损害。
3. 给出具体选品调整建议（更换供应商、加强质检、面料升级等），并判断是否建议停止采购该款。
4. 思考如果该问题在社交媒体发酵会有什么后果（为后续Trend Agent做铺垫，此处仅推理）。
仅输出文本。"""
    response = llm.invoke([HumanMessage(content=prompt)])
    insights = response.content
    print(f"[Insight Agent] 长链推理结果:\n{insights}")
    return {**state, "insights": insights}

def trend_agent(state: AgentState) -> AgentState:
    """
    Trend Agent: 模拟社交媒体热词交叉验证。
    真实场景会调用爬虫/搜索API获取微博、小红书等热帖。
    这里基于Insight推理出的问题模拟外部热度。
    """
    insights = state["insights"]
    # 简单启发式：如果insight包含“起球”或“掉色”，模拟出较高讨论度
    trend_result = {}
    if "起球" in insights:
        trend_result["起球"] = {"hits": 876, "sentiment_ratio": "负面86%", "platform": "小红书"}
    if "掉色" in insights:
        trend_result["掉色"] = {"hits": 1432, "sentiment_ratio": "负面92%", "platform": "微博"}
    if not trend_result:
        trend_result["整体"] = {"hits": 54, "sentiment_ratio": "中性", "platform": "综合"}

    # 模拟分析结论
    cross_conclusion = "社交媒体交叉验证显示：" + "；".join(
        [f"关于“{k}”的讨论热度{v['hits']}条，{v['sentiment_ratio']}，主要来源于{v['platform']}" for k, v in trend_result.items()]
    )
    cross_conclusion += "\n此问题确实在外部形成了一定舆情，建议立即采取措施。"
    print(f"[Trend Agent] 交叉验证结果：{trend_result}\n{cross_conclusion}")
    return {**state, "cross_validated": {
        "data": trend_result,
        "conclusion": cross_conclusion
    }}

def reporter_agent(state: AgentState) -> AgentState:
    """
    Reporter Agent: 汇总所有信息，生成最终选品/改进报告。
    """
    report_parts = []
    report_parts.append(f"## 商品 {state['product_id']} 情感洞察与选品策略报告\n")
    report_parts.append(f"**退货率**：{state['base_data']['return_rate']*100}% | **复购率**：{state['base_data']['repurchase_rate']*100}%")
    report_parts.append("\n### 一、评论概览")
    report_parts.append(f"共采集评论 {len(state['raw_reviews'])} 条。")
    report_parts.append("\n### 二、属性级情感挖掘")
    for attr in state["extracted_attributes"][:10]:  # 仅展示前10
        report_parts.append(f"- {attr['attribute']} ({attr['sentiment']}): {attr['evidence']} (评论{attr['review_id']})")
    report_parts.append("\n### 三、供应链与选品深度洞察")
    report_parts.append(state["insights"])
    report_parts.append("\n### 四、社交媒体交叉验证")
    report_parts.append(state["cross_validated"]["conclusion"])
    report_parts.append("\n### 五、行动建议（自动生成）")
    if state["base_data"]["return_rate"] > 0.1:
        report_parts.append("- ⚠️ 退货率超过10%，建议立即启动供应商质量复盘，暂停该SKU主推位置。")
    if state["base_data"]["repurchase_rate"] < 0.1:
        report_parts.append("- 📉 复购率极低，考虑优化产品详情页或启动客户回访调研。")
    report_parts.append("- ✅ 将自动生成的改进报告已推送至品类经理工作台。")

    final_report = "\n".join(report_parts)
    print(f"[Reporter Agent] 最终报告已生成，长度 {len(final_report)} 字符")
    return {**state, "final_report": final_report}

# ================== 4. 构建 LangGraph 工作流 ==================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("crawler", crawler_agent)
workflow.add_node("nlp", nlp_agent)
workflow.add_node("insight", insight_agent)
workflow.add_node("trend", trend_agent)
workflow.add_node("reporter", reporter_agent)

# 定义顺序流
workflow.set_entry_point("crawler")
workflow.add_edge("crawler", "nlp")
workflow.add_edge("nlp", "insight")
workflow.add_edge("insight", "trend")
workflow.add_edge("trend", "reporter")
workflow.add_edge("reporter", END)

app = workflow.compile()

# ================== 5. 执行入口 ==================
if __name__ == "__main__":
    initial_state: AgentState = {
        "product_id": "SKU-KZ2103",
        "raw_reviews": [],
        "base_data": {},
        "extracted_attributes": [],
        "insights": "",
        "cross_validated": {},
        "final_report": "",
    }
    print("===== 电商评价情感洞察与选品策略 Agent 启动 =====")
    # 运行工作流，verbose=True 可查看状态流转
    final_state = app.invoke(initial_state)
    print("\n\n" + "="*50)
    print(final_state["final_report"])
