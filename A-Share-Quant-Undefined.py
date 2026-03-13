import akshare as ak
import pandas as pd
from openai import OpenAI
import time
import requests
import os
from datetime import datetime

# ==========================================
# ⚙️ 配置区 (通过环境变量读取，保护隐私)
# ==========================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not all([OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    raise ValueError("⚠️ 环境变量未完全配置，请检查相关 Secrets！")

# 初始化 OpenRouter 客户端 (使用 openai SDK)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ==========================================
# 模块一：Telegram 消息推送
# ==========================================
def send_telegram_message(text):
    print("    正在推送到 Telegram...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"    ⚠️ Telegram 推送失败: {response.text}")
    except Exception as e:
        print(f"    ⚠️ Telegram 请求异常: {e}")

# ==========================================
# 模块二：宏观经济数据获取
# ==========================================
def get_macro_cpi():
    try:
        cpi_df = ak.macro_china_cpi_monthly()
        latest_record = cpi_df.iloc[0] 
        return f"最新中国CPI指数 ({latest_record['月份']}): 全国同比 {latest_record['全国-同比']}%"
    except Exception:
        return "宏观CPI数据暂未获取到"

# ==========================================
# 模块三：数据获取与机械初筛 (已优化)
# ==========================================
def fetch_and_mechanical_screen():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] >>> 开始执行每周量化初筛...")
    try:
        df = ak.stock_zh_a_spot_em()
    except Exception as e:
        print(f"数据获取失败: {e}")
        return pd.DataFrame()

    # 1. 剔除ST、退市、北交所(8开头)及老代码
    df = df[~df['名称'].str.contains('ST|退')]
    df = df[~df['代码'].str.startswith(('8', '4'))]

    # 2. 估值与市值底线过滤 (剔除50亿以下微盘股，放宽PE上限以容纳周期股)
    df = df[df['总市值'] > 5000000000] 
    df = df[(df['市盈率-动态'] > 0) & (df['市盈率-动态'] < 50)] 
    
    # 3. 盈利能力过滤 (公式：PB/PE = ROE)
    df['近似ROE(%)'] = (df['市净率'] / df['市盈率-动态']) * 100
    df = df[df['近似ROE(%)'] > 6.0] 

    # 4. 流动性与量价状态
    df = df[(df['换手率'] > 1.5) & (df['换手率'] < 20.0)]
    # 剔除单日暴跌的股票，保留左侧缩量回调的机会
    df = df[df['涨跌幅'] > -7.0]

    # 5. 综合打分
    df['综合得分'] = df['近似ROE(%)'].rank(ascending=False) 
    
    # 取 Top 3 演示
    top_stocks = df.nsmallest(3, '综合得分') 
    
    print(f"    初筛完成！选出 {len(top_stocks)} 只候选股进入 AI 研判池。")
    return top_stocks

# ==========================================
# 模块四：获取新闻
# ==========================================
def get_recent_news(stock_code):
    try:
        news_df = ak.stock_news_em(symbol=stock_code)
        if news_df.empty: return "暂无近期重要新闻。"
        
        news_text = ""
        for index, row in news_df.head(5).iterrows():
            news_text += f"【{row['发布时间']}】{row['新闻标题']}\n摘要: {row['新闻内容']}\n\n"
        return news_text
    except Exception:
        return "新闻获取失败。"

# ==========================================
# 模块五：AI 深度研判 (OpenRouter + Claude 3 Opus)
# ==========================================
def ai_qualitative_analysis(stock_name, stock_code, pe, roe, pct_chg, turnover, news_text, cpi_data):
    print(f"    正在调用 Claude 3 Opus 深度分析 [{stock_name}]...")
    
    report_theme = "【景气成长】、【外需突围】与【周期反转】"
    financial_data = f"动态市盈率: {pe:.2f}倍, 近似ROE: {roe:.2f}%"
    price_volume_status = f"当日涨跌幅: {pct_chg}%, 换手率: {turnover}%"
    
    prompt = f"""
# Role: 顶级A股“量化+基本面”（Quantamental）双重驱动基金经理
# Time: 2026年
# Context: 2026年A股市场风格趋向均衡。核心超额收益集中在【景气成长】、【外需突围】与【周期反转】三条主线。投资策略要求“基本面防守（严控质量与估值） + 量价进攻（顺势择时）”。

# 任务目标
请基于以下输入的股票与宏观信息，进行深度的多步逻辑推理，评估股票 {stock_name}({stock_code}) 是否具有建仓价值。

## 输入数据
- 宏观基本面：{cpi_data}
- 核心投资主线：{report_theme}
- 最新财务与估值数据：{financial_data}
- 近期量价与技术面状态：{price_volume_status}
- 最新新闻/研报摘要：{news_text}

## 推理步骤（请严格按照以下步骤思考并输出）：

### 第一步：基本面与财务排雷 (Financial Risk Check)
分析 {financial_data}。结合当前宏观数据，如果存在以下任意情况，请直接触发一票否决：
1. 存在明显的财务造假风险迹象。
2. 新闻中存在严重的监管问询或负面黑天鹅。
*请简述排雷结论。*

### 第二步：核心逻辑与主线契合度 (Moat & Theme Alignment)
结合 {news_text} 与宏观 CPI 数据，评估该公司的核心护城河是否真实存在，并分析其是否深度契合 {report_theme} 主线逻辑。对于“出海”或“科技”逻辑，必须重点评估其面临的地缘政治风险及供应链脆弱性。
*请给出深度定性分析。*

### 第三步：量价择时状态评估 (Quant/Timing Analysis)
分析 {price_volume_status}。结合基本面，判断当前是属于“右侧突破”、“左侧缩量回调”还是“估值修复”。判断当前是否具备安全的买入赔率。
*请给出量价匹配度分析。*

### 第四步：最终决策与结构化输出 (Final Decision)
综合以上三步，给出最终的投资评级。

## 输出格式要求
请务必以严格的 Markdown 格式输出，不要包含多余的客套话：

**【最终结论】**：[仅限填写：强烈推荐 / 逐步建仓 / 保留观察 / 建议剔除]
**【核心驱动因子】**：[用3-5个词概括，如：海外毛利扩张、缩量回调、出海逻辑兑现]
**【风险提示】**：[一句话指出最大风险点]
**【决策理由（150字以内）】**：[精炼总结你的投资逻辑，必须包含宏观/基本面与量价的共振点]
"""
    try:
        response = client.chat.completions.create(
            model="anthropic/claude-opus-4.6",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 分析失败: {e}"

# ==========================================
# 核心工作流 (单次运行)
# ==========================================
def weekly_rebalance_job():
    send_telegram_message("🚀 *2026 A股双擎选股系统* \n每周调仓研判已启动，正在进行全市场数据扫描...")
    
    # 优先获取宏观数据底座
    current_cpi = get_macro_cpi()
    
    candidate_stocks = fetch_and_mechanical_screen()
    
    if candidate_stocks.empty:
        send_telegram_message("⚠️ 今日无符合机械初筛条件的股票。")
        return

    for index, row in candidate_stocks.iterrows():
        stock_code = row['代码']
        stock_name = row['名称']
        pe = row['市盈率-动态']
        roe = row['近似ROE(%)']
        pct_chg = row['涨跌幅']
        turnover = row['换手率']
        
        news_text = get_recent_news(stock_code)
        
        # 将 CPI 数据一并喂给 AI
        ai_report = ai_qualitative_analysis(
            stock_name, stock_code, pe, roe, pct_chg, turnover, news_text, current_cpi
        )
        
        tg_msg = f"🎯 *【{stock_name} ({stock_code})】深度体检报告*\n"
        tg_msg += "-" * 30 + "\n"
        tg_msg += ai_report
        
        send_telegram_message(tg_msg)
        time.sleep(5) 
        
    send_telegram_message("✅ *本周调仓研判报告推送完毕！*")

if __name__ == "__main__":
    weekly_rebalance_job()
