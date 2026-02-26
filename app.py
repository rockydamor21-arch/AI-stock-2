import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── 页面配置 ────────────────────────────────────────────────
st.set_page_config(page_title="A股智能分析平台", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Serif SC', serif; background-color: #0a0e1a; color: #e0e6f0; }
.stApp { background-color: #0a0e1a; }
.score-high { color: #ff4d4d; font-weight: 700; font-size: 2em; }
.score-mid  { color: #ffd700; font-weight: 700; font-size: 2em; }
.score-low  { color: #4da6ff; font-weight: 700; font-size: 2em; }
.signal-bullish { background: rgba(255,77,77,0.15); border-left: 3px solid #ff4d4d; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
.signal-bearish { background: rgba(77,166,255,0.15); border-left: 3px solid #4da6ff; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
.signal-neutral { background: rgba(255,215,0,0.10); border-left: 3px solid #ffd700; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
.advice-box { background: linear-gradient(135deg,#0d1b2a,#0a1628); border: 1px solid #ffd700; border-radius: 12px; padding: 20px; margin: 10px 0; }
.stButton > button { background: linear-gradient(135deg,#c0392b,#e74c3c); color: white; border: none; border-radius: 8px; font-weight: 600; width: 100%; padding: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📈 A股智能量化分析平台")
st.markdown("##### 技术面 · 资金流向 · K线趋势 · AI综合研判")
st.divider()

# ─── 常用股票预设 ─────────────────────────────────────────────
POPULAR = {
    "贵州茅台": "600519.SS",
    "平安银行": "000001.SZ",
    "宁德时代": "300750.SZ",
    "比亚迪":   "002594.SZ",
    "中芯国际": "688981.SS",
    "招商银行": "600036.SS",
    "隆基绿能": "601012.SS",
    "迈瑞医疗": "300760.SZ",
}

# ─── 侧边栏 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 分析配置")

    st.markdown("**快速选股（点击填入）**")
    cols = st.columns(2)
    selected_quick = None
    for i, (name, code) in enumerate(POPULAR.items()):
        if cols[i % 2].button(name, key=f"q_{code}"):
            selected_quick = code.replace(".SS","").replace(".SZ","")

    st.markdown("---")
    default_val = selected_quick if selected_quick else "600519, 000001, 300750"
    symbol_input = st.text_input("输入股票代码（逗号分隔）", value=default_val,
                                  help="输入6位代码，如 600519（上证加.SS，深证加.SZ，系统自动识别）")
    symbols_raw = [s.strip() for s in symbol_input.split(",") if s.strip()]

    period_map = {"近1月": "1mo", "近3月": "3mo", "近6月": "6mo"}
    period_label = st.selectbox("分析周期", list(period_map.keys()), index=1)
    period_str = period_map[period_label]

    invest_style = st.multiselect("投资风格", ["短线(1-5天)", "中线(1-4周)"],
                                   default=["短线(1-5天)", "中线(1-4周)"])
    run_btn = st.button("🚀 开始智能分析")

# ─── 辅助函数 ─────────────────────────────────────────────────

def to_yf_symbol(code):
    """将6位A股代码转为yfinance格式"""
    code = code.strip()
    if "." in code:
        return code.upper()
    if code.startswith("6") or code.startswith("5"):
        return code + ".SS"
    elif code.startswith("0") or code.startswith("3") or code.startswith("2"):
        return code + ".SZ"
    elif code.startswith("8") or code.startswith("4"):
        return code + ".BJ"
    return code + ".SS"

@st.cache_data(ttl=300)
def get_data(yf_sym, period):
    try:
        ticker = yf.Ticker(yf_sym)
        df = ticker.history(period=period, interval="1d")
        if df is None or df.empty:
            return None, None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[['Open','High','Low','Close','Volume']].copy()
        df.columns = ['open','high','low','close','volume']
        df = df.astype(float).dropna()
        info = ticker.info
        name = info.get('longName', info.get('shortName', yf_sym))
        return df, name
    except Exception as e:
        return None, None

def compute_indicators(df):
    close = df['close']
    volume = df['volume']
    df['EMA5']  = ta.ema(close, length=5)
    df['EMA10'] = ta.ema(close, length=10)
    df['EMA20'] = ta.ema(close, length=20)
    df['EMA60'] = ta.ema(close, length=min(60, len(df)-1))
    df['RSI']   = ta.rsi(close, length=14)
    bb = ta.bbands(close, length=20, std=2)
    if bb is not None: df = pd.concat([df, bb], axis=1)
    macd = ta.macd(close)
    if macd is not None: df = pd.concat([df, macd], axis=1)
    df['vol_ma5'] = volume.rolling(5).mean()
    df.dropna(inplace=True)
    return df

def score_stock(df):
    if df is None or len(df) < 3: return None
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    close   = float(latest['close'])
    volume  = float(latest['volume'])
    vol_avg = float(df['volume'].tail(10).mean())
    signals, score = [], 0

    ema5  = float(latest.get('EMA5',  close))
    ema20 = float(latest.get('EMA20', close))
    ema60 = float(latest.get('EMA60', close))

    if close > ema5 > ema20:
        score += 8; signals.append(("🔴 多头排列", "bullish", "短均线站上中均线，趋势向上"))
    elif close < ema5 < ema20:
        score -= 5; signals.append(("🔵 空头排列", "bearish", "价格跌破短期均线，趋势向下"))

    if close > ema60:
        score += 5; signals.append(("🔴 站上60日均线", "bullish", "中期趋势健康"))
    else:
        signals.append(("🔵 跌破60日均线", "bearish", "中期趋势偏弱"))

    if volume > vol_avg * 2.0:
        score += 10; signals.append(("🔴 超级放量", "bullish", f"量比{volume/vol_avg:.1f}x，主力大举介入"))
    elif volume > vol_avg * 1.5:
        score += 6;  signals.append(("🔴 明显放量", "bullish", f"量比{volume/vol_avg:.1f}x，资金关注度提升"))
    elif volume < vol_avg * 0.6:
        score -= 3;  signals.append(("🔵 明显缩量", "bearish", "成交萎缩，参与度低"))

    bbu_col = [c for c in df.columns if 'BBU' in c]
    bbl_col = [c for c in df.columns if 'BBL' in c]
    bbm_col = [c for c in df.columns if 'BBM' in c]
    if bbu_col:
        bbu = float(latest[bbu_col[0]])
        bbl = float(latest[bbl_col[0]])
        bbm = float(latest[bbm_col[0]])
        if close > bbu:
            score += 8; signals.append(("🔴 突破布林上轨", "bullish", "强势突破，短期动能强劲"))
        elif close < bbl:
            score -= 6; signals.append(("🔵 跌破布林下轨", "bearish", "超卖但需警惕继续下跌"))
        elif close > bbm:
            score += 3; signals.append(("🟡 站上布林中轨", "neutral", "中性偏多"))

    macdh_col = [c for c in df.columns if 'MACDh' in c]
    if macdh_col:
        h_now  = float(latest[macdh_col[0]])
        h_prev = float(prev[macdh_col[0]])
        if h_now > 0 and h_now > h_prev:
            score += 6; signals.append(("🔴 MACD红柱扩张", "bullish", "动能持续增强"))
        elif h_now > 0 and h_now < h_prev:
            score += 2; signals.append(("🟡 MACD红柱收缩", "neutral", "上涨动能减弱，注意高位"))
        elif h_now < 0 and h_now < h_prev:
            score -= 5; signals.append(("🔵 MACD绿柱扩张", "bearish", "下跌动能持续"))
        elif h_now < 0 and h_now > h_prev:
            score += 3; signals.append(("🟡 MACD绿柱收缩", "neutral", "下跌动能减弱，可能反弹"))

    rsi = float(latest.get('RSI', 50))
    if rsi >= 70:
        score -= 4; signals.append(("🟡 RSI超买", "neutral", f"RSI={rsi:.1f}，短期回调风险"))
    elif rsi <= 30:
        score += 5; signals.append(("🔴 RSI超卖反弹", "bullish", f"RSI={rsi:.1f}，超跌反弹机会"))
    elif 50 <= rsi < 70:
        score += 3; signals.append(("🔴 RSI强势区间", "bullish", f"RSI={rsi:.1f}，趋势延续"))

    pct = (close - float(prev['close'])) / float(prev['close']) * 100

    return {
        "score": max(0, min(100, score + 50)),
        "signals": signals,
        "rsi": rsi,
        "vol_ratio": volume / vol_avg,
        "pct_change": pct,
        "close": close,
        "df": df,
    }

def generate_advice(result, invest_style):
    score = result['score']
    close = result['close']
    rsi   = result['rsi']
    df    = result['df']
    support = round(float(df['low'].tail(20).min()) * 1.01, 2)
    advice = {}

    if "短线(1-5天)" in invest_style:
        if score >= 70:
            advice['短线'] = {"操作": "🟢 积极做多",
                "入场": f"现价 {close:.2f} 附近分2-3次建仓",
                "目标": f"{round(close*1.05,2)} ~ {round(close*1.08,2)}",
                "止损": f"跌破 {round(close*0.96,2)} 坚决止损",
                "仓位": "30% ~ 50%", "理由": f"评分{score}分，多指标共振，量价配合好"}
        elif score >= 55:
            advice['短线'] = {"操作": "🟡 轻仓观察",
                "入场": f"{round(close*0.98,2)} 附近小仓试探",
                "目标": f"{round(close*1.03,2)} ~ {round(close*1.05,2)}",
                "止损": f"跌破 {round(close*0.96,2)} 止损",
                "仓位": "10% ~ 20%", "理由": f"评分{score}分，信号偏弱，等待确认"}
        else:
            advice['短线'] = {"操作": "🔴 回避观望",
                "入场": "当前不建议买入",
                "目标": "等待企稳信号",
                "止损": f"持仓跌破 {round(close*0.95,2)} 务必止损",
                "仓位": "0%", "理由": f"评分{score}分，技术面偏弱"}

    if "中线(1-4周)" in invest_style:
        if score >= 65 and rsi < 65:
            advice['中线'] = {"操作": "🟢 逢低布局",
                "入场": f"{support} ~ {close:.2f} 区间分批建仓",
                "目标": f"{round(close*1.10,2)} ~ {round(close*1.15,2)}",
                "止损": f"跌破支撑 {support}",
                "仓位": "20% ~ 40%", "理由": "趋势向上，RSI未超买，中线性价比高"}
        elif score >= 50:
            advice['中线'] = {"操作": "🟡 持股观望",
                "入场": "已持仓继续持有，未持仓暂不追",
                "目标": f"{round(close*1.08,2)}",
                "止损": f"跌破 {support} 减仓",
                "仓位": "维持现有", "理由": "趋势中性，等待方向选择"}
        else:
            advice['中线'] = {"操作": "🔴 规避风险",
                "入场": "不建议中线持有",
                "目标": "等待底部确认",
                "止损": f"持仓设 {round(close*0.93,2)} 为止损位",
                "仓位": "0% ~ 10%", "理由": "中期趋势偏弱"}
    return advice

def draw_kline(df, sym, name):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='#ff4d4d', decreasing_line_color='#4da6ff', name='K线'), row=1, col=1)

    for col, color in [('EMA5','#ffd700'),('EMA10','#ff9500'),('EMA20','#ff4dff'),('EMA60','#4dffff')]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=color, width=1), name=col), row=1, col=1)

    bbu_col = [c for c in df.columns if 'BBU' in c]
    bbl_col = [c for c in df.columns if 'BBL' in c]
    if bbu_col and bbl_col:
        fig.add_trace(go.Scatter(x=df.index, y=df[bbu_col[0]], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'), name='布林上轨'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[bbl_col[0]], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'), name='布林下轨', fill='tonexty', fillcolor='rgba(255,255,255,0.02)'), row=1, col=1)

    vol_colors = ['#ff4d4d' if df['close'].iloc[i] >= df['open'].iloc[i] else '#4da6ff' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=vol_colors, name='成交量', opacity=0.7), row=2, col=1)
    if 'vol_ma5' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['vol_ma5'], line=dict(color='#ffd700', width=1), name='量5均'), row=2, col=1)

    macdh_col = [c for c in df.columns if 'MACDh' in c]
    macd_col  = [c for c in df.columns if c.startswith('MACD_')]
    macds_col = [c for c in df.columns if c.startswith('MACDs')]
    if macdh_col:
        mc = ['#ff4d4d' if v >= 0 else '#4da6ff' for v in df[macdh_col[0]]]
        fig.add_trace(go.Bar(x=df.index, y=df[macdh_col[0]], marker_color=mc, name='MACD柱'), row=3, col=1)
    if macd_col:
        fig.add_trace(go.Scatter(x=df.index, y=df[macd_col[0]], line=dict(color='#ffd700', width=1), name='MACD'), row=3, col=1)
    if macds_col:
        fig.add_trace(go.Scatter(x=df.index, y=df[macds_col[0]], line=dict(color='#ff9500', width=1), name='Signal'), row=3, col=1)

    fig.update_layout(
        title=dict(text=f"{name}（{sym}）走势分析", font=dict(size=16, color='#e0e6f0')),
        template="plotly_dark", paper_bgcolor='#0d1b2a', plot_bgcolor='#0a1020',
        xaxis_rangeslider_visible=False, height=680,
        legend=dict(orientation='h', y=1.02, font=dict(size=10)),
        margin=dict(l=50, r=20, t=60, b=20)
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    return fig

# ─── 主逻辑 ──────────────────────────────────────────────────

if run_btn:
    symbols_yf = [(s, to_yf_symbol(s)) for s in symbols_raw]

    # 多股榜单
    if len(symbols_yf) > 1:
        st.markdown("## 📊 多股评分榜单")
        scan_results = []
        prog = st.progress(0)
        for i, (raw, yf_sym) in enumerate(symbols_yf):
            prog.progress((i+1)/len(symbols_yf))
            df, name = get_data(yf_sym, period_str)
            if df is not None and len(df) > 30:
                df = compute_indicators(df)
                r = score_stock(df)
                if r:
                    adv = generate_advice(r, invest_style)
                    scan_results.append({
                        "代码": raw, "名称": name[:10] if name else raw,
                        "现价": round(r['close'], 2),
                        "综合评分": r['score'],
                        "RSI": round(r['rsi'], 1),
                        "量比": f"{r['vol_ratio']:.2f}x",
                        "今日涨跌": f"{r['pct_change']:+.2f}%",
                        "短线建议": adv.get('短线', {}).get('操作', '-')
                    })
        prog.empty()
        if scan_results:
            df_scan = pd.DataFrame(scan_results).sort_values("综合评分", ascending=False)
            st.dataframe(df_scan, use_container_width=True, hide_index=True)

    # 逐股深度分析
    for raw, yf_sym in symbols_yf:
        st.markdown("---")
        st.markdown(f"## 🔍 {raw} 深度分析报告")

        with st.spinner(f"正在获取 {yf_sym} 数据..."):
            df, name = get_data(yf_sym, period_str)

        if df is None or len(df) < 30:
            st.error(f"**{raw}** 数据获取失败。请确认代码正确，如：600519（茅台）、000001（平安银行）")
            continue

        df = compute_indicators(df)
        result = score_stock(df)
        if not result:
            st.error(f"{raw} 指标计算失败，数据可能不足")
            continue

        result['df'] = df
        advice = generate_advice(result, invest_style)
        score = result['score']

        # 指标卡片
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("股票名称", (name[:8] if name else raw))
        with c2: st.metric("当前价格", f"¥{result['close']:.2f}", f"{result['pct_change']:+.2f}%")
        with c3:
            sc = "score-high" if score>=70 else "score-mid" if score>=50 else "score-low"
            st.markdown("**综合评分**")
            st.markdown(f"<span class='{sc}'>{score}</span>/100", unsafe_allow_html=True)
        with c4: st.metric("RSI", f"{result['rsi']:.1f}", "超买" if result['rsi']>70 else "超卖" if result['rsi']<30 else "正常")
        with c5: st.metric("量比", f"{result['vol_ratio']:.2f}x", "放量🔴" if result['vol_ratio']>1.5 else "缩量🔵" if result['vol_ratio']<0.7 else "正常")

        # K线图
        st.plotly_chart(draw_kline(df, raw, name or raw), use_container_width=True)

        # 信号 + 建议
        col_sig, col_adv = st.columns(2)
        with col_sig:
            st.markdown("### 📡 技术信号解读")
            for sig_name, sig_type, sig_desc in result['signals']:
                st.markdown(f"<div class='signal-{sig_type}'><b>{sig_name}</b><br><small>{sig_desc}</small></div>", unsafe_allow_html=True)

        with col_adv:
            st.markdown("### 💡 智能投资建议")
            for style_name, adv in advice.items():
                st.markdown(
                    f"<div class='advice-box'>"
                    f"<b>【{style_name}】{adv['操作']}</b><br><br>"
                    f"📌 <b>入场：</b>{adv['入场']}<br>"
                    f"🎯 <b>目标：</b>{adv['目标']}<br>"
                    f"🛡️ <b>止损：</b>{adv['止损']}<br>"
                    f"💼 <b>仓位：</b>{adv['仓位']}<br><br>"
                    f"<small>📝 {adv['理由']}</small></div>",
                    unsafe_allow_html=True
                )

        # AI提示词
        st.markdown("### 🤖 发送给AI深度分析")
        prompt = f"""你是资深A股操盘手，请对 {name}（{raw}）进行深度研判：

【量化评分】{score}/100
【技术数据】RSI={result['rsi']:.1f}，量比={result['vol_ratio']:.2f}x，今日{result['pct_change']:+.2f}%
【技术信号】{' | '.join([s[0] for s in result['signals']])}

请分析：
1. 当前技术形态与关键位置判断
2. 短线（1-3天）具体操作策略与点位
3. 中线（1-4周）趋势与持仓策略  
4. 必须止损离场的条件
5. 一句话总结：买入/观望/回避"""
        st.text_area("复制发送给 Claude / Gemini：", value=prompt, height=200, key=f"p_{raw}")

else:
    st.markdown("""
    <div style='text-align:center; padding:60px 20px;'>
        <h2 style='color:#ffd700'>欢迎使用 A股智能分析平台</h2>
        <p style='color:#8899aa; font-size:16px'>在左侧选择股票或输入代码，点击「开始智能分析」</p>
        <br>
        <table style='margin:auto; color:#aabbcc; border-collapse:collapse;'>
            <tr><td style='padding:12px 24px;border:1px solid #2a3f6f'>📊 K线技术分析</td><td style='padding:12px 24px;border:1px solid #2a3f6f'>均线·MACD·RSI·布林带</td></tr>
            <tr><td style='padding:12px 24px;border:1px solid #2a3f6f'>📈 量价分析</td><td style='padding:12px 24px;border:1px solid #2a3f6f'>成交量·量比·主力资金信号</td></tr>
            <tr><td style='padding:12px 24px;border:1px solid #2a3f6f'>💡 投资建议</td><td style='padding:12px 24px;border:1px solid #2a3f6f'>短线+中线双维度策略</td></tr>
            <tr><td style='padding:12px 24px;border:1px solid #2a3f6f'>🤖 AI研判</td><td style='padding:12px 24px;border:1px solid #2a3f6f'>自动生成提示词发给AI</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
