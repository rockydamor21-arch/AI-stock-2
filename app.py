import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI智能量化分析平台", page_icon="📈", layout="wide")

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
.info-box { background: rgba(255,215,0,0.08); border: 1px solid #ffd700; border-radius: 8px; padding: 12px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📈 AI智能量化分析平台")
st.markdown("##### 技术面 · K线趋势 · 量价分析 · AI综合研判")
st.divider()

# ─── 预设股票 ─────────────────────────────────────────────────
# 由于Streamlit Cloud服务器在海外，A股用.SS/.SZ，港股用.HK，美股直接代码
STOCKS = {
    "🇨🇳 贵州茅台": "600519.SS",
    "🇨🇳 平安银行": "000001.SZ",
    "🇨🇳 宁德时代": "300750.SZ",
    "🇨🇳 比亚迪":   "002594.SZ",
    "🇨🇳 招商银行": "600036.SS",
    "🇺🇸 英伟达":   "NVDA",
    "🇺🇸 苹果":     "AAPL",
    "🇺🇸 特斯拉":   "TSLA",
}

with st.sidebar:
    st.markdown("## ⚙️ 分析配置")
    st.markdown("**热门股票（点击填入）**")
    cols = st.columns(2)
    clicked_sym = None
    for i, (name, code) in enumerate(STOCKS.items()):
        if cols[i % 2].button(name, key=f"btn_{i}"):
            clicked_sym = code

    st.markdown("---")
    default = clicked_sym if clicked_sym else "NVDA, AAPL, TSLA"
    symbol_input = st.text_input(
        "输入股票代码（逗号分隔）",
        value=default,
        help="美股直接输入：NVDA\nA股加后缀：600519.SS（上证）或000001.SZ（深证）"
    )
    symbols = [s.strip().upper() for s in symbol_input.split(",") if s.strip()]

    period_map = {"近1月": "1mo", "近3月": "3mo", "近6月": "6mo"}
    period_label = st.selectbox("分析周期", list(period_map.keys()), index=1)
    period_str = period_map[period_label]

    invest_style = st.multiselect(
        "投资风格",
        ["短线(1-5天)", "中线(1-4周)"],
        default=["短线(1-5天)", "中线(1-4周)"]
    )

    st.markdown("---")
    st.markdown("""
    <div class='info-box'>
    <b>💡 A股说明</b><br>
    上交所：代码+.SS<br>
    例：600519.SS（茅台）<br><br>
    深交所：代码+.SZ<br>
    例：000001.SZ（平安）
    </div>
    """, unsafe_allow_html=True)

    run_btn = st.button("🚀 开始智能分析")

# ─── 核心函数 ─────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_data(symbol, period):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d", timeout=15)
        if df is None or len(df) < 5:
            return None, symbol
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[['Open','High','Low','Close','Volume']].copy()
        df.columns = ['open','high','low','close','volume']
        df = df.astype(float).dropna()
        try:
            info = ticker.fast_info
            name = getattr(info, 'last_price', None)
            info2 = ticker.info
            name = info2.get('longName') or info2.get('shortName') or symbol
        except:
            name = symbol
        return df, name
    except Exception as e:
        return None, symbol

def compute_indicators(df):
    close = df['close']
    vol   = df['volume']
    n = len(df)
    df['EMA5']  = ta.ema(close, length=min(5,  n-1))
    df['EMA10'] = ta.ema(close, length=min(10, n-1))
    df['EMA20'] = ta.ema(close, length=min(20, n-1))
    df['EMA60'] = ta.ema(close, length=min(60, n-1))
    df['RSI']   = ta.rsi(close,  length=min(14, n-1))
    if n >= 22:
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None: df = pd.concat([df, bb], axis=1)
    if n >= 35:
        macd = ta.macd(close)
        if macd is not None: df = pd.concat([df, macd], axis=1)
    df['vol_ma5'] = vol.rolling(min(5, n)).mean()
    df.dropna(inplace=True)
    return df

def score_stock(df):
    if df is None or len(df) < 2: return None
    latest  = df.iloc[-1]
    prev    = df.iloc[-2]
    close   = float(latest['close'])
    volume  = float(latest['volume'])
    vol_avg = float(df['volume'].tail(10).mean()) or 1
    signals, score = [], 0

    # 均线趋势
    e5  = float(latest.get('EMA5',  close))
    e20 = float(latest.get('EMA20', close))
    e60 = float(latest.get('EMA60', close))
    if close > e5 > e20:
        score += 8; signals.append(("🔴 多头排列", "bullish", "短均线站上中均线，趋势向上"))
    elif close < e5 < e20:
        score -= 5; signals.append(("🔵 空头排列", "bearish", "价格跌破短期均线，趋势向下"))
    if close > e60:
        score += 5; signals.append(("🔴 站上60日均线", "bullish", "中期趋势健康"))
    else:
        signals.append(("🔵 跌破60日均线", "bearish", "中期趋势偏弱"))

    # 量能
    vr = volume / vol_avg
    if vr > 2.0:
        score += 10; signals.append(("🔴 超级放量", "bullish", f"量比{vr:.1f}x，主力大举介入"))
    elif vr > 1.5:
        score += 6;  signals.append(("🔴 明显放量", "bullish", f"量比{vr:.1f}x，资金关注度提升"))
    elif vr < 0.6:
        score -= 3;  signals.append(("🔵 明显缩量", "bearish", "成交萎缩，参与度低"))

    # 布林带
    bbu_c = [c for c in df.columns if 'BBU' in c]
    bbl_c = [c for c in df.columns if 'BBL' in c]
    bbm_c = [c for c in df.columns if 'BBM' in c]
    if bbu_c and bbl_c and bbm_c:
        bbu = float(latest[bbu_c[0]]); bbl = float(latest[bbl_c[0]]); bbm = float(latest[bbm_c[0]])
        if close > bbu:
            score += 8; signals.append(("🔴 突破布林上轨", "bullish", "强势突破，短期动能强劲"))
        elif close < bbl:
            score -= 6; signals.append(("🔵 跌破布林下轨", "bearish", "超卖，需警惕继续下跌"))
        elif close > bbm:
            score += 3; signals.append(("🟡 布林中轨上方", "neutral", "中性偏多"))

    # MACD
    mh_c = [c for c in df.columns if 'MACDh' in c]
    if mh_c:
        h_now  = float(latest[mh_c[0]])
        h_prev = float(prev[mh_c[0]])
        if h_now > 0 and h_now > h_prev:
            score += 6; signals.append(("🔴 MACD红柱扩张", "bullish", "动能持续增强"))
        elif h_now > 0 and h_now < h_prev:
            score += 2; signals.append(("🟡 MACD红柱收缩", "neutral", "上涨动能减弱"))
        elif h_now < 0 and h_now < h_prev:
            score -= 5; signals.append(("🔵 MACD绿柱扩张", "bearish", "下跌动能持续"))
        elif h_now < 0 and h_now > h_prev:
            score += 3; signals.append(("🟡 MACD绿柱收缩", "neutral", "下跌动能减弱，可能反弹"))

    # RSI
    rsi = float(latest.get('RSI', 50))
    if rsi >= 70:
        score -= 4; signals.append(("🟡 RSI超买", "neutral", f"RSI={rsi:.1f}，短期回调风险"))
    elif rsi <= 30:
        score += 5; signals.append(("🔴 RSI超卖反弹", "bullish", f"RSI={rsi:.1f}，超跌反弹机会"))
    elif 50 <= rsi < 70:
        score += 3; signals.append(("🔴 RSI强势区间", "bullish", f"RSI={rsi:.1f}，趋势延续"))

    pct = (close - float(prev['close'])) / float(prev['close']) * 100

    return {
        "score":      max(0, min(100, score + 50)),
        "signals":    signals,
        "rsi":        rsi,
        "vol_ratio":  vr,
        "pct_change": pct,
        "close":      close,
        "df":         df,
    }

def generate_advice(result, style):
    score = result['score']; close = result['close']; rsi = result['rsi']
    df    = result['df']
    sup   = round(float(df['low'].tail(20).min()) * 1.01, 2)
    advice = {}
    if "短线(1-5天)" in style:
        if score >= 70:
            advice['短线'] = {"操作":"🟢 积极做多","入场":f"现价{close:.2f}附近分2-3次建仓","目标":f"{round(close*1.05,2)}~{round(close*1.08,2)}","止损":f"跌破{round(close*0.96,2)}止损","仓位":"30%~50%","理由":f"评分{score}分，多指标共振"}
        elif score >= 55:
            advice['短线'] = {"操作":"🟡 轻仓观察","入场":f"{round(close*0.98,2)}附近小仓试探","目标":f"{round(close*1.03,2)}~{round(close*1.05,2)}","止损":f"跌破{round(close*0.96,2)}","仓位":"10%~20%","理由":f"评分{score}分，等待更强确认"}
        else:
            advice['短线'] = {"操作":"🔴 回避观望","入场":"当前不建议买入","目标":"等待企稳","止损":f"持仓跌破{round(close*0.95,2)}止损","仓位":"0%","理由":f"评分{score}分，技术面偏弱"}
    if "中线(1-4周)" in style:
        if score >= 65 and rsi < 65:
            advice['中线'] = {"操作":"🟢 逢低布局","入场":f"{sup}~{close:.2f}分批建仓","目标":f"{round(close*1.10,2)}~{round(close*1.15,2)}","止损":f"跌破{sup}","仓位":"20%~40%","理由":"趋势向上，RSI未超买"}
        elif score >= 50:
            advice['中线'] = {"操作":"🟡 持股观望","入场":"未持仓暂不追高","目标":f"{round(close*1.08,2)}","止损":f"跌破{sup}减仓","仓位":"维持现有","理由":"趋势中性，等待方向"}
        else:
            advice['中线'] = {"操作":"🔴 规避风险","入场":"不建议中线持有","目标":"等待底部确认","止损":f"持仓设{round(close*0.93,2)}止损","仓位":"0%~10%","理由":"中期趋势偏弱"}
    return advice

def draw_kline(df, sym, name):
    has_macd = any('MACDh' in c for c in df.columns)
    rows = 3 if has_macd else 2
    heights = [0.6, 0.2, 0.2] if has_macd else [0.7, 0.3]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=heights)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#ff4d4d', decreasing_line_color='#4da6ff', name='K线'), row=1, col=1)

    for col, color in [('EMA5','#ffd700'),('EMA10','#ff9500'),('EMA20','#ff4dff'),('EMA60','#4dffff')]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=color, width=1), name=col), row=1, col=1)

    bbu_c = [c for c in df.columns if 'BBU' in c]
    bbl_c = [c for c in df.columns if 'BBL' in c]
    if bbu_c and bbl_c:
        fig.add_trace(go.Scatter(x=df.index, y=df[bbu_c[0]], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'), name='布林上轨'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[bbl_c[0]], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'), name='布林下轨', fill='tonexty', fillcolor='rgba(255,255,255,0.02)'), row=1, col=1)

    vc = ['#ff4d4d' if df['close'].iloc[i] >= df['open'].iloc[i] else '#4da6ff' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=vc, name='成交量', opacity=0.7), row=2, col=1)
    if 'vol_ma5' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['vol_ma5'], line=dict(color='#ffd700', width=1), name='量5均'), row=2, col=1)

    if has_macd:
        mh_c = [c for c in df.columns if 'MACDh' in c]
        ml_c = [c for c in df.columns if c.startswith('MACD_')]
        ms_c = [c for c in df.columns if c.startswith('MACDs')]
        mc   = ['#ff4d4d' if v >= 0 else '#4da6ff' for v in df[mh_c[0]]]
        fig.add_trace(go.Bar(x=df.index, y=df[mh_c[0]], marker_color=mc, name='MACD柱'), row=3, col=1)
        if ml_c: fig.add_trace(go.Scatter(x=df.index, y=df[ml_c[0]], line=dict(color='#ffd700', width=1), name='MACD'), row=3, col=1)
        if ms_c: fig.add_trace(go.Scatter(x=df.index, y=df[ms_c[0]], line=dict(color='#ff9500', width=1), name='Signal'), row=3, col=1)

    title_text = f"{name}（{sym}）" if name != sym else sym
    fig.update_layout(
        title=dict(text=title_text + " 走势分析", font=dict(size=16, color='#e0e6f0')),
        template="plotly_dark", paper_bgcolor='#0d1b2a', plot_bgcolor='#0a1020',
        xaxis_rangeslider_visible=False, height=650,
        legend=dict(orientation='h', y=1.02, font=dict(size=10)),
        margin=dict(l=50, r=20, t=60, b=20)
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    return fig

# ─── 主程序 ──────────────────────────────────────────────────

if run_btn:
    if not symbols:
        st.warning("请输入至少一个股票代码")
        st.stop()

    # 多股榜单
    if len(symbols) > 1:
        st.markdown("## 📊 多股评分榜单")
        scan_results = []
        prog = st.progress(0, text="正在扫描...")
        for i, sym in enumerate(symbols):
            prog.progress((i+1)/len(symbols), text=f"正在分析 {sym}...")
            df, name = get_data(sym, period_str)
            if df is not None and len(df) >= 5:
                df = compute_indicators(df)
                r = score_stock(df)
                if r:
                    adv = generate_advice(r, invest_style)
                    scan_results.append({
                        "代码": sym,
                        "名称": (name[:12] if name and name != sym else sym),
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
        else:
            st.error("所有股票数据获取失败，请检查代码格式")

    # 逐股深度分析
    for sym in symbols:
        st.markdown("---")
        st.markdown(f"## 🔍 {sym} 深度分析报告")

        with st.spinner(f"正在获取 {sym} 数据..."):
            df, name = get_data(sym, period_str)

        if df is None or len(df) < 5:
            st.error(f"""**{sym}** 数据获取失败。
            
**可能原因：**
- A股需加后缀：上证用 `600519.SS`，深证用 `000001.SZ`
- 代码不存在或输入有误
- 网络连接超时，请稍后重试
            """)
            continue

        st.caption(f"✅ 成功获取 {len(df)} 条数据，最新价：{df['close'].iloc[-1]:.2f}")

        df = compute_indicators(df)
        result = score_stock(df)
        if not result:
            st.error(f"{sym} 数据不足，无法计算指标（至少需要5条记录）")
            continue

        result['df'] = df
        advice = generate_advice(result, invest_style)
        score  = result['score']

        # 指标卡片
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("名称", name[:10] if name and len(name) > 2 else sym)
        with c2: st.metric("当前价", f"{result['close']:.2f}", f"{result['pct_change']:+.2f}%")
        with c3:
            sc = "score-high" if score>=70 else "score-mid" if score>=50 else "score-low"
            st.markdown("**综合评分**")
            st.markdown(f"<span class='{sc}'>{score}</span>/100", unsafe_allow_html=True)
        with c4: st.metric("RSI", f"{result['rsi']:.1f}", "超买⚠️" if result['rsi']>70 else "超卖🔥" if result['rsi']<30 else "正常✅")
        with c5: st.metric("量比", f"{result['vol_ratio']:.2f}x", "放量🔴" if result['vol_ratio']>1.5 else "缩量🔵" if result['vol_ratio']<0.7 else "正常")

        # K线图
        st.plotly_chart(draw_kline(df, sym, name or sym), use_container_width=True)

        # 信号 + 建议
        col_sig, col_adv = st.columns(2)
        with col_sig:
            st.markdown("### 📡 技术信号")
            for sig_name, sig_type, sig_desc in result['signals']:
                st.markdown(f"<div class='signal-{sig_type}'><b>{sig_name}</b><br><small>{sig_desc}</small></div>", unsafe_allow_html=True)

        with col_adv:
            st.markdown("### 💡 投资建议")
            for style_name, adv in advice.items():
                st.markdown(
                    f"<div class='advice-box'><b>【{style_name}】{adv['操作']}</b><br><br>"
                    f"📌 入场：{adv['入场']}<br>"
                    f"🎯 目标：{adv['目标']}<br>"
                    f"🛡️ 止损：{adv['止损']}<br>"
                    f"💼 仓位：{adv['仓位']}<br><br>"
                    f"<small>📝 {adv['理由']}</small></div>",
                    unsafe_allow_html=True
                )

        # AI提示词
        st.markdown("### 🤖 发给AI深度分析")
        prompt = f"""你是资深操盘手，请对【{name}（{sym}）】进行深度研判：

【量化评分】{score}/100
【技术数据】RSI={result['rsi']:.1f}，量比={result['vol_ratio']:.2f}x，今日{result['pct_change']:+.2f}%
【技术信号】{' | '.join([s[0] for s in result['signals']])}

请分析：
1. 当前技术形态与关键位置
2. 短线（1-3天）具体操作策略
3. 中线（1-4周）趋势与持仓策略
4. 必须止损离场的条件
5. 一句话总结：买入/观望/回避"""
        st.text_area("复制发送给 Claude/Gemini：", value=prompt, height=200, key=f"p_{sym}")

else:
    st.markdown("""
    <div style='text-align:center; padding:60px 20px;'>
        <h2 style='color:#ffd700'>欢迎使用 AI智能量化分析平台</h2>
        <p style='color:#8899aa; font-size:16px'>点击左侧热门股票按钮，或输入代码，开始分析</p>
        <br>
        <table style='margin:auto; color:#aabbcc; border-collapse:collapse;'>
            <tr><td style='padding:12px 24px;border:1px solid #2a3f6f'>📊 K线技术</td><td style='padding:12px 24px;border:1px solid #2a3f6f'>均线·MACD·RSI·布林带</td></tr>
            <tr><td style='padding:12px 24px;border:1px solid #2a3f6f'>📈 量价分析</td><td style='padding:12px 24px;border:1px solid #2a3f6f'>成交量·量比·主力信号</td></tr>
            <tr><td style='padding:12px 24px;border:1px solid #2a3f6f'>💡 投资建议</td><td style='padding:12px 24px;border:1px solid #2a3f6f'>短线+中线双维度策略</td></tr>
            <tr><td style='padding:12px 24px;border:1px solid #2a3f6f'>🤖 AI研判</td><td style='padding:12px 24px;border:1px solid #2a3f6f'>自动生成提示词</td></tr>
        </table>
        <br>
        <p style='color:#556677; font-size:13px'>⚠️ 本工具仅供学习研究，不构成投资建议</p>
    </div>
    """, unsafe_allow_html=True)
