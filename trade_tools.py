import streamlit as st

# configure the page
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded")


MC_intro = st.Page("MC/MC_intro.py", title="Introduction", icon="😊", default=True)
MC_trades_analysis = st.Page("MC/MC_trades_analysis.py", title="Trades Analysis", icon="💸")
# MC_robustness_testing = st.Page("MC/MC_robustness_testing.py", title="Robustness Testing", icon="💪")
# MC_cone_analysis = st.Page("MC/MC_cone_analysis.py", title="System Monitor", icon="👀")


introduction = st.Page("statistics/introduction.py", title="Introduction", icon="📊")
# summary = st.Page("statistics/summary.py", title="Summary Stats", icon="✍️")
# trades = st.Page("statistics/trades.py", title="Trade Stats", icon="💸")
# system = st.Page("statistics/system.py", title="System Stats", icon="⚙️")
# all_stats = st.Page("statistics/all_stats.py", title="All Stats", icon="🌐")

pg = st.navigation(
    {
        "Monte Carlo Tools": [MC_intro, MC_trades_analysis],
        "Statistics": [introduction]
    }
)

pg.run()