import streamlit as st

# configure the page
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded")


MC_trades_analysis = st.Page("algo_tools/MC_trades_analysis.py", title="Monte Carlo Analysis")
# MC_robustness_testing = st.Page("MC/MC_robustness_testing.py", title="Robustness Testing", icon="✔️")
# MC_cone_analysis = st.Page("MC/MC_cone_analysis.py", title="System Monitor", icon="✔️")

tools = st.Page("algo_tools/welcome.py", title="Welcome", default=True)
stats = st.Page("algo_tools/stats.py", title="Statistical Analysis")

pg = st.navigation(
    {
        "Algorithmic Trading Tools": [tools, MC_trades_analysis, stats]
    }
)

pg.run()