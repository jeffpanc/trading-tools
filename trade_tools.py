import streamlit as st

# configure the page
st.set_page_config(
    # page_title="Toolset For Algorithmic Trading",
    # page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded")

# set the title of the app, add a brief description, and set up the select boxes.
# st.header("🎰  Toolset For Algorithmic Trading", help='Author: Jeff Pancottine, November 2024.', divider = 'rainbow')
# st.write('Applications for quantitative decision making.')
# st.markdown("""**👈 Select a tool from the sidebar** """)

# st.markdown(
#     """
#     ### Introduction
#     This set of tools is useful for analyzing trading and testing outcomes.  It can be used on trading account or backtest data. 
#     It currently provides Monte Carlo testing of trading outcomes and account sizing.  Also, a set of trading statistics.
#     """
# )


MC_intro = st.Page("MC/MC_intro.py", title="Introduction", icon="✔️", default=True)
MC_trades_analysis = st.Page("MC/MC_trades_analysis.py", title="Trades Analysis", icon="✔️")
# MC_robustness_testing = st.Page("MC/MC_robustness_testing.py", title="Robustness Testing", icon="✔️")
# MC_cone_analysis = st.Page("MC/MC_cone_analysis.py", title="System Monitor", icon="✔️")


introduction = st.Page("statistics/introduction.py", title="Introduction", icon="✔️")
# summary = st.Page("statistics/summary.py", title="Summary Stats", icon="✔️")
# trades = st.Page("statistics/trades.py", title="Trade Stats", icon="✔️")
# system = st.Page("statistics/system.py", title="System Stats", icon="✔️")
# all_stats = st.Page("statistics/all_stats.py", title="All Stats", icon="✔️")

pg = st.navigation(
    {
        "Monte Carlo Tools": [MC_intro, MC_trades_analysis],
        "Statistics": [introduction]
    }
)

pg.run()