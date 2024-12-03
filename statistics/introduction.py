
# to run the app: enter the command at your terminal window in the same directory as the script you created. Multi-page requires a pages directory with the additional pages.
#     cd Dropbox/Python/Finance
#     streamlit run Monte_Carlo_Tools.py

# import the necessary libraries.
import streamlit as st



# # configure the page
# st.set_page_config(
#     page_title="Monte Carlo Analysis",
#     page_icon="🎰",
#     layout="wide",
#     initial_sidebar_state="expanded")

# set the title of the app, add a brief description, and set up the select boxes.
st.header("Statistical Analysis Toolset For Algorithmic Trading", help='Author: Jeff Pancottine, May 2024.', divider = 'rainbow')
st.write('An application to run statistical analysis tools on account and trading outcomes for quantitative decision making.')
st.markdown("""**👈 Select a tool from the sidebar** """)

# st.markdown(
    """
    ### Levels and types of statistical analysis.
    ** Level 0 (Summary):
    Initial Balance, Final Balance, CAGR, StDev, Best Year, Worst Year, Rate of Return, Max Drawdown, Sharpe Ratio, Sortino Ratio, 
    Market Correlation, Equity Curve Chart, Yearly Profit Chart, Drawdown Chart.
    
    ** Level 1 (Details - Trades):
    Summary +  total trades, winning trades, losing trades, % wins, win/loss ratio, max win trade, max loss trade, max consecutive wins, 
    max consecutive losses, total commisions, total slippage, avg drawdown, avg trade profit, avg win, avg loss, profit factor, expectancy.
    
    ** Level 2 (Details - System):
    Summary + Arithmetic Mean (monthly), Arithmetic Mean (annualized), Geometric Mean (monthly), Geometric Mean (annualized), 
    Standard Deviation (monthly), MAR Ratio, Standard Deviation (annualized), Downside Deviation (monthly), Beta(*), Alpha (annualized), R2, 
    Treynor Ratio (%), Calmar Ratio, Active Return, Tracking Error, Information Ratio, Skewness, Excess Kurtosis, Historical Value-at-Risk (5%), 
    Analytical Value-at-Risk (5%), Conditional Value-at-Risk (5%), Upside Capture Ratio (%), Downside Capture Ratio (%)
    
    Level 3 (All Metrics)
"""
)


