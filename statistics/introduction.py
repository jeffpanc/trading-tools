
# import the necessary libraries.
import streamlit as st


# set the title of the app, add a brief description, and set up the select boxes.
st.header("Statistical Analysis Toolkit For Algorithmic Trading", help='Author: Jeff Pancottine, May 2024.', divider = 'rainbow')
st.write('An application to run statistical analysis on trading and account outcomes for quantitative decision making.')
st.markdown("""**👈 Select a tool from the sidebar** """)

st.markdown(
    """
    ### Levels of Analysis
    * Level 0: Summary Statistics
        - Initial and Final Equity
        - CAGR
        - Annual Volatility
        - Cumulative Returns
        - Best and Worst Year
        - Max Drawdown
        - Annual Sharpe Ratio
        - Annual Sortino Ratio
        - Market Correlation
        - Equity Curve
        - Drawdown Curve
        - Account Returns
        - Yearly Returns
        - Monthly Returns Heatmap
        - Monthly Active Returns Heatmap
    * Level 1: Trade Statistics
        Number of Trades
        Wins
        Percent Wins
        Win/Loss Ratio
        Average Trade Profit
        Maximum Win and Loss
        Maximum Win an Loss Streaks
        Profit Factor
        Expectancy
        Trade Returns
    * Level 2: System Statistics
        Month and Year Returns Arithmetic Mean
        Month and Year Returns Geometric Mean
        Monthly and Yearly Volatility
        Downside Deveation
        Beta
        Annual Alpha
        R-Squared
        MAR Ratio
        Treynor Ratio
        3-Year Calmar Ratio
        Yearly Active Returns
        Tracking Error
        Information Ratio
        Skewness
        Kurtosis
        Analytical VaR
        Historical VaR
        Conditional VaR
        Upside Capture Ratio
        Downside Capture Ratio
        Total Capture Ratio
        Top 5 Drawdowns Portfolio and Benchmark
    * Level 3: All Statistics
    
"""
)


