# import the necessary libraries.
import streamlit as st



# set the title of the app, add a brief description, and set up the select boxes.
st.header("Algorithmic Trading Tools", help='Author: Jeff Pancottine, December 2024.', divider = 'rainbow')
st.write('A set of applications to analyze trading outcomes for quantitative decision making.')
st.markdown("""**👈 Select a tool from the sidebar** """)

st.markdown(
    """
    ### Monte Carlo Introduction
    Monte Carlo methods are a broad class of computational algorithms that rely on repeated random sampling to obtain numerical results. Their essential idea is using randomness to solve problems that might be
    deterministic in principle. Monte Carlo methods are mainly used in three distinct problem classes: optimization, numerical integration, and generating draws from a probability distribution.

    Monte Carlo methods were first introduced to finance in 1964 by David B. Hertz through his Harvard Business Review article, discussing their application in Corporate Finance. In 1977, Phelim Boyle pioneered the use of simulation in
    derivative valuation in his seminal Journal of Financial Economics paper.  They are used in finance and mathematical finance to value and analyze (complex) instruments, portfolios and investments by simulating the various sources of
    uncertainty affecting their value, and then determining the distribution of their value over the range of resultant outcomes. This is usually done by help of stochastic asset models. The advantage of Monte Carlo methods over other
    techniques increases as the dimensions (sources of uncertainty) of the problem increase.

    Trading, whether discretionary or systematic, results in wins and losses.  These wins and losses are the outcome of decisions based on the market at a point in time and represent a distribution.  To assert that the distribution has merit it
    must be built on rules that are followed with discipline.  This enables statistical analysis to be used to evaluate performance of the system over time.  If the results are made up of relatively random attempts to create wins then the data
    is unreliable and cannot be measured statistically.

    Assuming the results are the outcome of repeated use of a disciplined approach, such as algorithmic trading, we can assume they are repeatable but we cannot assume that they will happen in the same sequence or order.  This is true because
    we do not know what conditions exist in the market at the time and they will be different for any other previous sequence of trades.  What we do know is that the probability of the distribution of the trade sequence occurring in the same
    way again is near zero.

    To understand the likely outcomes of a trading strategy over time we can simulate the possible random sequences that the trades would occur in and the outcome of those trades using Monte Carlo methods.  Through the Central Limit Theorem of
    Statistics, if we use many random trade sequences, say over 100, we can create a sample distribution of the outcomes and apply statistical analysis to the distribution. This Monte Carlo approach enables a mean and variance to be computed 
    for many different important performance measurements such as: profit, maximum drawdown, equity, returns, Sharpe ratio, etc. As an example, the statistics of the maximum drawdown enables you to assess your risk of ruin for a given initial
    capitalization of your account. 

    **The main uses for Monte Carlo methods in trading simulation are:**
    - Better understanding of possible trading drawdown and future outcomes
    - The proper funding of a trading strategy
    - Understanding possible win and loss streak potential
    - Setting better profit and loss expectations
    - Testing the robustness of a trading strategy
    - Testing if a trading strategy is broken


    ##### Want to learn more?
    - https://www.investopedia.com/terms/m/montecarlosimulation.asp
    - https://en.wikipedia.org/wiki/Monte_Carlo_method
    - https://towardsdatascience.com/monte-carlo-simulation-a-practical-guide-85da45597f0e
    - https://www.sciencedirect.com/science/article/abs/pii/0304405X77900058#preview-section-abstract
    - https://hackernoon.com/u/buildalpha

"""
)

st.markdown(
"""
    ### Levels and types of statistical analysis.
    
    - **Level 0 (Summary):**
    Initial Balance, Final Balance, CAGR, StDev, Best Year, Worst Year, Rate of Return, Max Drawdown, Sharpe Ratio, Sortino Ratio, 
    Market Correlation, Equity Curve Chart, Yearly Profit Chart, Drawdown Chart.
    
    - **Level 1 (Details - Trades):**
    Summary +  total trades, winning trades, losing trades, % wins, win/loss ratio, max win trade, max loss trade, max consecutive wins, 
    max consecutive losses, total commisions, total slippage, avg drawdown, avg trade profit, avg win, avg loss, profit factor, expectancy.
    
    - **Level 2 (Details - System):**
    Summary + Arithmetic Mean (monthly), Arithmetic Mean (annualized), Geometric Mean (monthly), Geometric Mean (annualized), 
    Standard Deviation (monthly), MAR Ratio, Standard Deviation (annualized), Downside Deviation (monthly), Beta(*), Alpha (annualized), R2, 
    Treynor Ratio (%), Calmar Ratio, Active Return, Tracking Error, Information Ratio, Skewness, Excess Kurtosis, Historical Value-at-Risk (5%), 
    Analytical Value-at-Risk (5%), Conditional Value-at-Risk (5%), Upside Capture Ratio (%), Downside Capture Ratio (%)
    
    - **Level 3 (All Metrics)**
"""
)

