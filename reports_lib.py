###################################################################### Reports Library ########################################################
# Authored by Jeff Pancottine, first release February 2024
#
# Change Log
#    - April 2024 -> change df header formats to match account update formats 
#    - August 2024 -> updated for changes in statistics library functions
#
# To Do List
# 
###############################################################################################################################################



############################################################ Preliminaries - import libraries, etc. ###########################################


# import libraries

import pandas as pd
import warnings
import plots_lib
import statistics_lib
from IPython.display import display
import math

# there is some code within pyfolio, empyrical, pandas and matplotlib that has deprecated calls - this just supresses the warnings from showing
warnings.filterwarnings('ignore') 


################################################################################################################################################


########################################################  Functions - Create Report Metrics and Graphs #########################################
# 
# Level 0 (Summary):
# Initial Balance, Final Balance, CAGR, StDev, Best Year, Worst Year, Rate of Return, Max Drawdown, Sharpe Ratio, Sortino Ratio, 
# Market Correlation, Equity Curve Chart, Yearly Profit Chart, Drawdown Chart.
# 
# Level 1 (Details - Trades):
# Summary +  total trades, winning trades, losing trades, % wins, win/loss ratio, max win trade, max loss trade, max consecutive wins, 
# max consecutive losses, total commisions, total slippage, avg drawdown, avg trade profit, avg win, avg loss, profit factor, expectancy.
# 
# Level 2 (Details - System):
# Summary + Arithmetic Mean (monthly), Arithmetic Mean (annualized), Geometric Mean (monthly), Geometric Mean (annualized), 
# Standard Deviation (monthly), MAR Ratio, Standard Deviation (annualized), Downside Deviation (monthly), Beta(*), Alpha (annualized), R2, 
# Treynor Ratio (%), Calmar Ratio, Active Return, Tracking Error, Information Ratio, Skewness, Excess Kurtosis, Historical Value-at-Risk (5%), 
# Analytical Value-at-Risk (5%), Conditional Value-at-Risk (5%), Upside Capture Ratio (%), Downside Capture Ratio (%)
# 
# Level 3 (All Metrics)
# 
###############################################################################################################################################



###################### create L0 metrics #################################################
# input:  portfolio and bmark dfs, bmark ticker name and periodicity
# output: metrics DF for level 0 - summary stats
##########################################################################################

def make_L0_metrics(portfolio_df, bmark_df, bmark_name, periodicity, Rf):
    # output dataframe
    L0_metrics = pd.DataFrame(columns=['Metric', 'Model', bmark_name])                
    
    # initial balances and final balances 
    L0_metrics.loc[len(L0_metrics.index)] = ['Initial Balance', "${:,.2f}".format(statistics_lib.balances(portfolio_df.balance, 'initial')), "${:,.2f}".format(statistics_lib.balances(bmark_df.balance, 'initial'))]
    L0_metrics.loc[len(L0_metrics.index)] = ['Final Balance',"${:,.2f}".format(statistics_lib.balances(portfolio_df.balance, 'final')), "${:,.2f}".format(statistics_lib.balances(bmark_df.balance, 'final'))]
    
    # CAGR
    L0_metrics.loc[len(L0_metrics.index)] = ['CAGR', "{:.2%}".format(statistics_lib.cagr(portfolio_df.index[0], portfolio_df.index[-1], portfolio_df.balance.iloc[0], portfolio_df.balance.iloc[-1])), \
                                             "{:.2%}".format(statistics_lib.cagr(bmark_df.index[0], bmark_df.index[-1], bmark_df.balance.iloc[0], bmark_df.balance.iloc[-1]))]
    
    # annualized StDev
    if periodicity == 'M':
        L0_metrics.loc[len(L0_metrics.index)] = ['Annualized Standard Deviation', "{:.2%}".format(statistics_lib.annual_sd(portfolio_df.returns, 'M')), "{:.2%}".format(statistics_lib.annual_sd(bmark_df.returns, 'M'))]
    else: L0_metrics.loc[len(L0_metrics.index)] = ['Annualized Standard Deviation', "{:.2%}".format(statistics_lib.annual_sd(portfolio_df.returns, 'D')), "{:.2%}".format(statistics_lib.annual_sd(portfolio_df.returns, 'D'))]
    
    # cumulative return (RoR)
    L0_metrics.loc[len(L0_metrics.index)] = ['Cumulative Return', "{:.2%}".format(statistics_lib.RoR(portfolio_df.balance[0], portfolio_df.balance[-1])), "{:.2%}".format(statistics_lib.RoR(bmark_df.balance[0], bmark_df.balance[-1]))]
        
    # best and worst years
    if periodicity == "M":
        ypfreturns = portfolio_df['returns'].resample("Y").apply(lambda x: ((x + 1).cumprod() - 1).last("M"))
        ybmreturns = bmark_df['returns'].resample("Y").apply(lambda x: ((x + 1).cumprod() - 1).last("M"))
    elif periodicity == 'D':
        ypfreturns = portfolio_df['returns'].resample("Y").apply(lambda x: ((x + 1).cumprod() - 1).last("D"))
        ybmreturns = bmark_df['returns'].resample("Y").apply(lambda x: ((x + 1).cumprod() - 1).last("D"))
    L0_metrics.loc[len(L0_metrics.index)] = ['Best  Year', "{:.2%}".format(statistics_lib.best_of(ypfreturns, 'max')), "{:.2%}".format(statistics_lib.best_of(ybmreturns, 'max'))]
    L0_metrics.loc[len(L0_metrics.index)] = ['Worst Year', "{:.2%}".format(statistics_lib.worst_of(ypfreturns, 'min')), "{:.2%}".format(statistics_lib.worst_of(ybmreturns, 'min'))]
    
    # maximum drawdowns
    mDD, mMDD = statistics_lib.drawdowns_and_maxDD(portfolio_df['returns'])
    bDD, bMDD = statistics_lib.drawdowns_and_maxDD(bmark_df['returns'])
    L0_metrics.loc[len(L0_metrics.index)] = ['Maximum Drawdown', "{:.2%}".format(mMDD), "{:.2%}".format(bMDD)]
      
    # sharpe ratio
    if periodicity == 'M':
        L0_metrics.loc[len(L0_metrics.index)] = ['Sharpe Ratio', "{:.2f}".format(statistics_lib.annual_sharpe(portfolio_df['returns'], Rf, 'M')), "{:.2f}".format(statistics_lib.annual_sharpe(bmark_df['returns'], Rf, 'M'))]
    else: L0_metrics.loc[len(L0_metrics.index)] = ['Sharpe Ratio', "{:.2f}".format(statistics_lib.annual_sharpe(portfolio_df['returns'], Rf, 'D')), "{:.2f}".format(statistics_lib.annual_sharpe(bmark_df['returns'], Rf, 'D'))]
    
    # sortino ratio 
    if periodicity == 'M':
        L0_metrics.loc[len(L0_metrics.index)] = ['Sortino Ratio', "{:.2f}".format(statistics_lib.annual_sortino(portfolio_df['returns'], Rf, 'M')), "{:.2f}".format(statistics_lib.annual_sortino(bmark_df['returns'], Rf, 'M'))]
    else: L0_metrics.loc[len(L0_metrics.index)] = ['Sortino Ratio', "{:.2f}".format(statistics_lib.annual_sortino(portfolio_df['returns'], Rf, 'D')), "{:.2f}".format(statistics_lib.annual_sortino(bmark_df['returns'], Rf, 'D'))]
    
    # market correlation
    L0_metrics.loc[len(L0_metrics.index)] = ['Market Correlation', "{:.2f}".format(statistics_lib.market_correlation(portfolio_df['returns'], bmark_df['returns'])), "{:.2f}".format(statistics_lib.market_correlation(bmark_df['returns'], bmark_df['returns']))]


    return L0_metrics
    



###################### create L1 metrics #################################################
# input:  trades df
# output: metrics DF for level 1 - trade stats
##########################################################################################

def make_L1_metrics(trades_df):
    # output dataframe
    L1_metrics = pd.DataFrame(columns=['Metric', 'Model'])                
    
    # trades - total, wins, loses, pct wins, pct loses, win/loss ratio, avg win, avg loss, 
    trades = statistics_lib.number_of_trades(trades_df)
    model_wins = statistics_lib.trade_wins(trades_df['trade_ret_pct'])

    L1_metrics.loc[len(L1_metrics.index)] = ['Number of Trades', trades]
    L1_metrics.loc[len(L1_metrics.index)] = ['Wins', model_wins]
    L1_metrics.loc[len(L1_metrics.index)] = ['Percent Wins', "{:.2%}".format(statistics_lib.percent_trade_wins(trades_df['trade_ret_pct']))]
    L1_metrics.loc[len(L1_metrics.index)] = ['Win/Loss Ratio', "{:.2f}".format(statistics_lib.win_loss_ratio(trades_df['trade_ret_pct']))]
    L1_metrics.loc[len(L1_metrics.index)] = ['Average Trade Profit', "{:.2%}".format(statistics_lib.average_trade_profit(trades_df['trade_ret_pct']))]
    L1_metrics.loc[len(L1_metrics.index)] = ['Maximum Win', "{:.2%}".format(statistics_lib.maximum_trade_win(trades_df['trade_ret_pct']))]
    L1_metrics.loc[len(L1_metrics.index)] = ['Maximum Loss', "{:.2%}".format(statistics_lib.maximum_trade_loss(trades_df['trade_ret_pct']))]
 
    # max consecutive wins and losses
    L1_metrics.loc[len(L1_metrics.index)] = ['Maximum Winning Streak', statistics_lib.consecutive_trade_wins(trades_df['trade_ret_pct'])]
    L1_metrics.loc[len(L1_metrics.index)] = ['Maximum Losing Streak', statistics_lib.consecutive_trade_losses(trades_df['trade_ret_pct'])]
    
    # avg win and loss
    L1_metrics.loc[len(L1_metrics.index)] = ['Average Win', "{:.2%}".format(statistics_lib.average_trade_win(trades_df['trade_ret_pct']))]
    L1_metrics.loc[len(L1_metrics.index)] = ['Average Loss', "{:.2%}".format(statistics_lib.average_trade_loss(trades_df['trade_ret_pct']))]

    # profit factor - the amount of profit per unit of risk, with values greater than one indicating a profitable system.
    L1_metrics.loc[len(L1_metrics.index)] = ['Profit Factor', "{:.2f}".format(statistics_lib.profit_factor(trades_df['returns']))]

    # expectancy - determine the expected profit or loss of a single trade after taking into consideration all past trades and their wins and losses. 
    L1_metrics.loc[len(L1_metrics.index)] = ['Expectancy', "{:.2%}".format(statistics_lib.expectancy(trades_df['trade_ret_pct']))]
    
    
    return L1_metrics
    




###################### create L2 metrics #################################################
# input:  portfolio and bmark dfs, bmark ticker and periodicity
# output: metrics DF for level 2 - system stats
##########################################################################################

def make_L2_metrics(portfolio_df, bmark_df, bmark_name, periodicity, Rf):
    # output dataframe
    L2_metrics = pd.DataFrame(columns=['Metric', 'Model', bmark_name])                

    # arithmetic monthly and annualized mean
    L2_metrics.loc[len(L2_metrics.index)] = ['Returns Arithmetic Mean (Monthly)', "{:.2%}".format(statistics_lib.arithmetic_monthly_mean(portfolio_df['returns'], periodicity)), \
                                             "{:.2%}".format(statistics_lib.arithmetic_monthly_mean(bmark_df['returns'], periodicity))]
    L2_metrics.loc[len(L2_metrics.index)] = ['Returns Arithmetic Mean (Annualized)', "{:.2%}".format(statistics_lib.arithmetic_annual_mean(portfolio_df['returns'], periodicity)), \
                                             "{:.2%}".format(statistics_lib.arithmetic_annual_mean(bmark_df['returns'], periodicity))]

    # geometric monthly and annualized mean
    L2_metrics.loc[len(L2_metrics.index)] = ['Returns Geometric Mean (Monthly)', "{:.2%}".format(statistics_lib.geometric_monthly_mean(portfolio_df['returns'], periodicity)), \
                                             "{:.2%}".format(statistics_lib.geometric_monthly_mean(bmark_df['returns'], periodicity))]
    L2_metrics.loc[len(L2_metrics.index)] = ['Returns Geometric Mean (Annualized)', "{:.2%}".format(statistics_lib.geometric_annual_mean(portfolio_df['returns'], periodicity)), \
                                             "{:.2%}".format(statistics_lib.geometric_annual_mean(bmark_df['returns'], periodicity))]
        
    # monthly and annualized standard deviation
    L2_metrics.loc[len(L2_metrics.index)] = ['Returns Standard Deviation (Monthly)', "{:.2%}".format(statistics_lib.monthly_sd(portfolio_df['returns'], periodicity)), \
                                             "{:.2%}".format(statistics_lib.monthly_sd(bmark_df['returns'], periodicity))]
    L2_metrics.loc[len(L2_metrics.index)] = ['Returns Standard Deviation (Annualized)', "{:.2%}".format(statistics_lib.annual_sd(portfolio_df['returns'], periodicity)), \
                                             "{:.2%}".format(statistics_lib.annual_sd(bmark_df['returns'], periodicity))]

    # downside deviation - Downside deviation measures the downside volatility of the portfolio returns unlike standard deviation, which includes both upside and 
    # downside volatility. Downside volatility focuses on the negative returns that hurt the portfolio performance.
    L2_metrics.loc[len(L2_metrics.index)] = ['Downside Deviation', "{:.2%}".format(statistics_lib.downside_deviation(portfolio_df['returns'])), "{:.2%}".format(statistics_lib.downside_deviation(bmark_df['returns']))]
    
    # Beta - Beta is a measure of systematic risk and measures the volatility of a particular investment relative to the market or its benchmark as a whole.  
    L2_metrics.loc[len(L2_metrics.index)] = ['Beta', "{:.2f}".format(statistics_lib.beta(portfolio_df['returns'], bmark_df['returns'])), 'N/A']

    # Alpha - Alpha measures the active return of the investment compared to the market benchmark return. alpha = Fund Average Excess Return − Rf - (Beta × (Benchmark Average Excess Return - Rf))
    L2_metrics.loc[len(L2_metrics.index)] = ['Alpha (Annualized)', "{:.2%}".format(statistics_lib.annual_alpha(portfolio_df['returns'], bmark_df['returns'], Rf, periodicity)), 'N/A']
    
    # r squared - measures how closely each change in the price of an asset is correlated to a benchmark.
    L2_metrics.loc[len(L2_metrics.index)] = ['R Squared', "{:.2%}".format(statistics_lib.r2(portfolio_df['returns'], bmark_df['returns'])), 'N/A']
    
    # MAR ratio - The MAR ratio is a measurement of performance returns, adjusted for risk. The MAR ratio is calculated by dividing the compound annual growth rate (CAGR) 
    # of a fund or strategy since its inception by its most significant drawdown. 
    L2_metrics.loc[len(L2_metrics.index)] = ['MAR Ratio', "{:.2f}".format(statistics_lib.mar(portfolio_df['returns'], portfolio_df['balance'])), "{:.2f}".format(statistics_lib.mar(bmark_df['returns'], bmark_df['balance']))]
        
    # Treynor ratio - The Treynor ratio is a measure of risk-adjusted performance of the portfolio. It is similar to the Sharpe ratio, but it uses beta (systematic risk) 
    # as the risk metric in the denominator.
    L2_metrics.loc[len(L2_metrics.index)] = ['Treynor Ratio (Annualized)', "{:.2%}".format(statistics_lib.annual_treynor_ratio(portfolio_df['returns'], bmark_df['returns'], Rf, "portfolio", periodicity)), "{:.2%}".format(statistics_lib.annual_treynor_ratio(portfolio_df['returns'], bmark_df['returns'], Rf, "bmark", periodicity))]
        
    # calmar ratio - The Calmar ratio is a measure of risk-adjusted performance of the portfolio. It is calculated as the annualized return over the past 36 months divided 
    # by the maximum drawdown over the past 36 months.
    if (periodicity == 'M' and len(portfolio_df.index) >= 36) or (periodicity == 'D' and len(portfolio_df.index) >= 756):
        L2_metrics.loc[len(L2_metrics.index)] = ['Calmar Ratio (3-Year)', "{:.2f}".format(statistics_lib.calmar_ratio(portfolio_df.returns, portfolio_df.balance, periodicity)), \
                                                     "{:.2f}".format(statistics_lib.calmar_ratio(bmark_df.returns, bmark_df.balance, periodicity))]
    else:
         print('Calmar Ratio: Need 36 Months of Data')
        
    # active return - Active return is the investment return minus the return of its benchmark. This is displayed as annualized value, i.e., annualized investment 
    # return minus annualized benchmark return.
    L2_metrics.loc[len(L2_metrics.index)] = ['Active Return (Annualized)', "{:.2%}".format(statistics_lib.annual_active_return(portfolio_df.returns, bmark_df.returns, portfolio_df.balance, bmark_df.balance)), "N/A"]
 
    # tracking error - Tracking error is the standard deviation of active return. This is displayed as annualized value based on the standard deviation of monthly active returns.
    L2_metrics.loc[len(L2_metrics.index)] = ['Tracking Error', "{:.2%}".format(statistics_lib.annual_tracking_error(portfolio_df['returns'], bmark_df['returns'], periodicity)), "N/A"]    
    
    # information ratio - Information ratio is the active return divided by the tracking error. It measures whether the investment outperformed its benchmark consistently.
    L2_metrics.loc[len(L2_metrics.index)] = ['Information Ratio', "{:.2f}".format(statistics_lib.annual_information_ratio(portfolio_df.returns, bmark_df.returns, portfolio_df.balance, bmark_df.balance, periodicity)), "N/A"]    
    
    # skewness - Skewness is the degree of asymmetry observed in a probability distribution. Distributions can exhibit right (positive) skewness or left (negative) skewness to 
    # varying degrees. A normal distribution (bell curve) exhibits zero skewness. Investors note right-skewness when judging a return distribution because it better represents 
    # the extremes of the data set rather than focusing solely on the average. Skewness informs users of the direction of outliers, though it does not tell users the number that occurs.
    L2_metrics.loc[len(L2_metrics.index)] = ['Skewness', "{:.2f}".format(statistics_lib.skewness(portfolio_df['returns'])), "{:.2f}".format(statistics_lib.skewness(bmark_df['returns']))]
        
    # excess kurtosis - Kurtosis is used in financial analysis to measure an investment's risk of price volatility. Kurtosis measures the amount of volatility an investment's 
    # price has experienced regularly. High Kurtosis of the return distribution implies that an investment will yield occasional extreme returns.
    L2_metrics.loc[len(L2_metrics.index)] = ['Kurtosis', "{:.2f}".format(statistics_lib.kurtosis(portfolio_df['returns'])), "{:.2f}".format(statistics_lib.kurtosis(bmark_df['returns']))]
    
    # 5% analytical, historical and conditional VaR - Value at Risk (VaR) measures the scale of loss at a given confidence level. 
    # For example, if the 95% confidence one-month VaR is 3%, there is 95% confidence that over the next month the portfolio will not lose more than 3%. 
    # VaR represensts a loss, but it is conventionally reported as a positive number. 
    # Value at Risk can be calculated directly based on historical returns based on a given percentile or analytically based on the mean and standard deviation of the returns:
    # 5% VaR = E[R]−1.645×σ
    L2_metrics.loc[len(L2_metrics.index)] = ['Analytical VaR (5%)', "{:.2%}".format(statistics_lib.analytical_var(portfolio_df.returns)), "{:.2%}".format(statistics_lib.analytical_var(bmark_df.returns))]
    L2_metrics.loc[len(L2_metrics.index)] = ['Historical VaR (5%)', "{:.2%}".format(statistics_lib.historical_var(portfolio_df.returns)), "{:.2%}".format(statistics_lib.historical_var(bmark_df.returns))]
    L2_metrics.loc[len(L2_metrics.index)] = ['Conditional VaR (5%)', "{:.2%}".format(statistics_lib.conditional_var(portfolio_df.returns)), "{:.2%}".format(statistics_lib.conditional_var(bmark_df.returns))]
    
    # upside/downside capture ratios - The upside capture ratio measures how well the fund performed relative to the benchmark when the market was up, and the downside capture ratio 
    # measures how well the fund performed relative to the benchmark when the market was down. An upside capture ratio greater than 100 would indicate that the fund outperformed its 
    # benchmark when the market was up, and a downside capture ratio below 100 would indicate that the fund lost less than its benchmark when the market was down. 
    mup, mdown, mtotal = statistics_lib.capture_ratios(portfolio_df.returns, bmark_df.returns, periodicity)    
    L2_metrics.loc[len(L2_metrics.index)] = ['Upside Capture Ratio', "{:.2%}".format(mup), 'N/A']    
    L2_metrics.loc[len(L2_metrics.index)] = ['Downside Capture Ratio', "{:.2%}".format(mdown), 'N/A']
    L2_metrics.loc[len(L2_metrics.index)] = ['Capture Ratio', "{:.2%}".format(mtotal), 'N/A']
        
 
    return L2_metrics  


###################### reports output ###############################################################################################################
# input:  level of stats to output, account and trade df, benchmark df and name, start and end dates, periodicity and annual risk free rate of return
# output: level 0, 1, 2, 3 metrics
#####################################################################################################################################################


def reports_output(metric_level, portfolio_df, trades_df, bmark_df, bmark_name, start, end, period, rf):

    #################### if metric level = 0 or 3 #################################
    
    # print summary level metrics
    if (metric_level == 0 or metric_level == 3):
        L0 = make_L0_metrics(portfolio_df, bmark_df, bmark_name, period, rf)

        # create active returns for heatmap
        active_df = pd.DataFrame()
        active_df['returns'] = portfolio_df['returns'] - bmark_df['returns']
        active_df.index = portfolio_df.index
        
        print()
        print()
        print()
        print('\033[1;4m' + "Performance Summary:" + '\033[0m', f"{start:%B %d, %Y}", "to", f"{end:%B %d, %Y}")
        print()
        display(L0.style.hide(axis="index"))
        print()
        print()
        plots_lib.equity_curve_plot(portfolio_df, bmark_df, bmark_name)
        print()
        print()
        plots_lib.drawdown_curve_plot(portfolio_df, bmark_df, bmark_name)
        print()
        print()
        plots_lib.account_returns_plot(portfolio_df)
        print()
        print()
        plots_lib.account_returns_dist_plot(portfolio_df)
        print()
        print()
        plots_lib.yearly_returns_plot(portfolio_df, bmark_df, bmark_name, period)
        print()
        print()
        plots_lib.rolling_returns_plot(portfolio_df, bmark_df, bmark_name, period)
        print()
        print()
        plots_lib.monthly_returns_heatmap_plot(portfolio_df, period, 'notactive')
        print()
        print()
        plots_lib.monthly_returns_heatmap_plot(active_df, period, 'active')
        print()
        print()
        
        
    #################### if metric level = 1 or 3 #################################
    
    # print trade level metrics 
    if (metric_level == 1 or metric_level == 3):
        L1 = make_L1_metrics(trades_df)
        print()
        print()
        print()
        print('\033[1;4m' + "Trade Metrics:" + '\033[0m', f"{start:%B %d, %Y}", "to", f"{end:%B %d, %Y}")
        print()
        display(L1.style.hide(axis="index"))
        print()
        print()
        plots_lib.trade_returns_plot(trades_df)
        print()
        print()
        plots_lib.trade_returns_dist_plot(trades_df)
        
    #################### if metric level = 2 or 3 #################################

    # print system level metrics 
    if (metric_level == 2 or metric_level == 3):
        L2 = make_L2_metrics(portfolio_df, bmark_df, bmark_name, period, rf)
        print()
        print()
        print()
        print('\033[1;4m' + "System Metrics:" + '\033[0m', f"{start:%B %d, %Y}", "to", f"{end:%B %d, %Y}")
        print()
        display(L2.style.hide(axis="index"))
        print()
        print()
        print('Model Top 5 Drawdowns')
        print()
        plots_lib.calculate_top5_drawdowns(portfolio_df)        
        print()
        print()
        print(bmark_name, ' Top 5 Drawdowns')
        print()
        plots_lib.calculate_top5_drawdowns(bmark_df)        
        print()
        print()

    return
    

###################### MC reports output ###############################################################################################################
# input:  level of stats to output, trade df, start and end dates, number of runs(simulations), max DD allowed for ruin, and annual risk free rate of return
# output: MC stats, distributions and plots
#####################################################################################################################################################

def MC_reports_output(trades, start, end, runs, init_bal, max_dd_allowed, Rf, period):

    # run monte carlo simulation
    equity_df, drawdown_df, all_drawdown_df, CAGR_df, Ratios_df, WL_df, metrics, ci_score = statistics_lib.monte_carlo(trades, start, end, runs, init_bal, max_dd_allowed, Rf, period)
    
    # metrics table
    test_metrics = pd.DataFrame(columns=['Metric', 'Value', ''])
    test_metrics.loc[len(test_metrics.index)] = ['Initial Balance', "${:,.2f}".format(init_bal), '']
    test_metrics.loc[len(test_metrics.index)] = ['# Simulations', "{:,}".format(runs), '']
    test_metrics.loc[len(test_metrics.index)] = ['# Trades/Simulation', "{:,}".format(len(trades)), '']
    test_metrics.loc[len(test_metrics.index)] = ['Risk of Ruin', "{:.2%}".format(metrics.get('ruin')), '']
    test_metrics.loc[len(test_metrics.index)] = ['Risk of Ruin DD Value', "{:.2%}".format(metrics.get('DD_allowed')), '']
    print()
    print()
    print('\033[1;4m' + "Monte Carlo Test Metrics:" + '\033[0m', f"{start:%B %d, %Y}", "to", f"{end:%B %d, %Y}")
    print()
    display(test_metrics.style.hide(axis="index"))
    print()

    # plots
    print()
    print()
    plots_lib.MC_equity_bal_dist_plot(equity_df) 
    print()
    print()
    plots_lib.MC_DD_dist_plot(all_drawdown_df)
    print()
    print()
    # plots_lib.MC_maxDD_dist_plot(drawdown_df)
    # print()
    # print()
    plots_lib.MC_profit_dist_plot(equity_df)
    print()
    print()
    plots_lib.MC_returns_dist_plot(equity_df)
    print()
    print()
    plots_lib.MC_CAGR_dist_plot(CAGR_df)
    print()
    print()
    plots_lib.MC_MAR_dist_plot(Ratios_df)
    print()
    print()
    plots_lib.MC_sharpe_dist_plot(Ratios_df)
    print()
    print()
    plots_lib.MC_Wins_dist_plot(WL_df)
    print()
    print()
    plots_lib.MC_Losses_dist_plot(WL_df)
    print()
    print()

    # confidence intervals
    conf_intervals = pd.DataFrame(columns=['Metric', 'Lower', 'Upper'])
    conf_intervals.loc[len(conf_intervals.index)] = ['Equity', "${:,.2f}".format(metrics.get('equity_low')), "${:,.2f}".format(metrics.get('equity_high'))]
#    conf_intervals.loc[len(conf_intervals.index)] = ['Max-DD', "{:,.2%}".format(metrics.get('maxdd_low')), "{:,.2%}".format(metrics.get('maxdd_high'))]
    conf_intervals.loc[len(conf_intervals.index)] = ['All-DD', "{:,.2%}".format(metrics.get('all_dd_low')), "{:,.2%}".format(metrics.get('all_dd_high'))]
    conf_intervals.loc[len(conf_intervals.index)] = ['Profit', "${:,.2f}".format(metrics.get('profit_low')), "${:,.2f}".format(metrics.get('profit_high'))]
    conf_intervals.loc[len(conf_intervals.index)] = ['Total Return', "{:,.2%}".format(metrics.get('return_low')), "{:,.2%}".format(metrics.get('return_high'))]
    conf_intervals.loc[len(conf_intervals.index)] = ['CAGR', "{:,.2%}".format(metrics.get('CAGR_low')), "{:,.2%}".format(metrics.get('CAGR_high'))]
    conf_intervals.loc[len(conf_intervals.index)] = ['MAR', "{:,.2f}".format(metrics.get('MAR_low')), "{:,.2f}".format(metrics.get('MAR_high'))]
    conf_intervals.loc[len(conf_intervals.index)] = ['Sharpe', "{:,.2f}".format(metrics.get('sharpe_low')), "{:,.2f}".format(metrics.get('sharpe_high'))]
    conf_intervals.loc[len(conf_intervals.index)] = ['Consecutive Wins', "{:,.2f}".format(metrics.get('wins_low')), "{:,.2f}".format(metrics.get('wins_high'))]
    conf_intervals.loc[len(conf_intervals.index)] = ['Consecutive Losses', "{:,.2f}".format(metrics.get('losses_low')), "{:,.2f}".format(metrics.get('losses_high'))]
    print()
    print()
    print('\033[1;4m' + "Monte Carlo Confidence Intervals (" + str(ci_score * 100) + "%):" + '\033[0m', f"{start:%B %d, %Y}", "to", f"{end:%B %d, %Y}")
    print()
    display(conf_intervals.style.hide(axis="index"))

    return






