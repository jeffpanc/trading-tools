########################################################## Statistics Library ###################################
#
# Authored by Jeff Pancottine, first release February 2024
#
# Change Log
#   - May '24: added historical and conditional VaR
#   - July '24: added support for calculating consecutive wins/losses in monte carlo simulation
#   - August '24: adjusted some calculations
#
# To Do List
#   - Add support for stress testing in monte carlo simulations
#   - Ensure all functions can handle day or month data
#   - add other stats and ratios to MC
#   - check MC calcs for sharpe, mar, ruin and DD

###################################################################################################################


########################################## Preliminaries -  import libraries, etc. ################################

import pandas as pd
import numpy as np
import warnings
import math
import scipy
from dateutil.relativedelta import relativedelta
from datetime import timedelta, datetime
import numba as nb # v0.56.4, no support for numpy >= 1.22.0
from scipy.stats import norm

# the following supresses all warnings - mainly because there is some code within pyfolio, empyrical, pandas and matplotlib that has deprecated calls 
# - this just supresses the warnings from showing
warnings.filterwarnings('ignore')

##################################################################################################################


########################################################### Create stats functions ###################################################################################
 
# Level 1 (Summary):
# Initial Balance, Final Balance, CAGR, StDev, Best Year, Worst Year, Rate of Return, Max Drawdown, Sharpe Ratio, Sortino Ratio, Market Correlation.
 
# Level 2 (Details - Trades):
# Total trades, winning trades, losing trades, % wins, win/loss ratio, max win trade, max loss trade, max consecutive wins, max consecutive losses, total commisions, total slippage, avg drawdown, avg trade profit, avg win, avg loss, profit factor, expectancy.

# Level 3 (Details - System):
# Arithmetic Mean (monthly), Arithmetic Mean (annualized), Geometric Mean (monthly), Geometric Mean (annualized), Standard Deviation (monthly), MAR Ratio,
# Standard Deviation (annualized), Downside Deviation (monthly), Beta, Alpha (annualized), R2, Treynor Ratio (%), Calmar Ratio, Active Return, Tracking Error, Information Ratio, Skewness, Excess Kurtosis, Historical Value-at-Risk (5%), Analytical Value-at-Risk (5%), Conditional Value-at-Risk (5%), Upside Capture Ratio (%), Downside Capture Ratio (%)

###################################################################################################################################################################### 



######################################### annualized sharpe ratio ##########################################################
# description: finds the annualized sharpe ratio of a series of percent returns in decimal
# input:       series of returns in decimal, annual risk free rate of return in decimal, period = 'D' for day, 'M' for month
# output:      annualized sharpe ratio in decimal or 0 if period not 'D' or 'M'
############################################################################################################################
    
def annual_sharpe(returns, rf, period):
    
    if period == 'D':
        excess_returns = returns.mean() * 252 - rf                               # excess returns average 
        std_excess_returns = returns.std() * math.sqrt(252)                      # standard deviation of excess returns
    elif period == 'M':
        excess_returns = returns.mean() * 12 - rf                                # excess returns average 
        std_excess_returns = returns.std()  * math.sqrt(12)                      # standard deviation of excess returns
    else:
        return 0                                                                 # return 0 if not day or month data
            
    
    return (excess_returns / std_excess_returns)                                 # annualized sharpe

######################################### annualized sortino ratio ######################################################
# description: finds the annualized sortino ratio of a series of percent returns in decimal
# input:       series of returns in decimal, annual risk free rate of return in decimal, period = 'D' for day, 'M' for month
# output:      annualized sortino ratio in decimal or 0 if period not 'D' or 'M'
########################################################################################################################
    
def annual_sortino(returns, rf, period):

    if period == 'D':
        excess_returns = returns.mean() * 252 - rf                               # excess returns average 
        downside_dev = downside_deviation(returns) * math.sqrt(252)              # standard deviation of excess returns
    elif period == 'M':
        excess_returns = returns.mean() * 12 - rf                                # excess returns average 
        downside_dev = downside_deviation(returns)  * math.sqrt(12)              # standard deviation of excess returns
    else:
        return 0                                                                 # return 0 if not day or month data
    
    return (excess_returns / downside_dev)                                       # annualized sortiono 


######################################### annualized standard deviation ###########################################
# description: finds the annualized standard deviation for a set of percent returns in decimal
# input:       series of returns in decimal format, period: 'D' = Day, 'M' = Month
# output:      annualized standard deviation in decimal format, if period not 'D' or 'M' return 0
###################################################################################################################
    
def annual_sd(returns, period):
            
    if period == 'D':
        return returns.std() * math.sqrt(252)
    elif period == 'M':
        return returns.std() * math.sqrt(12)
    else:
        return 0
    

######################################### balances #####################################################################
# description: finds the initial or final balance in a series that contains account balances
# input:       the account balance series
# output:      the initial or final balance, otherwise 0 if initial_or_final parameter is not 'initial' or 'final'
########################################################################################################################

def balances(bal_series, initial_or_final):
    
    if initial_or_final == 'initial':
        return bal_series.iloc[0]
    elif initial_or_final == 'final':
        return bal_series.iloc[-1]
    else:
        return 0


######################################### best of... ###########################################
# description: finds the best of a series of numbers
# input:       series, max_or_min = use 'max' or 'min' as best
# output:      best number based on inputs, 0 if max_or_min is not 'max' or 'min'
###################################################################################################################
    
def best_of(series, max_or_min):

    if (max_or_min == 'max'):
        return series.max()
    elif max_or_min == 'min':
        return series.min()
    else:
        return 0


######################################### cagr #####################################################################
# description: finds the cagr for a period of years
# input:       start and end dates of period in datetime format, start and end balances of period
# output:      the cagr in decimal format, 0 if < 1 year
####################################################################################################################
    
def cagr(start_date, end_date, start_bal, end_bal):
    days = (end_date - start_date).days                                        # days between start and end date
    years = (days / 30 / 12)                                                   # years for exp calculation, can be partial year
    cagr_exp = 1 / years                                                       # exponential value to calculate cagr

    if(years >= 1):
        return  ((end_bal / start_bal) ** (cagr_exp) - 1)                      # cagr = ((end balance - start balance) ** (cagr exponential)) - 1
    else:
        return 0
    

######################################### downside deviation #########################################################
# description: downside deviation measures the downside volatility of the portfolio returns unlike standard deviation
#              which includes both upside and downside volatility. downside volatility focuses on the negative returns 
#              that hurt the portfolio performance.
# input:       series of returns in decimal
# output:      downside deviation in decimal
######################################################################################################################
    
def downside_deviation(returns):

    downside = pd.Series(index=returns.index)                                          # create a series to hold downside returns
    downside[:] = np.minimum(0, returns)                                               # find the downside returns only, 0 for others
    downside_squared = downside[:] ** 2                                                # square the downside returns
    downside_squared_sum = downside_squared.sum()                                      # sum the downside squared returns    
    
    return math.sqrt(downside_squared_sum / len(returns))                              # downside deviation = sqrt(sum of the squared returns / number of returns)


###################### drawdowns and maxDD ############################################################
# description: calculate the drawdowns of a series of percent returns in decimal
# input:       a series with percent returns in decimal
# output:      a series with percent drawdowns in decimal and the maximum drawdown in decimal
#######################################################################################################

def drawdowns_and_maxDD(returns):
    
    r = returns.add(1).cumprod()           # calculate the cumulative returns
    dd = r.div(r.cummax()).sub(1)          # find each period drawdown by calculating the (max return / cumulative return) - 1
    mdd = dd.min()                         # get the maximum drawdown
    
    return dd, mdd


######################################### market correlation ######################################################
# description: finds the pearson market correlation to a benchmark of a series of percent returns in decimal
# input:       series of percent returns in decimal, series of percent returns of a benchmark in decimal
# output:      correlation coefficent in decimal
###################################################################################################################
    
def market_correlation(returns, bmark_returns):

    return returns.corr(bmark_returns, method = 'pearson')


######################################### rate of return (RoR) ###########################################
# description: finds the rate of return for an intial and final balance
# input:       start and end balances
# output:      rate of return in decimal format
###################################################################################################################
    
def RoR(start_bal, end_bal):
          
    return (end_bal - start_bal) / start_bal


######################################### worst of... ###########################################
# description: finds the worst of a series of numbers
# input:       series, max_or_min = use 'max' or 'min' as worst
# output:      worst number based on inputs, 0 if max_or_min is not 'max' or 'min'
###################################################################################################################
    
def worst_of(series, max_or_min):

    if max_or_min == 'max':
        return series.max()
    elif max_or_min == 'min':
        return series.min()
    else:
        return 0


######################################### average trade loss ###########################################
# description: calculate average trade loss
# input:       series of trades percent returns in decimal
# output:      average trade loss in decimal
########################################################################################################
    
def average_trade_loss(trades):
    
    return  trades.where(trades < 0).sum() / trade_losses(trades)


######################################### average trade profit ###########################################
# description: calculate average trade profit
# input:       series of trades percent returns in decimal
# output:      average trade profit in decimal
##########################################################################################################
    
def average_trade_profit(trades):

    return trades.mean()
    

######################################### average trade win ###########################################
# description: calculate average trade win
# input:       series of trades percent returns in decimal
# output:      average trade win in decimal
######################################################################################################
    
def average_trade_win(trades):
    
    return  trades.where(trades > 0).sum() / trade_wins(trades)
    

######################################### expectancy #########################################################################################################
# description: expectancy is the expected profit or loss of a single trade after taking into consideration all past trades and their wins and losses.
# input:       series of trades percent returns in decimal
# output:      expectancy in decimal
##############################################################################################################################################################
    
def expectancy(trades):
     
    return (average_trade_win(trades) * (trade_wins(trades) / number_of_trades(trades))) - (average_trade_loss(trades) * (trade_losses(trades) / number_of_trades(trades))) 


######################################### maximum consecutive trade losses ###########################################
# description: calculate maximum consecutive trade losses
# input:       series of trades percent returns in decimal
# output:      maximum trade losses 
######################################################################################################################
    
def consecutive_trade_losses(trades):
    
    ml = trades.ge(0)
    
    return ml.cumsum()[~ml].value_counts().iat[0]
    

######################################### maximum consecutive trade wins ###########################################
# description: calculate maximum consecutive trade wins
# input:       series of trades percent returns in decimal
# output:      maximum trade wins 
####################################################################################################################
    
def consecutive_trade_wins(trades):
    
    mw = trades.le(0)
    
    return mw.cumsum()[~mw].value_counts().iat[0]
    

######################################### maximum trade loss ###########################################
# description: calculate maximum trade loss
# input:       series of trades percent returns in decimal
# output:      maximum trade loss  in decimal
######################################################################################################
    
def maximum_trade_loss(trades):

    return trades.min()
    

######################################### maximum trade win ###########################################
# description: calculate maximum trade win
# input:       series of trades percent returns in decimal
# output:      maximum trade win  in decimal
######################################################################################################
    
def maximum_trade_win(trades):

    return trades.max()


######################################### number of trades ###########################################
# description: calculate how many trades occured
# input:       series of trades percent returns in decimal
# output:      # trades 
######################################################################################################
    
def number_of_trades(trades):

    return len(trades)
    

######################################### number of losses ###########################################
# description: calculate how many losing trades occured
# input:       series of trades percent returns in decimal
# output:      # losing trades 
######################################################################################################
    
def trade_losses(trades):

    return len(trades[(trades < 0)])
    

######################################### number of wins ###########################################
# description: calculate how many winning trades 
# input:       series of trades percent returns in decimal
# output:      # winning trades 
####################################################################################################
    
def trade_wins(trades):

    return len(trades[(trades > 0)])
    

######################################### percent trade wins ###########################################
# description: calculate percent trade wins
# input:       series of trades percent returns in decimal
# output:      % trades wins in decimal
########################################################################################################
    
def percent_trade_wins(trades):

    return trade_wins(trades) / number_of_trades(trades)
    

######################################### profit factor ###########################################################################
# description: profit factor is the amount of profit per unit of risk, with values greater than one indicating a profitable system.
# input:       series of trades dollar returns 
# output:      profit factor in decimal
###################################################################################################################################
    
def profit_factor(trades):
    
    gross_profit = trades[(trades > 0)].sum()
    gross_loss = abs(trades[(trades < 0)].sum())
    
    return gross_profit / gross_loss


######################################### win/loss trade ratio ###########################################
# description: calculate win/loss ratio
# input:       series of trades percent returns in decimal
# output:      % win/loss ratio in decimal
##########################################################################################################
    
def win_loss_ratio(trades):

    return trade_wins(trades) / trade_losses(trades)
    


######################################### arithmetic monthly mean return ###########################################
# description: calculate arithmetic monthly mean return
# input:       series of account percent returns in decimal, period 'D' or 'M'
# output:      arithmetic monthly mean return in decimal or 0 if period not 'D' or 'M'
###################################################################################################################
    
def arithmetic_monthly_mean(returns, period):

    if period == 'D':
        mreturns = returns.resample("M").apply(lambda x: ((x + 1).cumprod() - 1).last("D"))
        return mreturns.mean()
    elif period == 'M':
        return returns.mean()
    else:
        return 0
 

######################################### arithmetic annual mean return ###########################################
# description: calculate arithmetic annual mean return
# input:       series of account percent returns in decimal, period 'D' or 'M'
# output:      arithmetic annual mean return in decimal or 0 if period not 'D' or 'M'
###################################################################################################################
    
def arithmetic_annual_mean(returns, period):
    
    if period == 'D':
        mreturns = returns.resample("M").apply(lambda x: ((x + 1).cumprod() - 1).last("D"))
    elif period == 'M':
        mreturns = returns
    else:
        return 0
        
    mreturns_mean = mreturns.mean()
    
    return (mreturns_mean + 1) ** 12 - 1


######################################### geometric monthly mean return ###########################################
# description: calculate geometric monthly mean return
# input:       series of account percent returns in decimal, period 'D' or 'M'
# output:      geometric monthly mean return in decimal or 0 if period not 'D' or 'M'
###################################################################################################################
    
def geometric_monthly_mean(returns, period):

    if period == 'D':
        mreturns = returns.resample("M").apply(lambda x: ((x + 1).cumprod() - 1).last("D"))
    elif period == 'M':
        mreturns = returns
    else:
        return 0
        
    monthly_data = mreturns + 1
    product = math.prod(monthly_data)
    
    return pow(product, 1/len(returns)) - 1
    

######################################### geometric annual mean return ###########################################
# description: calculate geometric annual mean return
# input:       series of account percent returns in decimal, period 'D' or 'M'
# output:      geometric annual mean return in decimal or 0 if period not 'D' or 'M'
#################################################################################################################
    
def geometric_annual_mean(returns, period):

    if period == 'D':
        mreturns = returns.resample("M").apply(lambda x: ((x + 1).cumprod() - 1).last("D"))
    elif period == 'M':
        mreturns = returns
    else:
        return 0
        
    monthly_data = mreturns + 1
    product = math.prod(monthly_data)
    
    return pow(product, 12/len(returns)) - 1
   

######################################### monthly standard deviation ###############################################
# description: finds the monthly standard deviation for a set of percent returns
# input:       series of returns in decimal format, period: 'D' = Day, 'M' = Month
# output:      monthly standard deviation in decimal format, if period not 'D' or 'M' return 0
###################################################################################################################
    
def monthly_sd(returns, period):
        
    if period == 'D':
        return returns.std() * math.sqrt(21)
    elif period == 'M':
        return returns.std()
    else:
        return 0
    

######################################### beta ###############################################################################################################
# description: beta is a measure of systematic risk and measures the volatility of a particular investment relative to the market or its benchmark as a whole.
# input:       series of model returns in decimal format, benchmark returns in decimal format
# output:      beta in decimal format
##############################################################################################################################################################
    
def beta(model_returns, bmark_returns):
        
    covariance_model = model_returns.cov(bmark_returns)
    variance_model = bmark_returns.var()

    return (covariance_model / variance_model)
    

######################################### annualized alpha ##############################################################################################################################################
# description: alpha measures the active return of the investment compared to the market benchmark return. alpha = (Fund Return.mean() - Rf) − (Beta × (Benchmark Average Excess Return.mean() - Rf)
# input:       model returns in decimal format, benchmark returns in decimal format, risk free return in decimal and "D" or "M" time period
# output:      alpha in decimal format
##############################################################################################################################################################################################
    
def annual_alpha(returns, bmark_returns, rf, period):

    al_beta = beta(returns, bmark_returns)
    if period == 'D':
        return ((returns.mean() - rf) - (al_beta * (bmark_returns.mean() - rf))) * 252    
    elif period == 'M':
        return ((returns.mean() - rf) - (al_beta * (bmark_returns.mean() - rf))) * 12
    else:
        return 0    
        

######################################### r squared ##################################################################################
# description: r squared measures how closely each change in the price of an asset is correlated to a benchmark. using Pearson method.
#              measured between 0 (no correlation) and 100 (matches benchmark correlation). r2% of returns are determined by the 
#              returns of its benchmark.
# input:       model returns in decimal format, benchmark returns in decimal format
# output:      r2 in decimal format
######################################################################################################################################
    
def r2(model_returns, bmark_returns):
        
    return (model_returns.corr(bmark_returns, method='pearson')) ** 2
    

######################################### MAR ratio ###############################################################################################################
# description: MAR ratio is a measurement of performance returns, adjusted for risk. The MAR ratio is calculated by dividing the compound annual growth rate (CAGR) 
#              of a fund or strategy since its inception by its most significant drawdown.
# input:       returns and balances in decimal format
# output:      MAR ratio in decimal format
###################################################################################################################################################################
    
def mar(returns, balance_series):
    
    DD, Max_DD = drawdowns_and_maxDD(returns)
    returns_cagr = cagr(returns.index[0], returns.index[-1], balance_series.iloc[0], balance_series.iloc[-1])

    if Max_DD == 0:
        return abs(returns_cagr / 1)         # use abs() since MDD is a negative number
    else:
        return abs(returns_cagr / Max_DD)    # use abs() since MDD is a negative number
        


######################################### Annualized Treynor ratio ######################################################################################################
# description: Treynor ratio is a measure of risk-adjusted performance of the portfolio. It is similar to the Sharpe ratio, but it uses beta (systematic risk) 
#              as the risk metric in the denominator.
# input:       returns in decimal format and annual risk free rate in decimal
# output:      Treynor ratio in decimal format
##############################################################################################################################################################
    
def annual_treynor_ratio(returns, bmark_returns, rf, portfolio, period):
    if period == "M":
        multiplier = 12
    elif period == "D":
        multiplier = 252
    else:
        return "Treynor Ratio: Period must be D or M."
    
    if portfolio == 'portfolio':
        tr_beta = beta(returns, bmark_returns)
        return ((returns.mean() * multiplier - rf) / tr_beta)      # ((annualized returns -rf) / beta) 
    else:
        tr_beta = 1        # benchmark
        return ((bmark_returns.mean() * multiplier - rf) / tr_beta)      # ((annualized returns -rf) / beta) 
    

######################################### calmar ratio ##############################################################################################################
# description: the Calmar ratio is a measure of risk-adjusted performance of the portfolio. It is calculated as the annualized return over the past 36 months divided 
#              by the maximum drawdown over the past 36 months.
# input:       returns and balances in decimal format
# output:      calmar ratio in decimal format if >= 36 months, otherwise 0
#####################################################################################################################################################################
    
def calmar_ratio(returns, balance_series, period):

    offset = 37
    
    if (period == 'D'):
        mreturns = returns.resample("M").apply(lambda x: ((x + 1).cumprod() - 1).last('D'))
        mbalance_series = balance_series.resample("M").sum()
        
        mreturns = mreturns.iloc[len(mreturns.index) - offset: len(mreturns.index) - 1]
        mbalance_series = mbalance_series.iloc[len(mbalance_series.index) - offset: len(mbalance_series.index) - 1]
    elif (period == 'M'):
        mreturns = returns.iloc[len(returns.index) - offset: len(returns.index) - 1]
        mbalance_series = balance_series.iloc[len(balance_series.index) - offset: len(balance_series.index) - 1]
    
    DD, Max_DD = drawdowns_and_maxDD(mreturns)
    CAGR = cagr(mreturns.index[0], mreturns.index[-1], mbalance_series.iloc[0], mbalance_series.iloc[-1])

    return CAGR / abs(Max_DD)


######################################### annual active return #################################################################################################################
# description: annual active return is the annual investment return minus the annual return of its benchmark. This is displayed as annualized value, i.e., annualized investment 
#              return minus annualized benchmark return.
# input:       model returns, benchmark returns, benchmark balances and account balances in decimal
# output:      annualized active return
##################################################################################################################################################################
    
def annual_active_return(returns, bmark_returns, model_balance_series, bmark_balance_series):

    returns_cagr = cagr(returns.index[0], returns.index[-1], model_balance_series.iloc[0], model_balance_series.iloc[-1])
    bmark_returns_cagr = cagr(bmark_returns.index[0], bmark_returns.index[-1], bmark_balance_series.iloc[0], bmark_balance_series.iloc[-1])
    
    return returns_cagr - bmark_returns_cagr
     

######################################### monthly active return ###################################################################################################
# description: monthly active return is the monthly investment return minus the monthly return of its benchmark. 
# input:       model returns, benchmark returns in decimal
# output:      monthly active return
###################################################################################################################################################################
    
def monthly_active_return(returns, bmark_returns):
        
    return returns.mean() - bmark_returns.mean()
     

######################################### annual tracking error #################################################################################################################
# description: tracking error is the standard deviation of active return. This is displayed as annualized value based on the standard deviation of monthly active returns.
# input:       model and benchmark monthly return in decimal
# output:      annualized tracking error in decimal
##########################################################################################################################################################################
    
def annual_tracking_error(model_return, benchmark_return, period):
    if period == "D":
        param = 252
    elif period == "M":
        param = 12 
    else:
        return "Tracking Error: Period must be D or M."
        
    return (model_return - benchmark_return).std() * math.sqrt(param)
    

######################################### annual information ratio ##############################################################################################################################
# description: information ratio is the annual active return divided by the annual tracking error. It measures whether the investment outperformed its benchmark consistently.
# input:       model and benchmark returns in decimal
# output:      information ratio in decimal
##########################################################################################################################################################################################
    
def annual_information_ratio(returns, bmark_returns, portfolio_balance, bmark_balance, period):
        
#    return ((returns - bmark_returns).mean() / (returns - bmark_returns).std()) * math.sqrt(12)
    return annual_active_return(returns, bmark_returns, portfolio_balance, bmark_balance) / annual_tracking_error(returns, bmark_returns, period)      

######################################### skewness ###############################################################################################################################################
# description: skewness is the degree of asymmetry observed in a probability distribution. Distributions can exhibit right (positive) skewness or left (negative) skewness to 
#              varying degrees. A normal distribution (bell curve) exhibits zero skewness. Investors note right-skewness when judging a return distribution because it better represents 
#              the extremes of the data set rather than focusing solely on the average. Skewness informs users of the direction of outliers, though it does not tell users the number that occurs.
# input:       model return in decimal
# output:      skew in decimal
##################################################################################################################################################################################################
    
def skewness(returns):
         
    return scipy.stats.skew(returns)
        

######################################### excess kurtosis ###########################################################################################################################
# description: excess kurtosis is used in financial analysis to measure an investment's risk of price volatility. Kurtosis measures the amount of volatility an investment's 
#              price has experienced regularly. High Kurtosis of the return distribution implies that an investment will yield occasional extreme returns.
# input:       model percent return in decimal
# output:      kurtosis in decimal
#######################################################################################################################################################################################
    
def kurtosis(returns):
        
    return scipy.stats.kurtosis(returns, bias=False)
    

######################################### analytical VaR 5% ###########################################################################################################################################
# description: Value at Risk (VaR) measures the scale of loss at a given confidence level. For example, if the 95% confidence one-month VaR is 3%, there is 95% confidence that over the 
#              next month the portfolio will not lose more than 3%. VaR represensts a loss, but it is conventionally reported as a positive number. Value at Risk can be calculated 
#              directly based on historical returns based on a given percentile or analytically based on the mean and standard deviation of the returns.
# input:       model percent return in decimal
# output:      5% VaR in decimal
##########################################################################################################################################################################################
    
def analytical_var(returns):

    return norm.ppf(0.05, loc=returns.mean(), scale=returns.std())


######################################### historical VaR 5% ###########################################################################################################################################
# description: Value at Risk (VaR) measures the scale of loss at a given confidence level. For example, if the 95% confidence one-month VaR is 3%, there is 95% confidence that over the 
#              next month the portfolio will not lose more than 3%. VaR represensts a loss, but it is conventionally reported as a positive number. Value at Risk can be calculated 
#              directly based on historical returns based on a given percentile or analytically based on the mean and standard deviation of the returns.
# input:       model percent return in decimal
# output:      5% VaR in decimal
##########################################################################################################################################################################################
    
def historical_var(returns):

    return np.percentile(returns, 5)  


######################################### conditional VaR 5% ###########################################################################################################################################
# description: Conditional Value at Risk, or CVaR, is an estimate of expected losses sustained in the worst 1 - x% of scenarios. CVaR is commonly quoted with quantiles such as 95, 99, and 99.9.
#              example: CVaR(95) = -2.5%.  in the worst 5% of cases, losses were on average exceed -2.5% historically.
# input:       model percent return in decimal
# output:      5% VaR in decimal
##########################################################################################################################################################################################
    
def conditional_var(returns):
    
    return returns[returns <= historical_var(returns)].mean()
   
   
######################################### capture ratios ######################################################################################
# description: upside capture ratio measures how well the fund performed relative to the benchmark when the market was up.
#              An upside capture ratio greater than 100 would indicate that the fund outperformed its benchmark when the 
#              market was up.  downside capture ratio measures how well the fund performed relative to the benchmark when the market was down.
#              A downside capture ratio less than 100 would indicate that the fund lost less than its benchmark when the 
#              market was down.
# input:       model percent return, benchmark percent returns in decimal
# output:      upside, downside and overall capture ratio in decimal
###############################################################################################################################################
    
def capture_ratios(model_returns, benchmark_returns, period):
    
    if period == 'M':
        num_years = len(model_returns) / 12
    else:
        num_years = len(model_returns) / 252
    
    umask = benchmark_returns >= 0
    mureturns = model_returns.loc[umask]
    bureturns = benchmark_returns.loc[umask]
    upside_capture_ratio = (((1 + mureturns).prod()) ** (1/num_years) - 1) / (((1 + bureturns).prod()) ** (1/num_years) - 1) 
            
    dmask = benchmark_returns < 0
    mdreturns = model_returns.loc[dmask]
    bdreturns = benchmark_returns.loc[dmask]
    downside_capture_ratio = (((1 + mdreturns).prod()) ** (1/num_years) - 1) / (((1 + bdreturns).prod()) ** (1/num_years) - 1) 
       
    return upside_capture_ratio, downside_capture_ratio, (upside_capture_ratio / downside_capture_ratio) 


######################################### reset cumsum ######################################################################################
# description: find the consecutive wins and losses in a run
# input:             
# output:      
############################################################################################################################################

@nb.vectorize([nb.int64(nb.int64, nb.int64)])
def reset_cumsum(x, y):
    return x + y if y else 0


######################################### monte carlo ######################################################################################
# description: run a monte carlo simulation with replacement, based on a set of trades over a set period of time
# input:       trades df, start and end dates, number of runs, initial balance, risk of ruin dd level, risk free rate of return
# output:      MC stats, distributions and plots
############################################################################################################################################
    

def monte_carlo(trades, start, end, runs, init_bal, max_dd_allowed, Rf, period):
    # data structure creation
    equity = pd.DataFrame(columns=['end_balance', 'profit', 'return'], dtype=float, index=range(0,runs))           # holds the ending balance, profit and return of each simulation
    winloss = pd.DataFrame(columns=['consecutive_wins', 'consecutive_losses'], dtype=int, index=range(0,runs))     # holds the consecutive wins/losses of each simulation
    drawdown = pd.DataFrame(columns=['max_dd'], dtype=float, index=range(0,runs))                                  # holds the maximum drawdowns for each simulation
    all_drawdown = pd.DataFrame( dtype=float, index=range(0,runs))                                                 # holds all of the drawdowns for each simulation
    ratios = pd.DataFrame(columns=['mar', 'sharpe'], dtype=float, index=range(0,runs))                             # holds the ratios of each simulation
    metrics_dict = {}                                                                                              # holds metrics
        
    # create a random sequence of trade results for each run and add initial balance to start of each
    randy = np.random.choice(trades['returns'], (len(trades.index), runs))                                 # create a random mix of trade values for each simulation
    randy = np.insert(randy, 0, init_bal, axis=0)                                                          # insert the starting balance at the beginning of each simulation array
    randy_df = pd.DataFrame(randy)                                                                         # collect all simulation run arrays into a dataframe
 
    # calculate the individual run balances and percent returns
    df_bal = randy_df.cumsum(axis=0)                                                                       # create a running balance for each simulation
    df_pct_returns = df_bal.pct_change(axis=0).fillna(0)                                                   # create the running percent returns for each simulation
    
    # calcuate the final equity, profit and return values from all runs
    equity['end_balance'] = randy_df.sum()                                                                 # create the ending balance for each simulation
    equity['profit'] = equity['end_balance'] - init_bal                                                    # create the profit for each simulation
    equity['return'] = equity['profit'] / init_bal                                                         # create the return for each simulation
    
    # calculate the individual run DDs from all runs
    r = df_pct_returns.add(1).cumprod()                                                                    # calculate the cumulative returns
    df_dds = r.div(r.cummax()).sub(1)                                                                      # find each period drawdown by calculating the (max return / cumulative return) - 1

    # calculate the final Max DDs 
    drawdown['max_dd'] = df_dds.min()                                                                      # find the maximum drawdown for each simulation
    all_drawdown = df_dds

    # calculate CAGR
    days = (end - start).days                                                                              # calculate the number of trading days in the trading time period  
    CAGR = (equity['end_balance'] / init_bal) ** (1/(days / 365)) - 1                                      # calculate the yearly CAGR across the time period for each simulation

    # calculate consecutive wins and losses
    win = np.where(randy_df.gt(0), 1, 0)                                                                   # https://stackoverflow.com/questions/75233896/how-to-calculate-
                                                                                                           # cumulative-sums-of-ones-with-a-reset-each-time-a-zero-is-encoun
    win_df = pd.DataFrame(win)
    loss = np.where(randy_df.lt(0), 1, 0)
    loss_df = pd.DataFrame(loss)
    winloss['consecutive_wins'] = reset_cumsum.accumulate(win_df, axis=0).max()
    winloss['consecutive_losses'] = reset_cumsum.accumulate(loss_df, axis=0).max()
   
    # calculate risk of ruin (% chance of exceeding maximum allowed drawdown)
    num_ruins_mean = all_drawdown.mean()
    num_ruins = num_ruins_mean[num_ruins_mean <= -max_dd_allowed].count()                                  # check how many times the max allowed drawdown level was pierced for all simulations
    risk_of_ruin = num_ruins/runs                                                                          # calculate the % risk of ruin
    metrics_dict['ruin'] = risk_of_ruin
    metrics_dict['DD_allowed'] = max_dd_allowed
    
    # calculate MAR, Sharpe ratios 
    ratios['mar'] = (CAGR / abs(drawdown['max_dd']))                                                       # calculate the MAR ratio from the mean GAGR and mean max DD for all runs
    ratios['sharpe'] = annual_sharpe(df_pct_returns, Rf, period)                                           # calculate the Sharpe ratio from the mean return and full standard deviation for all runs 
    
    # calculate upper and lower confidence intervals
    level = 0.95                                                                                           # provide confidence level
    alpha = 1 - level                                                                                      # determine alpha
    if len(trades) > 30:                                                                                   # > 30 trades use z-distribution
        critical = scipy.stats.norm.ppf(1-alpha/2)
    else:                                                                                                  # otherwise use t-distribution
        critical = scipy.stats.t.ppf(1-alpha, len(trades) - 1)
    
    #### equity
    equity_MoE = critical * ((equity['end_balance'].std()) / math.sqrt(runs))                              # calculate margin of error 
    conf_high_equity = equity['end_balance'].mean() + equity_MoE                                           # add to mean and get the high level value
    conf_low_equity = equity['end_balance'].mean() - equity_MoE                                            # subtract from the mean and get the low level value
    metrics_dict['equity_high'] = conf_high_equity
    metrics_dict['equity_low'] = conf_low_equity
        
    #### return
    return_MoE = critical * ((equity['return'].std()) / math.sqrt(runs))                                   # calculate margin of error
    conf_high_return = equity['return'].mean() + return_MoE                                                # add to mean and get the high level value
    conf_low_return = equity['return'].mean() - return_MoE                                                 # subtract from the mean and get the low level value
    metrics_dict['return_high'] = conf_high_return
    metrics_dict['return_low'] = conf_low_return
    
    #### profit
    profit_MoE = critical * ((equity['profit'].std()) / math.sqrt(runs))                                   # calculate margin of error  
    conf_high_profit = equity['profit'].mean() + profit_MoE                                                # add to mean and get the high level value
    conf_low_profit = equity['profit'].mean() - profit_MoE                                                 # subtract from the mean and get the low level value
    metrics_dict['profit_high'] = conf_high_profit
    metrics_dict['profit_low'] = conf_low_profit
    
    #### all_dd
    all_dd_mean = all_drawdown.mean()
    all_dd_std = all_drawdown.std()
    all_dd_MoE = critical * ((all_dd_std.std()) / math.sqrt(runs))                                          # calculate margin of error 
    conf_high_all_dd = all_dd_mean.mean() + all_dd_MoE                                                       # add to mean and get the high level value
    if conf_high_all_dd > 0:
        conf_high_all_dd = 0
    conf_low_all_dd = all_dd_mean.mean() - all_dd_MoE                                                        # subtract from the mean and get the low level value
    metrics_dict['all_dd_high'] = conf_high_all_dd
    metrics_dict['all_dd_low'] = conf_low_all_dd
    
    #### max_dd
    maxdd_MoE = critical * ((drawdown['max_dd'].std()) / math.sqrt(runs))                                  # calculate margin of error 
    conf_high_maxdd = drawdown['max_dd'].mean() + maxdd_MoE                                                # add to mean and get the high level value
    if conf_high_maxdd > 0:
        conf_high_maxdd = 0
    conf_low_maxdd = drawdown['max_dd'].mean() - maxdd_MoE                                                 # subtract from the mean and get the low level value
    metrics_dict['maxdd_high'] = conf_high_maxdd
    metrics_dict['maxdd_low'] = conf_low_maxdd
    
    #### CAGR
    CAGR_MoE = critical * ((CAGR.std()) / math.sqrt(runs))                                                 # calculate margin of error 
    conf_high_CAGR = CAGR.mean() + CAGR_MoE                                                                # add to mean and get the high level value
    conf_low_CAGR = CAGR.mean() - CAGR_MoE                                                                 # subtract from the mean and get the low level value
    metrics_dict['CAGR_high'] = conf_high_CAGR
    metrics_dict['CAGR_low'] = conf_low_CAGR

    #### MAR
    MAR_MoE = critical * ((ratios['mar'].std()) / math.sqrt(runs))                                         # calculate margin of error 
    conf_high_MAR = ratios['mar'].mean() + MAR_MoE                                                         # add to mean and get the high level value
    conf_low_MAR = ratios['mar'].mean() - MAR_MoE                                                          # subtract from the mean and get the low level value
    metrics_dict['MAR_high'] = conf_high_MAR
    metrics_dict['MAR_low'] = conf_low_MAR

    #### Sharpe
    sharpe_MoE = critical * ((ratios['sharpe'].std()) / math.sqrt(runs))                                   # calculate margin of error 
    conf_high_sharpe = ratios['sharpe'].mean() + sharpe_MoE                                                # add to mean and get the high level value
    conf_low_sharpe = ratios['sharpe'].mean() - sharpe_MoE                                                 # subtract from the mean and get the low level value
    metrics_dict['sharpe_high'] = conf_high_sharpe
    metrics_dict['sharpe_low'] = conf_low_sharpe

    #### consecutive wins
    wins_MoE = critical * ((winloss['consecutive_wins'].std()) / math.sqrt(runs))                          # calculate margin of error 
    conf_high_wins = winloss['consecutive_wins'].mean() + wins_MoE                                         # add to mean and get the high level value
    conf_low_wins = winloss['consecutive_wins'].mean() - wins_MoE                                          # subtract from the mean and get the low level value
    metrics_dict['wins_high'] = conf_high_wins
    metrics_dict['wins_low'] = conf_low_wins

    #### consecutive losses
    losses_MoE = critical * ((winloss['consecutive_losses'].std()) / math.sqrt(runs))                      # calculate margin of error 
    conf_high_losses = winloss['consecutive_losses'].mean() + losses_MoE                                   # add to mean and get the high level value
    conf_low_losses = winloss['consecutive_losses'].mean() - losses_MoE                                    # subtract from the mean and get the low level value
    metrics_dict['losses_high'] = conf_high_losses
    metrics_dict['losses_low'] = conf_low_losses


    return equity, drawdown, all_drawdown, CAGR, ratios, winloss, metrics_dict, level


