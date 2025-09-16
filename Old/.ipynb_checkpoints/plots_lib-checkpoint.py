########################################### Plots Library ####################################################
#
# Authored by Jeff Pancottine, first release February 2024
#
#
# Change Log
#    - April 2024 -> change df header formats to match account update formats
#
# To Do List
#
##############################################################################################################


################################## Preliminaries - import libraries, etc. ####################################

# import libraries

import pandas as pd
import plotly.express as px
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
from dateutil import relativedelta
from datetime import datetime
import matplotlib.ticker as mtick
from scipy.stats import bootstrap
import matplotlib.pyplot as plt


# there is some code within pyfolio, empyrical, pandas and matplotlib that has deprecated calls 
# - this just supresses the warnings from showing
warnings.filterwarnings('ignore') 


###############################################################################################################


###################### equity curve plot ########################################################################
# input:  portfolio and benchmark series with column names 'Balance' in decimal and date as index, benchmark name
# output: equity curve plot for model and benchmark
#################################################################################################################

def equity_curve_plot(portfolio_df, bmark_df, bmark_name):
    fig = px.line(portfolio_df,  x=portfolio_df.index, y=['balance', bmark_df['balance']], title="Equity Curves", labels={'value': 'US$', 'index':'Year','variable': ''}) 
    newnames = {'balance':'Model', 'wide_variable_1': bmark_name}
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))

    fig.show()

    return




###################### drawdown curve plot #################################################################
# input:  portfolio and benchmark series with column names 'DD' in decimal and date as index, benchmark name
# output: drawdown curve plot for model and benchmark
############################################################################################################

def drawdown_curve_plot(portfolio_DD, bmark_DD, bmark_name):
    fig = px.line(portfolio_DD,  x=portfolio_DD.index, y=[portfolio_DD['DD'], bmark_DD['DD']], title="Drawdown Curves", labels={'value': 'Percent', 'index':'Year', 'variable': ''} )
    newnames = {'DD':'Model', 'wide_variable_1': bmark_name}
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
    fig.show()
    
    return




###################### account returns plot #################################################
# input:  portfolio with column 'Returns' in decimal and date as index
# output: account returns plot
###################################################################################################

def account_returns_plot(portfolio_pct_returns):
    # Define colors based on y values
    portfolio_pct_returns["color"] = portfolio_pct_returns.returns.apply(lambda val: "less then 0" if val < 0 else "greater than or equal 0")

    fig = px.bar(portfolio_pct_returns,  x=portfolio_pct_returns.index, y=['returns'], color=portfolio_pct_returns["color"], color_discrete_map={"less then 0": "red", "greater than or equal 0": "green"}, title="Portfolio Account Returns (%)",  labels={'value': 'Percent', 'index':'Year', 'variable': ''} )
    fig.update_layout(legend_title_text="Account Return Value")
    fig.show()

    return
    




###################### account returns distribution plot #################################################
# input:  portfolio percent returns in decimal with column 'Returns'
# output: account returns distribution plot
################################################################################################################

def account_returns_dist_plot(portfolio_pct_returns):
    # Define colors based on y values
    portfolio_pct_returns["color"] = portfolio_pct_returns.returns.apply(lambda val: "less then 0" if val < 0 else "greater than or equal 0")

    fig = px.histogram(portfolio_pct_returns,  x=['returns'], title="Portfolio Account Returns Distribution (%)", color=portfolio_pct_returns["color"], color_discrete_map={"less then 0": "red", "greater than or equal 0": "green"},nbins=250, labels={'value': '% Return', 'variable': ''} )
    
    fig.update_layout(
        yaxis_title_text='Count', # yaxis label
        legend_title_text="Account Return Values"
    )

    fig.show()

    return 
    




###################### trade returns plot ######################################################
# input:  trades series with percent return in decimal and column 'Return_Pct' and date as index
# output: trade returns plot
################################################################################################

def trade_returns_plot(trades_returns):
    # Define colors based on y values
    trades_returns["color"] = trades_returns.trade_ret_pct.apply(lambda val: "less then 0" if val < 0 else "greater than or equal 0")
    
    # Create scatter plot
    fig = px.scatter(trades_returns, x=trades_returns.index, y=['trade_ret_pct'], color=trades_returns["color"], color_discrete_map={"less then 0": "red", "greater than or equal 0": "green"},  title="Portfolio Trade Returns (%)", labels={'value': 'Percent', 'index':'Trade Number', 'variable': ''})
    fig.update_layout(legend_title_text="Trade Return Value")

    fig.show()

    return 
    




###################### trade returns distribution plot #################################################
# input:  trades series with percent return in decimal and column 'Return_Pct'
# output: trade returns distribution plot
########################################################################################################

def trade_returns_dist_plot(trades_returns):
    # Define colors based on y values
    trades_returns["color"] = trades_returns.trade_ret_pct.apply(lambda val: "less then 0" if val < 0 else "greater than or equal 0")

    fig = px.histogram(trades_returns,  x=['trade_ret_pct'], title="Portfolio Trade Returns Distribution (%)", color=trades_returns["color"], color_discrete_map={"less then 0": "red", "greater than or equal 0": "green"},nbins=100, labels={'value': '% Return', 'variable': ''} )
    
    fig.update_layout(
        yaxis_title_text='Count', # yaxis label
        legend_title_text="Trade Returns Value"
    )

    fig.show()

    return 
    




###################### yearly returns plot ############################################################################################
# input:  portfolio and benchmark series with percent 'Return' columns in decimal, period - 'D' = day and 'M' = month, benchmark symbol
# output: yearly returns plot or 0 if period not 'D' or 'M'
#######################################################################################################################################

def yearly_returns_plot(portfolio_df, bmark_df, bmark, period):
    if period == 'M':
        sample = 'M'
    elif period == "D":
        sample = "D"
        
    ypfreturns = portfolio_df['returns'].resample("Y").apply(lambda x: ((x + 1).cumprod() - 1).last(sample))  
    ypf = ypfreturns.index.to_list()
    pfyears = []
    for tstamp in ypf:
        pfyears.append(tstamp.year)
    ypfreturns.index = pfyears
    ypfreturns.index.name = 'Year'

    ybmreturns = bmark_df['returns'].resample("Y").apply(lambda x: ((x + 1).cumprod() - 1).last(sample))
    ybm = ybmreturns.index.to_list()
    bmyears = []
    for tstamp in ybm:
        bmyears.append(tstamp.year)
    ybmreturns.index = bmyears
    ybmreturns.index.name = 'Year'
    
    fig = px.bar(ypfreturns, x=ypfreturns.index, y=['returns', ybmreturns],  text_auto='0.2%', title="Yearly Returns (%)", labels={'value': 'Percent', 'variable': ''} )
    fig.update_layout(barmode='group')
    newnames = {'returns':'Model', 'wide_variable_1': bmark}
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
    fig.show()

    return 
    



###################### rolling returns plot ############################################################################################
# input:  portfolio and benchmark series with balance columns, period - 'D' = day and 'M' = month, benchmark symbol
# output: rolling 3/6/12 month returns plot or 0 if period not 'D' or 'M'
#######################################################################################################################################

def rolling_returns_plot(portfolio_df, bmark_df, bmark, period):
    if period == 'D':     # add extra month to getting correct open
        roll1 = 63 + 21
        roll2 = 126 + 21
        roll3 = 252 + 21
    elif period == 'M':
        roll1 = 3 + 1
        roll2 = 6 + 1
        roll3 = 12 + 1
        
    # rolling returns
    three_month = (portfolio_df['balance'].iloc[-1]-portfolio_df['balance'].iloc[-roll1])/portfolio_df['balance'].iloc[-roll1]
    six_month = (portfolio_df['balance'].iloc[-1]-portfolio_df['balance'].iloc[-roll2])/portfolio_df['balance'].iloc[-roll2]
    twelve_month = (portfolio_df['balance'].iloc[-1]-portfolio_df['balance'].iloc[-roll3])/portfolio_df['balance'].iloc[-roll3]
    pfroll = ['3-Months', '6-Months', '12-Months']
    pfreturns = pd.DataFrame({'returns':[three_month, six_month, twelve_month]}, index=pfroll)
    pfreturns.index.name = 'Period'
    
    b_three_month = (bmark_df['balance'].iloc[-1]-bmark_df['balance'].iloc[-roll1])/bmark_df['balance'].iloc[-roll1]    
    b_six_month = (bmark_df['balance'].iloc[-1]-bmark_df['balance'].iloc[-roll2])/bmark_df['balance'].iloc[-roll2]    
    b_twelve_month = (bmark_df['balance'].iloc[-1]-bmark_df['balance'].iloc[-roll3])/bmark_df['balance'].iloc[-roll3]    
    bmroll = ['3-Months', '6-Months', '12-Months']
    bmreturns = pd.DataFrame({'returns':[b_three_month, b_six_month, b_twelve_month]}, index=bmroll)
    bmreturns.index.name = 'Period'

    fig = px.bar(pfreturns, x=pfreturns.index, y=['returns', bmreturns['returns']],  text_auto='0.2%', title="Rolling Returns (%)", labels={'value': 'Percent', 'variable': ''} )
    fig.update_layout(barmode='group')
    newnames = {'returns':'Model', 'wide_variable_1': bmark}
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
    fig.show()

    return 
    




###################### monthly returns heatmap plot #########################################################
# input:  portfolio df with 'Returns' column in decimal, period - 'D' = day and 'M' = month, benchmark symbol
# output: monthly returns heatmap plot or 0 if period not 'D' or 'M'
#############################################################################################################

def monthly_returns_heatmap_plot(portfolio_df, period, data_type):

    if period == 'D':
        mreturns = portfolio_df['returns'].resample("M").apply(lambda x: ((x + 1).cumprod() - 1).last("D"))
        yreturns = portfolio_df['returns'].resample("Y").apply(lambda x: ((x + 1).cumprod() - 1).last("D"))
    elif period == 'M':
        mreturns = portfolio_df['returns']
        yreturns =  portfolio_df['returns'].resample("Y").apply(lambda x: ((x + 1).cumprod() - 1).last("M")) 
    else:
        return "Monthly Heatmap: Period must be D or M."

    mlist = mreturns.to_list()
    first_month = mreturns.index[0].month
    
    y = yreturns.index.to_list()
    
    month_matrix = np.zeros(len(y) * 12)
    for i in range(0, len(mlist)):
        month_matrix[i + (first_month-1)] = mlist[i]
    
    years = []
    for tstamp in y:
        years.append(tstamp.year)
    years = list(map(str, years))     # change from int to string for formatting
    
    z = month_matrix.reshape((len(y),12))
    
    fig = px.imshow(z,
                    color_continuous_scale = 'RdYlGn',
                    color_continuous_midpoint = 0,
                    labels=dict(x="Month", y="Year", color="Monthly Percent Return"),
                    x = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
                    y=years, text_auto='0.2%', aspect="auto", 
                   )
    if data_type == 'active':
        fig.update_layout(title='Monthly Active Returns (%) - Heatmap')
    else:
        fig.update_layout(title='Monthly Returns (%) - Heatmap')
    
    fig.show()

    return 
    



    
####################### top 5 drawdowns table #################################################
# input:  account series in decimal with datetime index, Balance, Returns and DD
# output: top 5 drawdowns table
###############################################################################################

def calculate_top5_drawdowns(account):

    drawdowns_stats = []
    start = 0

    for i in range(1, len(account)):
        if account.DD.iloc[i] < 0 and account.DD.iloc[i-1] == 0:               # find the peak
            peak_date = account.index[i]
            start = i
        elif account.DD.iloc[i] == 0 and account.DD.iloc[i-1] < 0:             # find the recovery
            record = account[account.DD == account.DD.iloc[start:i].min()]
            valley_date = record.index[0]
            recovery_date = account.index[i]
            r1 = relativedelta.relativedelta(valley_date, peak_date) 
            r2 = relativedelta.relativedelta(recovery_date, valley_date)
            drawdowns_stats.append({
                'Drawdown': account.DD.iloc[start:i].min(),
                'Start Date': datetime.strftime(peak_date, '%Y-%m'),
                'End Date': datetime.strftime(valley_date, '%Y-%m'),
                'Drawdown (Months)': r1.months + 1 + (12 * r1.years),          # add 1 for current month
                'Recovery Date': datetime.strftime(recovery_date, '%Y-%m'),
                'Recovery (Months)': r2.months + (12 * r2.years),
                'Underwater (Months)': i - start + 1,
            })
            start = i
        elif i == len(account) - 1 and account.DD.iloc[i] < 0:                 # handle ongoing drawdown
            record2 = account[account.DD == account.DD.iloc[start:].min()]
            valley_date = record2.index[0]
            peak_date = account.index[start]
            r3 = relativedelta.relativedelta(valley_date, peak_date)
            drawdowns_stats.append({
                'Drawdown': account.DD.iloc[start:].min(),
                'Start Date': datetime.strftime(peak_date, '%Y-%m'),
                'End Date': datetime.strftime(valley_date, '%Y-%m'),
                'Drawdown (Months)': r3.months + 1 + (12 * r3.years),          # add 1 for current month
                'Recovery Date': 'On Going',       # Ongoing drawdown
                'Recovery (Months)': 'On Going',   # Ongoing drawdown
                'Underwater (Months)': 'On Going', # Ongoing drawdown
            })

    # Sort drawdowns by depth
    drawdowns_stats = sorted(drawdowns_stats, key=lambda x: x['Drawdown'])[:5]
    

    # Convert to DataFrame for easier viewing/manipulation
    drawdowns_df = pd.DataFrame(drawdowns_stats)
    drawdowns_df.index = drawdowns_df.index + 1

    # Display the top 5 drawdowns
    display(drawdowns_df)




########################################## trade entry/exit chart #################################################################################################################
# description: build an interactive chart of OHLC and trade markers - render chart and display trades df
# input:       symbol dataframe with OHLC; trades dataframe with - symbol, entry_date, entry_price, num_shares, exit_date, exit_price, trade_pnl, trade_ret; symbol of traded asset
# output:      interactive chart and trades df
###################################################################################################################################################################################

def trade_chart(ttf, trades_df, symbol):
    # # get  all bars elder impulses
    # impulses = elder_impulses(ttf)
    
    # make the figure to chart on
    fig = go.Figure()
    
    # # make three subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02,
                        row_heights=[0.6,0.2,0.2])
    
    # make the candlesticks to plot 1
    fig.add_trace(go.Candlestick(x=ttf.index,
        open=ttf['Open'],
        high=ttf['High'],
        low=ttf['Low'],
        close=ttf['Close'],
        increasing_line_color= 'green', decreasing_line_color= 'red',
        name=symbol + " OHLC"))
    fig.update_layout(xaxis_rangeslider_visible=False, autosize=False,
        width=1700,
        height=1000,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=2),
        title=symbol + " (Daily)")
    
    # # add the EMAs to plot 1
    # fig.add_trace(go.Scatter(x=ttf.index, 
    #                       y=ttf['ema_200'], 
    #                       mode = 'lines',
    #                       name="200 Day EMA"))
    # fig.add_trace(go.Line(x=ttf.index, 
    #                       y=ttf['ema_20'], 
    #                       name="20 Day EMA"))
    
    # add entries, exits and impulses to plot 1
    fig.add_trace(go.Scatter(x=trades_df['entry_date'], 
                             y=trades_df['entry_price'], 
                             mode='markers',
                             customdata = trades_df,
                             marker_symbol = 'diamond-dot',
                             marker_size = 13,
                             marker_line_width = 2,
                             marker_line_color = "rgba(0,0,0,0.7)",
                             marker_color = "rgba(0,255,0,0.7)",
                             hovertemplate = "Entry Date: %{customdata[1]}<br>" +\
                                             "Entry Price: %{y:.2f}<br>" +\
                                             "Size: %{customdata[3]}<br>" +\
                                             "Profit Percent: %{customdata[7]:.2%}",
                             name = "Entries"                         
                ), row=1, col = 1)
    fig.add_trace(go.Scatter(x=trades_df['exit_date'], 
                             y=trades_df['exit_price'], 
                             mode='markers',
                             customdata = trades_df,
                             marker_symbol = 'diamond-dot',
                             marker_size = 13,
                             marker_line_width = 2,
                             marker_line_color = "rgba(0,0,0,0.7)",
                             marker_color = "rgba(255,0,0,0.7)",
                             hovertemplate = "Exit Date: %{customdata[4]}<br>" +\
                                             "Exit Price: %{y:.2f}<br>" +\
                                             "Size: %{customdata[3]}<br>" +\
                                             "Profit Percent: %{customdata[7]:.2%}",
                             name = "Exits"                         
                ), row=1, col=1)
    # impulse_colors = impulses.to_list()
    # fig.add_trace(go.Scatter(x=ttf.index, 
    #                          y=ttf.Low * 0.98, 
    #                          mode='markers',
    #                          marker_symbol = 'circle',
    #                          marker_color = impulse_colors,
    #                          name = "Elder Impulse"                         
    #             ),row=1, col=1)
    
    # # add the macdh to plot 2
    # colors = ['green' if val >= 0
    #           else 'red' for val in ttf['macdh_12_26_9']]
    # fig.add_trace(go.Bar(x=ttf.index,
    #                      y=ttf['macdh_12_26_9'],
    #                      marker_color = colors, name="MACDH"
    #                     ), row=2, col=1)
    
    # # add the rsi to plot 3
    # colors = 'blue'
    # fig.add_trace(go.Line(x=ttf.index,
    #                      y=ttf['rsi_2'],
    #                      marker_color = colors, name="RSI 2"
    #                     ), row=3, col=1)
    # fig.add_hline(y=85, line_color="green",
    #               annotation_text="85", 
    #               annotation_position="top right", row=3)
    # fig.add_hline(y=15, line_color="red",
    #               annotation_text="15", 
    #               annotation_position="bottom right", row=3)
    
    # # add exits to rsi plot 3
    # trades_df.drop(0, inplace = True)                            # remove first row of trades holding only initial balance
    # fig.add_trace(go.Scatter(x=trades_df['exit_date'], 
    #                          y=ttf.rsi_2.loc[trades_df.exit_date], 
    #                          mode='markers',
    #                          customdata = trades_df,
    #                          marker_symbol = 'circle-dot',
    #                          marker_size = 13,
    #                          marker_line_width = 2,
    #                          marker_line_color = "rgba(0,0,0,0.7)",
    #                          marker_color = "rgba(255,0,0,0.7)",
    #                          name = "RSI Exit"                         
    #             ), row=3, col=1)
    
    # name the plots
    fig.update_yaxes(title_text="Price")
    # fig.update_yaxes(title_text="MACDH", row=2, col=1)
    # fig.update_yaxes(title_text="RSI 2", row=3, col=1)
    # fig.update_yaxes(title_text="EMA 13", row=4, col=1)
    
    # print the charts and trades df
    fig.show()
    print(trades_df)

    # save chart
    # fig.write_html("chart_file.html")
    

###################### MC equity distribution plot #################################################
# input:  equity df
# output: equity distribution plots and metrics
##########################################################################################

def MC_equity_bal_dist_plot(equity_df):

    fig, ax = plt.subplots()
    ax.hist(equity_df['end_balance'], bins=100)
    fmt = '${x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.xaxis.set_major_formatter(tick) 
    plt.xticks(rotation=25)
    ax.set_title('Equity Distribution')
    ax.set_xlabel('Equity (US$)')
    ax.set_ylabel('Frequency')
    plt.axvline(equity_df['end_balance'].mean(), color= 'red', linestyle= 'dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(1, .5, 'Min: ${:,.2f}'.format(equity_df['end_balance'].min()), transform=plt.gcf().transFigure)
    plt.text(1, .45, 'Max: ${:,.2f}'.format(equity_df['end_balance'].max()), transform=plt.gcf().transFigure)
    plt.text(1, .4, 'Mean: ${:,.2f}'.format(equity_df['end_balance'].mean()), transform=plt.gcf().transFigure, color='red')
    plt.text(1, .35, 'Median: ${:,.2f}'.format(equity_df['end_balance'].median()), transform=plt.gcf().transFigure)
    plt.text(1, .3, 'StDev: ${:,.2f}'.format(equity_df['end_balance'].std()), transform=plt.gcf().transFigure)
    plt.show()

    return


###################### MC all DD distribution plot #################################################
# input:  all DD df
# output: all DD distribution plots and metrics
##########################################################################################

def MC_DD_dist_plot(all_drawdown_df):

    all_dd_mean = all_drawdown_df.mean()
    all_dd_median = all_drawdown_df.median()
    all_dd_min = all_drawdown_df.min()
    all_dd_max = all_drawdown_df.max()
    all_dd_std = all_drawdown_df.std()

    fig, ax = plt.subplots()
    ax.hist(all_dd_mean * 100, bins=100)
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.xaxis.set_major_formatter(tick) 
    plt.xticks(rotation=25)
    ax.set_title('Drawdowns Distribution')
    ax.set_xlabel('DD (%)')
    ax.set_ylabel('Frequency')
    plt.axvline(all_dd_mean.mean() * 100, color= 'red', linestyle= 'dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(1, .5,  'Min: {:,.2%}'.format(all_dd_max.max()), transform=plt.gcf().transFigure)
    plt.text(1, .45, 'Max: {:,.2%}'.format(all_dd_min.min()), transform=plt.gcf().transFigure)
    plt.text(1, .4, 'Mean: {:,.2%}'.format(all_dd_mean.mean()), transform=plt.gcf().transFigure, color='red')
    plt.text(1, .35, 'Median: {:,.2%}'.format(all_dd_median.median()), transform=plt.gcf().transFigure)
    plt.text(1, .3, 'StDev: {:,.2%}'.format(all_dd_std.std()), transform=plt.gcf().transFigure)
    plt.show()

    return
###################### MC max DD distribution plot #################################################
# input:  DD df
# output: DD distribution plots and metrics
##########################################################################################

def MC_maxDD_dist_plot(drawdown_df):

    fig, ax = plt.subplots()
    ax.hist(drawdown_df['max_dd'] * 100, bins=100)
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.xaxis.set_major_formatter(tick) 
    plt.xticks(rotation=25)
    ax.set_title('Max Drawdown Distribution')
    ax.set_xlabel('Max DD (%)')
    ax.set_ylabel('Frequency')
    plt.axvline(drawdown_df['max_dd'].mean() * 100, color= 'red', linestyle= 'dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(1, .5,  'Min: {:,.2%}'.format(drawdown_df['max_dd'].max()), transform=plt.gcf().transFigure)
    plt.text(1, .45, 'Max: {:,.2%}'.format(drawdown_df['max_dd'].min()), transform=plt.gcf().transFigure)
    plt.text(1, .4, 'Mean: {:,.2%}'.format(drawdown_df['max_dd'].mean()), transform=plt.gcf().transFigure, color='red')
    plt.text(1, .35, 'Median: {:,.2%}'.format(drawdown_df['max_dd'].median()), transform=plt.gcf().transFigure)
    plt.text(1, .3, 'StDev: {:,.2%}'.format(drawdown_df['max_dd'].std()), transform=plt.gcf().transFigure)
    plt.show()

    return

###################### MC profit distribution plot #################################################
# input:  equity df
# output: profit distribution plots and metrics
##########################################################################################

def MC_profit_dist_plot(equity_df):

    fig, ax = plt.subplots()
    ax.hist(equity_df['profit'], bins=100)
    fmt = '${x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.xaxis.set_major_formatter(tick) 
    plt.xticks(rotation=25)
    ax.set_title('Profit Distribution')
    ax.set_xlabel('Profit (US$)')
    ax.set_ylabel('Frequency')
    plt.axvline(equity_df['profit'].mean(), color= 'red', linestyle= 'dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(1, .5, 'Min: ${:,.2f}'.format(equity_df['profit'].min()), transform=plt.gcf().transFigure)
    plt.text(1, .45, 'Max: ${:,.2f}'.format(equity_df['profit'].max()), transform=plt.gcf().transFigure)
    plt.text(1, .4, 'Mean: ${:,.2f}'.format(equity_df['profit'].mean()), transform=plt.gcf().transFigure, color='red')
    plt.text(1, .35, 'Median: ${:,.2f}'.format(equity_df['profit'].median()), transform=plt.gcf().transFigure)
    plt.text(1, .3, 'StDev: ${:,.2f}'.format(equity_df['profit'].std()), transform=plt.gcf().transFigure)
    plt.show()

    return


###################### MC total returns distribution plot #################################################
# input:  equity df
# output: total returns distribution plots and metrics
##########################################################################################

def MC_returns_dist_plot(equity_df):

    fig, ax = plt.subplots()
    ax.hist(equity_df['return'], bins=100)
    fmt = '{x:,.0%}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.xaxis.set_major_formatter(tick) 
    plt.xticks(rotation=25)
    ax.set_title('Total Return Distribution')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    plt.axvline(equity_df['return'].mean(), color= 'red', linestyle= 'dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(1, .5, 'Min: {:,.2%}'.format(equity_df['return'].min()), transform=plt.gcf().transFigure)
    plt.text(1, .45, 'Max: {:,.2%}'.format(equity_df['return'].max()), transform=plt.gcf().transFigure)
    plt.text(1, .4, 'Mean: {:,.2%}'.format(equity_df['return'].mean()), transform=plt.gcf().transFigure, color='red')
    plt.text(1, .35, 'Median: {:,.2%}'.format(equity_df['return'].median()), transform=plt.gcf().transFigure)
    plt.text(1, .3, 'StDev: {:,.2%}'.format(equity_df['return'].std()), transform=plt.gcf().transFigure)
    plt.show()

    return



###################### MC CAGR distribution plot #################################################
# input:  CAGR df
# output: CAGR distribution plots and metrics
##########################################################################################

def MC_CAGR_dist_plot(CAGR_df):

    fig, ax = plt.subplots()
    ax.hist(CAGR_df * 100, bins=100)
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.xaxis.set_major_formatter(tick) 
    plt.xticks(rotation=25)
    ax.set_title('CAGR Distribution')
    ax.set_xlabel('CAGR (%)')
    ax.set_ylabel('Frequency')
    plt.axvline(CAGR_df.mean() * 100, color= 'red', linestyle= 'dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(1, .5, 'Min: {:,.2%}'.format(CAGR_df.min()), transform=plt.gcf().transFigure)
    plt.text(1, .45, 'Max: {:,.2%}'.format(CAGR_df.max()), transform=plt.gcf().transFigure)
    plt.text(1, .4, 'Mean: {:,.2%}'.format(CAGR_df.mean()), transform=plt.gcf().transFigure, color='red')
    plt.text(1, .35, 'Median: {:,.2%}'.format(CAGR_df.median()), transform=plt.gcf().transFigure)
    plt.text(1, .3, 'StDev: {:,.2%}'.format(CAGR_df.std()), transform=plt.gcf().transFigure)
    plt.show()

    return



###################### MC MAR distribution plot #################################################
# input:  MAR df
# output: MAR distribution plots and metrics
##########################################################################################

def MC_MAR_dist_plot(Ratios_df):

    fig, ax = plt.subplots()
    ax.hist(Ratios_df['mar'], bins=100)
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.xaxis.set_major_formatter(tick) 
    plt.xticks(rotation=25)
    ax.set_title('MAR Distribution')
    ax.set_xlabel('MAR')
    ax.set_ylabel('Frequency')
    plt.axvline(Ratios_df['mar'].mean(), color= 'red', linestyle= 'dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(1, .5, 'Min: {:,.2f}'.format(Ratios_df['mar'].min()), transform=plt.gcf().transFigure)
    plt.text(1, .45, 'Max: {:,.2f}'.format(Ratios_df['mar'].max()), transform=plt.gcf().transFigure)
    plt.text(1, .4, 'Mean: {:,.2f}'.format(Ratios_df['mar'].mean()), transform=plt.gcf().transFigure, color='red')
    plt.text(1, .35, 'Median: {:,.2f}'.format(Ratios_df['mar'].median()), transform=plt.gcf().transFigure)
    plt.text(1, .3, 'StDev: {:,.2f}'.format(Ratios_df['mar'].std()), transform=plt.gcf().transFigure)
    plt.show()

    return



###################### MC Sharpe distribution plot #################################################
# input:  sharpe df
# output: sharpe distribution plots and metrics
##########################################################################################

def MC_sharpe_dist_plot(Ratios_df):

    fig, ax = plt.subplots()
    ax.hist(Ratios_df['sharpe'], bins=100)
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.xaxis.set_major_formatter(tick) 
    plt.xticks(rotation=25)
    ax.set_title('Sharpe Distribution')
    ax.set_xlabel('Sharpe')
    ax.set_ylabel('Frequency')
    plt.axvline(Ratios_df['sharpe'].mean(), color= 'red', linestyle= 'dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(1, .5, 'Min: {:,.2f}'.format(Ratios_df['sharpe'].min()), transform=plt.gcf().transFigure)
    plt.text(1, .45, 'Max: {:,.2f}'.format(Ratios_df['sharpe'].max()), transform=plt.gcf().transFigure)
    plt.text(1, .4, 'Mean: {:,.2f}'.format(Ratios_df['sharpe'].mean()), transform=plt.gcf().transFigure, color='red')
    plt.text(1, .35, 'Median: {:,.2f}'.format(Ratios_df['sharpe'].median()), transform=plt.gcf().transFigure)
    plt.text(1, .3, 'StDev: {:,.2f}'.format(Ratios_df['sharpe'].std()), transform=plt.gcf().transFigure)
    plt.show()

    return



###################### win distribution plot #################################################
# input:  winloss df
# output: consecutive wins distribution plots and metrics
##########################################################################################

def MC_Wins_dist_plot(winloss_df):

    fig, ax = plt.subplots()
    ax.hist(winloss_df['consecutive_wins'], bins=100)
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.xaxis.set_major_formatter(tick) 
    plt.xticks(rotation=25)
    ax.set_title('Consecutive Wins Distribution')
    ax.set_xlabel('# Wins')
    ax.set_ylabel('Frequency')
    plt.axvline(winloss_df['consecutive_wins'].mean(), color= 'red', linestyle= 'dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(1, .5, 'Min: {:,.2f}'.format(winloss_df['consecutive_wins'].min()), transform=plt.gcf().transFigure)
    plt.text(1, .45, 'Max: {:,.2f}'.format(winloss_df['consecutive_wins'].max()), transform=plt.gcf().transFigure)
    plt.text(1, .4, 'Mean: {:,.2f}'.format(winloss_df['consecutive_wins'].mean()), transform=plt.gcf().transFigure, color='red')
    plt.text(1, .35, 'Median: {:,.2f}'.format(winloss_df['consecutive_wins'].median()), transform=plt.gcf().transFigure)
    plt.text(1, .3, 'StDev: {:,.2f}'.format(winloss_df['consecutive_wins'].std()), transform=plt.gcf().transFigure)
    plt.show()

    return


###################### loss distribution plot #################################################
# input:  winloss df
# output: consecutive losses distribution plots and metrics
##########################################################################################

def MC_Losses_dist_plot(winloss_df):

    fig, ax = plt.subplots()
    ax.hist(winloss_df['consecutive_losses'], bins=100)
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.xaxis.set_major_formatter(tick) 
    plt.xticks(rotation=25)
    ax.set_title('Consecutive Losses Distribution')
    ax.set_xlabel('# Losses')
    ax.set_ylabel('Frequency')
    plt.axvline(winloss_df['consecutive_losses'].mean(), color= 'red', linestyle= 'dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(1, .5, 'Min: {:,.2f}'.format(winloss_df['consecutive_losses'].min()), transform=plt.gcf().transFigure)
    plt.text(1, .45, 'Max: {:,.2f}'.format(winloss_df['consecutive_losses'].max()), transform=plt.gcf().transFigure)
    plt.text(1, .4, 'Mean: {:,.2f}'.format(winloss_df['consecutive_losses'].mean()), transform=plt.gcf().transFigure, color='red')
    plt.text(1, .35, 'Median: {:,.2f}'.format(winloss_df['consecutive_losses'].median()), transform=plt.gcf().transFigure)
    plt.text(1, .3, 'StDev: {:,.2f}'.format(winloss_df['consecutive_losses'].std()), transform=plt.gcf().transFigure)
    plt.show()

    return


# rng = np.random.default_rng()
# data = (drawdown_runs['max_dd'],)  # samples must be in a sequence
# res = bootstrap(data, np.mean, confidence_level=0.99,
#                 random_state=rng)
# print(res.confidence_interval)
# print(res.standard_error)
# fig, ax = plt.subplots()
# ax.hist(res.bootstrap_distribution, bins=100)
# ax.set_title('Bootstrap Distribution')
# ax.set_xlabel('statistic value')
# ax.set_ylabel('frequency')
# plt.show()




