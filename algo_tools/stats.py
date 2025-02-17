# path for custom libraries to import
import sys
sys.path.append('/Users/jeff/Dropbox/Trading/Python/src/JP_Tools')

# import the necessary libraries.
import streamlit as st
import pandas as pd
import statistics_lib
import plots_lib
import reports_lib
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
from dateutil import relativedelta
from datetime import datetime
from scipy.stats import bootstrap

# there is some code within pyfolio, empyrical, pandas and matplotlib that has deprecated calls 
# - this just supresses the warnings from showing
warnings.filterwarnings('ignore') 



# level selection data structure
levels = [
    0,
    1,
    2,
    3,
]


# set the title of the app, add a brief description, and set up the select boxes.
st.header("📊 Statistical Analysis of Account and Trading Outcomes", divider = 'rainbow')
st.markdown("""**👈 Select parameters from the sidebar and press the "Start Analysis" button** """)

st.sidebar.header("Parameters", divider = 'rainbow')

with st.form('User Input'):
    selected_level = st.sidebar.number_input("Select the level of statistics to use:", min_value=0, max_value=3, value=0, step=1, format="%i", help='See the welcome page for level contents.')    
    selected_Rf = st.sidebar.number_input("Select the risk free rate of return percentage:", min_value=0.00, max_value=0.10, value=0.00, step=0.01, format="%.2f", help='Typically the short-term, 3-month T-Bill rate.')
    selected_periodicity = st.sidebar.text_input("Select the periodicity:", value="D", max_chars=1, help=' M=month, D=Day')
    selected_bmark = st.sidebar.text_input("Select the benchmark name:", value="SPY", max_chars=4, help='Provide ticker (e.g. SPY, QQQ, etc.)')
    selected_trades_file = st.sidebar.file_uploader("Choose a .csv file for trades data:", help='File header and field format is: | date (trade end date) | returns (trade $ return) | return_pct (trade return percent) |. Best to have 100+ trades.')
    selected_account_file = st.sidebar.file_uploader("Choose a .csv file for account data:", help='File header and field format is: | date | return_pct (return percent) | cum_return_pct (cumulative return percent) | returns ($ return) | balance | DD (drawdown percent)')
    selected_bmark_file = st.sidebar.file_uploader("Choose a .csv file for benchmark data:", help='File header and field format is: | date | return_pct (return percent) | cum_return_pct (cumulative return percent) | returns ($ return) | balance | DD (drawdown percent)')
    submit = st.form_submit_button('Start Analysis')

    if submit:
        # download the data, apply the selected parameters, and run the statistics routine
        trades = pd.read_csv(selected_trades_file)
        trades.set_index('date', drop=True, inplace=True)
        trades.index = pd.to_datetime(trades.index)
        
        account = pd.read_csv(selected_account_file)
        account.set_index('date', drop=True, inplace=True)
        account.index = pd.to_datetime(account.index)
        
        bmark = pd.read_csv(selected_bmark_file)
        bmark.set_index('date', drop=True, inplace=True)
        bmark.index = pd.to_datetime(bmark.index)

        if selected_level == 0 or selected_level == 1 or selected_level == 2 or selected_level == 3:       # always print summary
            # create active returns for heatmap
            # active_df = pd.DataFrame()
            # active_df['returns'] = account['returns'] - bmark['returns']
            # active_df.index = account.index

            st.write('')
            st.write("__Performance Summary:__", f"{account.index[0]:%B %d, %Y}" , "  ", "to","  ", f"{account.index[-1]:%B %d, %Y}")
            st.write('')

            # L0 stats table
            L0 = reports_lib.make_L0_metrics(account, bmark, selected_bmark, selected_periodicity, selected_Rf)
            st.dataframe(L0)
            st.write('')

            # Equity curves
            fig = px.line(account,  x=account.index, y=['balance', bmark['balance']], title="Equity Curves", labels={'value': 'US$', 'index':'Year','variable': ''}, color_discrete_map={
                 "balance": "blue",
                 "bmark['balance']": "orange"
             }) 
            newnames = {'balance':'Model', 'wide_variable_1': selected_bmark}
            fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
            st.plotly_chart(fig)
            st.write('')
    
            # Drawdown curves
            fig = px.line(account,  x=account.index, y=['DD', bmark['DD']], title="Drawdown Curves", labels={'value': 'Percent', 'index':'Year', 'variable': ''}, color_discrete_map={
                 "DD": "blue",
                 "bmark['DD']": "orange"
             } )
            newnames = {'DD':'Model', 'wide_variable_1': selected_bmark}
            fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
            st.plotly_chart(fig)
            st.write('')

            # Account returns
            # Define colors based on y values
            account["color"] = account.returns.apply(lambda val: "less then 0" if val < 0 else "greater than or equal 0")
        
            fig = px.bar(account,  x=account.index, y=['returns'], color=account["color"], color_discrete_map={"less then 0": "red", "greater than or equal 0": "green"}, title="Portfolio Account Returns (%)",  labels={'value': 'Percent', 'index':'Year', 'variable': ''} )
            fig.update_layout(legend_title_text="Account Return Value")
            st.plotly_chart(fig)
            st.write('')

            # Account returns distribution
            # Define colors based on y values
            account["color"] = account.returns.apply(lambda val: "less then 0" if val < 0 else "greater than or equal 0")
        
            fig = px.histogram(account,  x=['returns'], title="Portfolio Account Returns Distribution (%)", color=account["color"], color_discrete_map={"less then 0": "red", "greater than or equal 0": "green"},nbins=250, labels={'value': '% Return', 'variable': ''} )
            
            fig.update_layout(
                yaxis_title_text='Count', # yaxis label
                legend_title_text="Account Return Values"
            )
            st.plotly_chart(fig)
            st.write('')

            
        if selected_level == 1 or selected_level == 3:
            st.write('')
            st.write("__Trade Metrics:__   " ,  f"{account.index[0]:%B %d, %Y}" , "  to  ", f"{account.index[-1]:%B %d, %Y}")
            st.write('')

            L1 = reports_lib.make_L1_metrics(trades)
            st.dataframe(L1)

            # Trade returns
            # Define colors based on y values
            trades["color"] = trades.trade_ret_pct.apply(lambda val: "less then 0" if val < 0 else "greater than or equal 0")
            
            # Create scatter plot
            fig = px.scatter(trades, x=trades.index, y=['trade_ret_pct'], color=trades["color"], color_discrete_map={"less then 0": "red", "greater than or equal 0": "green"},  title="Portfolio Trade Returns (%)", labels={'value': 'Percent', 'index':'Trade Number', 'variable': ''})
            fig.update_layout(legend_title_text="Trade Return Value")
            st.plotly_chart(fig)
            st.write('')
       
            # Trade returns distribution        
            # Define colors based on y values
            trades["color"] = trades.trade_ret_pct.apply(lambda val: "less then 0" if val < 0 else "greater than or equal 0")
        
            fig = px.histogram(trades,  x=['trade_ret_pct'], title="Portfolio Trade Returns Distribution (%)", color=trades["color"], color_discrete_map={"less then 0": "red", "greater than or equal 0": "green"},nbins=100, labels={'value': '% Return', 'variable': ''} )
            
            fig.update_layout(
                yaxis_title_text='Count', # yaxis label
                legend_title_text="Trade Returns Value"
            )    
            st.plotly_chart(fig)
            st.write('')

        
        if selected_level == 2 or selected_level == 3:
            st.write('')
            st.write("__System Metrics:__   " ,  f"{account.index[0]:%B %d, %Y}" , "  to  ", f"{account.index[-1]:%B %d, %Y}")
            st.write('')

            L2 = reports_lib.make_L2_metrics(account, bmark, selected_bmark, selected_periodicity, selected_Rf)
            st.dataframe(L2)
        
            drawdowns_stats = []
            start = 0

            # Top 5 drawdowns
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
                        'Underwater (Months)': f"{(i - start + 1)/30:.2f}",
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
            st.write('')
            st.write('__Model Top 5 Drawdowns__')
            st.write('')
            st.dataframe(drawdowns_df)




