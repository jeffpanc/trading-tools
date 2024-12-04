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
    selected_level = st.sidebar.number_input("Select the level of statistics to use:", min_value=0, max_value=3, value=0, step=1, format="%i", help='See the intro page for level contents.')    
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

        if selected_level == 0:
            st.write('')
            st.write("Performance Summary: " +  f"{account.index[0]:%B %d, %Y}" + " to " f"{account.index[-1]:%B %d, %Y}")
            st.write('')

            L0 = reports_lib.make_L0_metrics(account, bmark, selected_bmark, selected_periodicity, selected_Rf)
            st.dataframe(L0)
            st.write('')

            fig = px.line(account,  x=account.index, y=['balance', bmark['balance']], title="Equity Curves (Log Scale)", labels={'value': 'US$', 'index':'Year','variable': ''}, log_x=True) 
            newnames = {'balance':'Model', 'wide_variable_1': selected_bmark}
            fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
            st.plotly_chart(fig)

        elif selected_level == 1:
            L0 = reports_lib.make_L0_metrics(account, bmark, selected_bmark, selected_periodicity, selected_Rf)
            st.dataframe(L0)
            L1 = reports_lib.make_L1_metrics(trades)
            st.dataframe(L1)
        elif selected_level == 2:
            L0 = reports_lib.make_L0_metrics(account, bmark, selected_bmark, selected_periodicity, selected_Rf)
            st.dataframe(L0)
            L2 = reports_lib.make_L2_metrics(account, bmark, selected_bmark, selected_periodicity, selected_Rf)
            st.dataframe(L2)
        elif selected_level == 3:
            L0 = reports_lib.make_L0_metrics(account, bmark, selected_bmark, selected_periodicity, selected_Rf)
            st.dataframe(L0)
            L1 = reports_lib.make_L1_metrics(trades)
            st.dataframe(L1)
            L2 = reports_lib.make_L2_metrics(account, bmark, selected_bmark, selected_periodicity, selected_Rf)
            st.dataframe(L2)
        
        



