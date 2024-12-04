
# to run the app: enter the command at your terminal window in the same directory as the script you created. 
#     cd Dropbox/Python/Finance
#     streamlit run MC_web.py

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
import numba as nb



# selection data structures
runs = [
    "100",
    "1000",
    "5000",
    "10000",
]

ruin = [
    "0.10",
    "0.25",
    "0.35",
    "0.50",
    "0.75",
    "1.00",
]

balance = [
    "10000.00",
    "100000.00",
    "500000.00",
    "1000000.00",
]

risk_free = [
    "0.00",
    "0.01",
    "0.03",
    "0.05",
]

# # configure the page
# st.set_page_config(
#      page_title="Monte Carlo Trading Analysis",
#      page_icon="📈",
#      layout="wide",
#      initial_sidebar_state="expanded")

# set the title of the app, add a brief description, and set up the select boxes.
st.header("🎲 Possible Future Outcomes and Account Sizing", divider = 'rainbow')
st.markdown(
    """
    **Uses:**
    - **Set better expectations of future outcomes -->** Perform resampling techniques with replacement on a set of trades to analyze possible future profit, equity, drawdown, CAGR, consecutive wins and losses, and returns.  Provides confidence intervals and ratio analysis.  
    - **Estimate correct account size -->** Change beginning balance until risk of ruin is 10% or less. Larger account size and/or larger drawdown decrease risk of ruin. 
    """) 
st.markdown("""**👈 Select parameters from the sidebar and press the "Start Analysis" button** """)


st.sidebar.header("Parameters", divider = 'rainbow')

with st.form('User Input'):
    selected_runs = st.sidebar.number_input("Select the number of simulations to run:", min_value=1000, max_value=10000, value=10000, step=1000, format="%i", help='The more simulations, the more accurate the results.')    
    selected_ruin = st.sidebar.number_input("Select the maximum percent drawdown to consider risk of ruin:", min_value=0.10, max_value=1.00, value=0.25, step=0.05, format="%.2f", help='How much of a drawdown can you handle?')
    selected_balance = st.sidebar.number_input("Select the beginning account balance (US$):", min_value=100000.00, max_value=10000000.00, value=100000.00, step=250000.00, format="%.2f", help='Increase value to lower drawdown.')
    selected_Rf = st.sidebar.number_input("Select the risk free rate of return percentage:", min_value=0.00, max_value=0.10, value=0.00, step=0.01, format="%.2f", help='Typically the short-term, 3-month T-Bill rate.')
    selected_period = st.sidebar.text_input("Select the periodicity:", value="D", max_chars=1, help=' M=month, D=Day')
    selected_file = st.sidebar.file_uploader("Choose a .csv file for trades:", help='File header and field format is: | date (trade end date) | returns (trade $ return) | return_pct (trade return percent) |. Best to have 100+ trades.')
    submit = st.form_submit_button('Start Analysis')

    if submit:
        # download the data, apply the selected parameters, and run the monte carlo routine
        data = pd.read_csv(selected_file)
        data.set_index('date', drop=True, inplace=True)
        data.index = pd.to_datetime(data.index)
        equity_df, drawdown_df, all_drawdown_df, CAGR_df, Ratios_df, winloss_df, metrics, ci_score = statistics_lib.monte_carlo(data, data.index[0], data.index[-1], int(selected_runs), float(selected_balance), float(selected_ruin), float(selected_Rf), selected_period)
        
        # set up and display tables and charts
        col = st.columns((5.0, 5.0), gap='medium')

        # with col[0]:  
        #    st.markdown('## <center> - Analysis -', unsafe_allow_html=True)
       
        # with col[1]:
        #     st.markdown(' ')
        #     st.markdown(' ')
        #     st.markdown(' ')
        #     st.markdown(' ')
         
        with col[0]:
            st.markdown('#### Summary')
            test_metrics = pd.DataFrame(columns=['Metric', 'Value'])
            test_metrics.loc[len(test_metrics.index)] = ['Initial Balance', "${:,.2f}".format(float(selected_balance))]
            test_metrics.loc[len(test_metrics.index)] = ['# Simulations', "{:,}".format(int(selected_runs))]
            test_metrics.loc[len(test_metrics.index)] = ['# Trades/Simulation', "{:,}".format(len(data))]
            test_metrics.loc[len(test_metrics.index)] = ['Risk of Ruin Percent Drawdown', "{:,.2f}%".format(float(selected_ruin)*100)]
            test_metrics.loc[len(test_metrics.index)] = ['Risk Free Rate of Return Percent', "{:,.2f}%".format(float(selected_Rf)*100)]
            test_metrics.loc[len(test_metrics.index)] = ['First Trade - Start Date', "{:%m-%d-%Y}".format(data.index[0])]
            test_metrics.loc[len(test_metrics.index)] = ['Last Trade - End Date', "{:%m-%d-%Y}".format(data.index[-1])]
            test_metrics.loc[len(test_metrics.index)] = ['Risk of Ruin', "{:.2%}".format(metrics.get('ruin'))]
            test_metrics.loc[len(test_metrics.index)] = [' ', ' ']
            st.dataframe(test_metrics, hide_index=True)
        
        with col[1]:
            st.markdown('#### Confidence Intervals (' + str(ci_score * 100) + '%)')
            conf_intervals = pd.DataFrame(columns=['Metric', 'Lower', 'Upper'])
            conf_intervals.loc[len(conf_intervals.index)] = ['Equity', "${:,.2f}".format(metrics.get('equity_low')), "${:,.2f}".format(metrics.get('equity_high'))]
            conf_intervals.loc[len(conf_intervals.index)] = ['All-DD', "{:,.2%}".format(metrics.get('all_dd_low')), "{:,.2%}".format(metrics.get('all_dd_high'))]
            conf_intervals.loc[len(conf_intervals.index)] = ['Profit', "${:,.2f}".format(metrics.get('profit_low')), "${:,.2f}".format(metrics.get('profit_high'))]
            conf_intervals.loc[len(conf_intervals.index)] = ['Total Return', "{:,.2%}".format(metrics.get('return_low')), "{:,.2%}".format(metrics.get('return_high'))]
            conf_intervals.loc[len(conf_intervals.index)] = ['CAGR', "{:,.2%}".format(metrics.get('CAGR_low')), "{:,.2%}".format(metrics.get('CAGR_high'))]
            conf_intervals.loc[len(conf_intervals.index)] = ['Consecutive Wins', "{:,}".format(int(metrics.get('wins_low'))), "{:,}".format(int(metrics.get('wins_high')))]
            conf_intervals.loc[len(conf_intervals.index)] = ['Consecutive Losses', "{:,}".format(int(metrics.get('losses_low'))), "{:,}".format(int(metrics.get('losses_high')))]
            conf_intervals.loc[len(conf_intervals.index)] = ['Sharpe Ratio', "{:,.2f}".format(metrics.get('sharpe_low')), "{:,.2f}".format(metrics.get('sharpe_high'))]
            conf_intervals.loc[len(conf_intervals.index)] = ['MAR Ratio', "{:,.2f}".format(metrics.get('MAR_low')), "{:,.2f}".format(metrics.get('MAR_high'))]
            st.dataframe(conf_intervals, hide_index=True)

        # with col[0]:
        #     st.markdown('## <center> - Distributions -', unsafe_allow_html=True)
       
        # with col[1]:
        #     st.markdown(' ')
        #     st.markdown(' ')
        #     st.markdown(' ')
        #     st.markdown(' ')
       
        with col[0]:
            # st.markdown('#### Equity')
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
            st.pyplot(fig)
        
        with col[1]:
            # st.markdown('#### All Drawdown')
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
            st.pyplot(fig)
        
        with col[0]:
            # st.markdown('#### Profit')
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
            st.pyplot(fig)
        
        with col[1]:
            # st.markdown('#### Total Returns')
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
            st.pyplot(fig)
        
        with col[0]:
            # st.markdown('#### CAGR')
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
            st.pyplot(fig)

        with col[0]:
            # st.markdown('#### MAR')
            fig, ax = plt.subplots()
            ax.hist(Ratios_df['mar'], bins=100)
            fmt = '{x:,.0f}'
            tick = mtick.StrMethodFormatter(fmt)
            ax.xaxis.set_major_formatter(tick) 
            plt.xticks(rotation=25)
            ax.set_title('MAR Ratio Distribution')
            ax.set_xlabel('MAR')
            ax.set_ylabel('Frequency')
            plt.axvline(Ratios_df['mar'].mean(), color= 'red', linestyle= 'dashed', linewidth=1)
            min_ylim, max_ylim = plt.ylim()
            plt.text(1, .5, 'Min: {:,.2f}'.format(Ratios_df['mar'].min()), transform=plt.gcf().transFigure)
            plt.text(1, .45, 'Max: {:,.2f}'.format(Ratios_df['mar'].max()), transform=plt.gcf().transFigure)
            plt.text(1, .4, 'Mean: {:,.2f}'.format(Ratios_df['mar'].mean()), transform=plt.gcf().transFigure, color='red')
            plt.text(1, .35, 'Median: {:,.2f}'.format(Ratios_df['mar'].median()), transform=plt.gcf().transFigure)
            plt.text(1, .3, 'StDev: {:,.2f}'.format(Ratios_df['mar'].std()), transform=plt.gcf().transFigure)
            st.pyplot(fig)
        
        with col[1]:
            # st.markdown('#### Sharpe')
            fig, ax = plt.subplots()
            ax.hist(Ratios_df['sharpe'], bins=100)
            fmt = '{x:,.0f}'
            tick = mtick.StrMethodFormatter(fmt)
            ax.xaxis.set_major_formatter(tick) 
            plt.xticks(rotation=25)
            ax.set_title('Sharpe Ratio Distribution')
            ax.set_xlabel('Sharpe')
            ax.set_ylabel('Frequency')
            plt.axvline(Ratios_df['sharpe'].mean(), color= 'red', linestyle= 'dashed', linewidth=1)
            min_ylim, max_ylim = plt.ylim()
            plt.text(1, .5, 'Min: {:,.2f}'.format(Ratios_df['sharpe'].min()), transform=plt.gcf().transFigure)
            plt.text(1, .45, 'Max: {:,.2f}'.format(Ratios_df['sharpe'].max()), transform=plt.gcf().transFigure)
            plt.text(1, .4, 'Mean: {:,.2f}'.format(Ratios_df['sharpe'].mean()), transform=plt.gcf().transFigure, color='red')
            plt.text(1, .35, 'Median: {:,.2f}'.format(Ratios_df['sharpe'].median()), transform=plt.gcf().transFigure)
            plt.text(1, .3, 'StDev: {:,.2f}'.format(Ratios_df['sharpe'].std()), transform=plt.gcf().transFigure)
            st.pyplot(fig)
        
        with col[1]:
            # st.markdown('#### Consecutive Wins')
            fig, ax = plt.subplots()
            ax.hist(winloss_df['consecutive_wins'], bins=25)
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
            st.pyplot(fig)

        
        with col[0]:
            # st.markdown('#### Consecutive Losses')
            fig, ax = plt.subplots()
            ax.hist(winloss_df['consecutive_losses'], bins=25)
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
            st.pyplot(fig)



