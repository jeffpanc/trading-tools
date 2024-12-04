
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
st.header("Algorithmic Trading Tools", help='Author: Jeff Pancottine, December 2024.', divider = 'rainbow')
st.write('A set of applications to analyze trading outcomes for quantitative decision making.')
st.markdown("""**👈 Select a tool from the sidebar** """)

st.markdown(
    """
    ### Trading Tools


    ### Want to learn more?
    - https://www.investopedia.com/terms/m/montecarlosimulation.asp
    - https://en.wikipedia.org/wiki/Monte_Carlo_method
    - https://towardsdatascience.com/monte-carlo-simulation-a-practical-guide-85da45597f0e
    - https://www.sciencedirect.com/science/article/abs/pii/0304405X77900058#preview-section-abstract
    - https://hackernoon.com/u/buildalpha

"""
)

