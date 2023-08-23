import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import openpyxl

# Config
st.set_page_config(
    page_title="GetAround Dashboard",
    page_icon=":blue_car:",
    layout="wide"
)

#read in wine image and render with streamlit
logo_image = Image.open('GetAround_logo.png')

# Display the logo image with a small width
small_width = 500  # Adjust this value as needed
st.image(logo_image, width=small_width)

### VIDEO EXPANDER
with st.expander("⏯️ **French commercial for Getaround!**"):
    # st.markdown('<p style="font-size: 24px;">**French commercial for Getaround!**</p>', unsafe_allow_html=True)
  
    st.video("https://youtu.be/A9NGC9-vsqo")

# video_url = "https://www.youtube.com/watch?v=ftXVbgWtVj8&ab_channel=PraiseWorshipMusic"
# video_embed_code = f'<iframe width="560" height="315" src="{video_url}" frameborder="0" allowfullscreen></iframe>'
# st.markdown(video_embed_code, unsafe_allow_html=True)


st.markdown("---")


st.markdown("""
    The concept is straightforward, yet regrettably, there are instances where drivers arrive late for checkouts. Such scenarios can pose challenges, particularly if there's a subsequent rental scheduled immediately after. Presented below are data-driven insights to empower you in making informed choices regarding the optimal interval between successive car rentals.  👇
""")

# Set title and markdown 
st.title("Getaround Analysis on Delays ⏱")
st.markdown('''DELAYS IMPACT ANALYSIS ON GETAROUND USERS''')

# Use `st.cache` to put data in cache
# The dataset will not be reloaded each time the app is refreshed
@st.cache_data()
def load_data(nrows=''):
    data = pd.DataFrame()
    if(nrows == ''):
        data = pd.read_excel("https://storage.googleapis.com/jedha-projects/get_around_delay_analysis.xlsx")
    else:
        data = pd.read_excel("https://storage.googleapis.com/jedha-projects/get_around_delay_analysis.xlsx",nrows=nrows)

    return data

# ### SIDEBAR
# st.sidebar.header("Sections")
# st.sidebar.markdown("""
#     * **Original Data and Basic Information**
#     * **Focus on Delays**
#     * **Threshold Simulation**
# """)
st.sidebar.header("Sections")
st.sidebar.markdown("""
    * <span style='color: #ffffff; font-weight: bold'>Original Data and Basic Information</span>
    * <span style='color: #ffffff; font-weight: bold'>Focus on Delays</span>
    * <span style='color: #ffffff; font-weight: bold'>Threshold Simulation</span>
""", unsafe_allow_html=True)


e = st.sidebar.empty()
e.write("")
st.sidebar.write("Contact [Valérie MUTHIANI (Linkedin)](https://www.linkedin.com/in/val%C3%A9rie-muthiani-58864458/s)")


# Load the data
data_loading_status = st.text('Loading data...')
data = load_data()
data_loading_status.text('data_loaded ✔️')

st.markdown("")

# Data exploration with some informations about the dataset
st.subheader("DATA EXPLORATION")

# Run the below code if the check is checked ✅
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data) 

# We consider no values on "delay_at_checkout_in_minutes" as oversights and consider them "in time checkout"
data["delay_at_checkout_in_minutes"].fillna(0, inplace=True) 

# Creating a new df to be able to merge both of them and add the values from the previous renting on a global df
data_bis = data.loc[:, ['rental_id', 'checkin_type', 'state', 'delay_at_checkout_in_minutes']]
data_bis = data_bis.rename(columns={"rental_id": "previous_rental_id","checkin_type": "previous_checkin_type", "state": "previous_state","delay_at_checkout_in_minutes": "previous_delay_at_checkout_in_minutes" })

# Mergin dataframes
full_df = pd.merge(data, data_bis, how='left', left_on='previous_ended_rental_id', right_on='previous_rental_id')
full_df = full_df.drop("previous_rental_id", axis=1)

# Adding new columns to see if checkout on-time or late, and to have the time delta between 2 rentals (planned + real one)
full_df['on_time-late'] = full_df["delay_at_checkout_in_minutes"].apply(lambda x: "In time or in advance" if x <= 0 else "Late")
full_df['previous_on_time-late'] = full_df["previous_delay_at_checkout_in_minutes"].apply(lambda x: "In time or in advance" if x <= 0 else ("Late" if x > 0 else "no previous renting"))
full_df = full_df[['rental_id', 'car_id', 'checkin_type', 'state',
       'delay_at_checkout_in_minutes','on_time-late', 'previous_ended_rental_id',
       'time_delta_with_previous_rental_in_minutes', 'previous_checkin_type',
       'previous_state', 'previous_delay_at_checkout_in_minutes','previous_on_time-late']]
full_df["time_delta_with_previous_rental_in_minutes"].fillna(1440, inplace=True) # if no information about past rantal, we set a timedelta of 24h
full_df['real_time_delta'] = full_df['time_delta_with_previous_rental_in_minutes'] - full_df['previous_delay_at_checkout_in_minutes']



col1, col2 = st.columns(2)

with col1:
    st.metric(label="Number of cars", value=full_df['car_id'].nunique())

with col2:
    st.metric(label="Number of rentals", value=full_df['rental_id'].nunique())


st.markdown("")
st.markdown("")
st.markdown("")

# # Present the dataset
# columns=" "
# for column in data.columns:
#     columns=columns+" "+str(column)+"/ "
# st.markdown(f"""
#     The dataset represent {len(data)} rental records and {data['car_id'].nunique()}.\n
#     The informations contained are: {columns}""")

col1, col2, col3 = st.columns([15,5,40])

with col1:
    st.markdown('')
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("**Proportion of rentals: Mobile / Connect**")
    st.markdown("")
    st.markdown("")
    fig = px.pie(full_df, values='rental_id', names="checkin_type", width= 1000, color='checkin_type', hole=0.33)       
    st.plotly_chart(fig, use_container_width=True)



with col3:
    # checkin_type1 = st.selectbox("<span style='color: purple;'>Checkin type</span>", ['all', 'mobile', 'connect'], key=1, unsafe_allow_html=True)
    checkin_type1 = st.selectbox("Checkin type", ['all', 'mobile', 'connect'], key=1)
   
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('**Proportion of late checkouts per type**')
            st.markdown(
        """
        <style>
        span[data-baseweb="tag"] {
        background-color: blue !important;
        }
        </style>
         """,
        unsafe_allow_html=True)
            
            df_delay = full_df if checkin_type1 == 'all' else full_df[full_df["checkin_type"]==checkin_type1]
            fig = px.pie(df_delay, values='rental_id', names="on_time-late",
                        color='on_time-late',
                        color_discrete_map={'Late':'#eb2f2f','In time or in advance':'#317AC1'}, hole=0.33)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Proportion of canceled rentals per type**")
            df_cancellation = full_df if checkin_type1 == 'all' else full_df[full_df["checkin_type"]==checkin_type1]
            fig = px.pie(df_cancellation, values='rental_id', names="state",
                        color='state', hole=0.33,
                        color_discrete_map={'canceled':'#eb2f2f','ended':'#317AC1'})
            st.plotly_chart(fig, use_container_width=True)


st.markdown("**There are more late checkouts with mobile rentals**")
st.markdown("We can assume that checkouts are much quicker when there is only one person involved (connect rentals: no need for the owner to be present).")
st.markdown("")
st.markdown("**There are more cancellations with connect rentals**")
st.markdown("We can assume that people are less ashamed of canceling the rental when there is no appointment with the owner.")
st.markdown("")
st.markdown("")

# # Late checkouts proportions
# # We can consider that negatif delays as in time 
# st.subheader('LATE CHECKOUT PROPORTION')
data['checkout_status']=["Late" if x>0 else "in_time" for x in data.delay_at_checkout_in_minutes]
# fig = px.pie(data, names='checkout_status', title='LATE CHECKOUT PROPORTION')
# st.plotly_chart(fig)

 #Figure #1
data_filtred = data[data['time_delta_with_previous_rental_in_minutes'].notna() & data['delay_at_checkout_in_minutes'].notna()]
total_rows = len(data_filtred[data_filtred['delay_at_checkout_in_minutes'] <= 0])
delay_gt_0 = len(data_filtred[data_filtred['delay_at_checkout_in_minutes'] > 0])
delay_gt_previous = len(data_filtred[data_filtred['delay_at_checkout_in_minutes'] > data_filtred['time_delta_with_previous_rental_in_minutes']])
percentage_gt_0 = (delay_gt_0 / total_rows) * 100
percentage_gt_previous = (delay_gt_previous / total_rows) * 100

colors = ['#317AC1', '#eb2f2f', '#77021d']
colors2 = ['#eb2f2f', '#317AC1']
labels = ['Rentals without delay or in advance at checkout', 'Rentals with delay at checkout', 'Rentals with delay at checkout preventing the next rental from being done on time']
values = [total_rows, delay_gt_0, delay_gt_previous]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.33)])

fig.update_traces(marker=dict(colors=colors))

fig.update_layout(title='Percentage of checkout delays',
                    showlegend=True)

st.plotly_chart(fig, use_container_width=True)

# Delay at checkout repartition visualization
st.subheader('DELAY REPARTITION VISUALIZATION')
st.markdown("""Let's have a closer look to the delays repartition""")
choice = st.selectbox("select values to be displayed", ["all_values","values_without_outliers"])
data_with_delay=data[data.delay_at_checkout_in_minutes>0]
if choice=="all_values":
    # Graph showing the time passed in minutes before a late checkout
    fig = px.histogram(
        data_with_delay["delay_at_checkout_in_minutes"],
        x="delay_at_checkout_in_minutes")
    fig.update_layout()
    st.plotly_chart(fig, use_container_width=True)
    
    
else:
    # Graph showing the time passed in minutes before a late checkout
    fig = px.histogram(
        data_with_delay[
            (data_with_delay["delay_at_checkout_in_minutes"] <(2*data_with_delay['delay_at_checkout_in_minutes'].std()))
        ]["delay_at_checkout_in_minutes"],
        x="delay_at_checkout_in_minutes")
    fig.update_layout()
    st.plotly_chart(fig, use_container_width=True)



# Consecutive rentals
st.subheader("CONSECUTIVE RENTALS")


# Get one dataset with 
consecutive_rental_data = pd.merge(data, data, how='inner', left_on = 'previous_ended_rental_id', right_on = 'rental_id')

consecutive_rental_data.drop(
    [
        "delay_at_checkout_in_minutes_x",
        "rental_id_y", 
        "car_id_y", 
        "state_y",
        "time_delta_with_previous_rental_in_minutes_y",
        "previous_ended_rental_id_y",
        "checkout_status_x"
    ], 
    axis=1,
    inplace=True
)

consecutive_rental_data.columns = [
    'rental_id',
    'car_id',
    'checkin_type',
    'state',
    'previous_ended_rental_id',
    'time_delta_with_previous_rental_in_minutes',
    'previous_checkin_type',
    'previous_delay_at_checkout_in_minutes',
    "previous_checkout_status"
]

# Remove rows with missing previous rental delay values
consecutive_rental_data = consecutive_rental_data[~consecutive_rental_data["previous_delay_at_checkout_in_minutes"].isnull()]
consecutive_rental_data.reset_index(drop=True, inplace=True)

# Run the below code if the check is checked ✅
if st.checkbox('Show consecutive rental dataset'):
    st.write(consecutive_rental_data) 

# Count the number of consecutive rentals cases
st.markdown(f"""
    Let's have a look to the consecutive rentals to understand the impact of delays on next users.
 
    The total number of usable cases is: **{len(consecutive_rental_data)}**
""")



# Impacted users with previous delay
consecutive_rental_data['delayed_checkin_in_minutes']=[
    consecutive_rental_data.previous_delay_at_checkout_in_minutes[i]-consecutive_rental_data.time_delta_with_previous_rental_in_minutes[i] for i in range(len(consecutive_rental_data))
    ]

cancellation_df=consecutive_rental_data[
    (consecutive_rental_data["delayed_checkin_in_minutes"]>0) & (consecutive_rental_data["state"]=="canceled")
    ]

impacted_df= consecutive_rental_data[consecutive_rental_data.delayed_checkin_in_minutes>0]
st.markdown(f"""
    The number of checkins impacted by previous delays is:  **{len(impacted_df)}**

    The number of potential cancellations due to delays is:  **{len(cancellation_df)}**\n
    
    ---------------------------------------------------------------------------------\n

""")

#### Create two columns
col1, col2 = st.columns(2)

with col1:
    # Run the below code if the check is checked ✅
    if st.checkbox(' Histogram without outliers'):
        fig = px.histogram(
            impacted_df[impacted_df.delayed_checkin_in_minutes<2*impacted_df.delayed_checkin_in_minutes.std()],
            x="delayed_checkin_in_minutes", color="state", color_discrete_map={'canceled':'#eb2f2f','ended':'#317AC1'}, title='DELAYED CHECKIN'
            )
        st.plotly_chart(fig)        
    else:
        fig = px.histogram(impacted_df,x="delayed_checkin_in_minutes", color="state", color_discrete_map={'canceled':'#eb2f2f','ended':'#317AC1'},title='DELAYED CHECKIN')
        st.plotly_chart(fig)

with col2:
    fig = px.pie(consecutive_rental_data, hole=0.33, names='state',color='state', color_discrete_map={'canceled':'#eb2f2f','ended':'#317AC1'}, title='DELAYED CHECKIN STATUS')
    st.plotly_chart(fig)


# Threshold: minimum time between two rentals
st.subheader("Threshold testing")


# Threshold form
with st.form("Threshhold"):
    threshold = st.number_input("Threshold in minutes", min_value = 0, step = 1)
    checkin_type = st.selectbox("Checkin types", ["Connect only", "Mobile only","All"])
    submit = st.form_submit_button("submit")

    if submit:
        consecutive_rental_data_selected = impacted_df
        cancellation_df_selected= cancellation_df
        #select checkin type "connect"
        if checkin_type == "Connect only":
            consecutive_rental_data_selected = consecutive_rental_data_selected[consecutive_rental_data_selected["checkin_type"] == "connect"]
            cancellation_df_selected= cancellation_df[cancellation_df["checkin_type"] == "connect"]
        elif checkin_type == "Mobile only":
            consecutive_rental_data_selected = consecutive_rental_data_selected[consecutive_rental_data_selected["checkin_type"] == "mobile"]
            cancellation_df_selected= cancellation_df[cancellation_df["checkin_type"] == "mobile"]
        


        avoided_checkin_delays = len(consecutive_rental_data_selected[consecutive_rental_data_selected["delayed_checkin_in_minutes"] < threshold])
            
        avoided_cancellation = len(cancellation_df_selected[cancellation_df_selected["delayed_checkin_in_minutes"] < threshold])


        percentage_avoided_checkin_delays=round((avoided_checkin_delays/len(consecutive_rental_data_selected))*100, 1)
        precentage_avoided_cancellations=round((avoided_cancellation/len(cancellation_df_selected))*100, 1)
        st.markdown(f"""
            With a threshold of **{threshold}**minutes on **{checkin_type}** there is:
            - **{avoided_checkin_delays}** avoided checkin delays cases ({percentage_avoided_checkin_delays}% solved)
            - **{avoided_cancellation}** avoided cancellations (due to delays) cases ({precentage_avoided_cancellations}% solved)
        """)

st.markdown("---")
st.markdown("made by [Valerie MUTHIANI (Github : Valla77)](https://github.com/Valla77)")
