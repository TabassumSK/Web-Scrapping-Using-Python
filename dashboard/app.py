import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# -----------------
# Page Configuration
# -----------------
st.set_page_config(
    page_title="Airline Sentiment Analysis",
    page_icon="✈️",
    layout="wide"
)

# -----------------
# API Configuration
# -----------------
# This is the URL of your FastAPI backend
API_URL = "http://127.0.0.1:8000/predict"

# -----------------
# Helper Functions
# -----------------
@st.cache_data  # Cache the data to avoid reloading on every interaction
def load_data(filepath):
    data = pd.read_csv(filepath)
    # Just keep the columns we need
    data = data[['tweet_id', 'airline_sentiment', 'airline', 'text', 'negativereason']]
    data['airline_sentiment'] = data['airline_sentiment'].astype('category')
    return data

def get_realtime_prediction(tweet_text):
    """Calls the FastAPI backend to get a sentiment prediction."""
    try:
        response = requests.post(API_URL, json={"text": tweet_text})
        if response.status_code == 200:
            result = response.json()
            return result['sentiment'], result['confidence']
        else:
            return "Error: Could not connect to API", None
    except requests.exceptions.ConnectionError:
        return "Error: Backend API is not running.", None

# -----------------
# Main Dashboard
# -----------------
st.title("✈️ U.S. Airline Customer Service Sentiment")

# Load the dataset
data = load_data("Tweets.csv")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Project Dashboard", "Real-Time Prediction"])

if page == "Project Dashboard":
    st.header("Analysis of 14,000+ Customer Tweets")

    # --- Filter by Airline ---
    st.sidebar.header("Filters")
    selected_airline = st.sidebar.selectbox(
        "Select an Airline (or all)",
        options=["All"] + sorted(data['airline'].unique())
    )
    
    if selected_airline == "All":
        filtered_data = data
    else:
        filtered_data = data[data['airline'] == selected_airline]

    st.header(f"Dashboard for: {selected_airline}")

    # --- Key Metrics ---
    total_tweets = len(filtered_data)
    positive_tweets = len(filtered_data[filtered_data['airline_sentiment'] == 'positive'])
    neutral_tweets = len(filtered_data[filtered_data['airline_sentiment'] == 'neutral'])
    negative_tweets = len(filtered_data[filtered_data['airline_sentiment'] == 'negative'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tweets", f"{total_tweets}")
    col2.metric("Positive", f"{positive_tweets} ({positive_tweets/total_tweets:.1%})")
    col3.metric("Neutral", f"{neutral_tweets} ({neutral_tweets/total_tweets:.1%})")
    col4.metric("Negative", f"{negative_tweets} ({negative_tweets/total_tweets:.1%})")
    
    st.markdown("---")

    # --- Visualizations ---
    col1, col2 = st.columns(2)

    # 1. Sentiment Breakdown (Pie Chart)
    with col1:
        st.subheader("Overall Sentiment Breakdown")
        if total_tweets > 0:
            sentiment_counts = filtered_data['airline_sentiment'].value_counts()
            fig_pie = px.pie(
                sentiment_counts, 
                values=sentiment_counts.values, 
                names=sentiment_counts.index, 
                title="Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map={'positive':'green', 'negative':'red', 'neutral':'blue'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("No data to display for this airline.")

    # 2. Top 5 Negative Reasons (Bar Chart) - The "Advanced" part
    with col2:
        st.subheader("Top 5 Negative Feedback Reasons")
        negative_data = filtered_data[filtered_data['airline_sentiment'] == 'negative']
        if not negative_data.empty:
            reason_counts = negative_data['negativereason'].value_counts().nlargest(5)
            fig_bar = px.bar(
                reason_counts, 
                x=reason_counts.index, 
                y=reason_counts.values,
                title="Top 5 Negative Reasons",
                labels={'x': 'Reason', 'y': 'Number of Tweets'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No negative feedback to display.")

    # --- Raw Data ---
    st.subheader("Raw Tweet Data")
    st.dataframe(filtered_data[['airline', 'airline_sentiment', 'negativereason', 'text']].head(10))


elif page == "Real-Time Prediction":
    st.header("🤖 Real-Time Tweet Sentiment Predictor")
    st.write("Test the underlying `roBERTa` model by typing a tweet.")
    
    tweet_input = st.text_area("Enter a tweet:", "The flight was great, but the service was terrible.")

    if st.button("Predict Sentiment"):
        sentiment, confidence = get_realtime_prediction(tweet_input)
        
        if "Error" in sentiment:
            st.error(sentiment)
        else:
            if sentiment == "Positive":
                st.success(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
            elif sentiment == "Negative":
                st.error(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
            else:
                st.info(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")