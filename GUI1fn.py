import streamlit as st
import pandas as pd
import pickle

# Load data and cosine similarity matrix
data_hotel_comments = pd.read_csv('hotel_info.csv')

with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

# Function to get recommendations
def get_recommendations(hotel_id, nums=5):
    if hotel_id not in data_hotel_comments['Hotel_ID'].values:
        st.write("Hotel ID not found.")
        return pd.DataFrame()
    
    idx = data_hotel_comments[data_hotel_comments['Hotel_ID'] == hotel_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums+1]  # Exclude the input hotel itself
    hotel_indices = [i[0] for i in sim_scores]
    return data_hotel_comments.iloc[hotel_indices]

# Streamlit GUI
st.title("Hotel Recommendation System")

# Input for hotel ID
hotel_id = st.text_input("Enter a Hotel ID to get similar hotel recommendations")

if st.button("Get Recommendations"):
    if hotel_id:
        recommendations = get_recommendations(hotel_id)
        if not recommendations.empty:
            st.write("Top Recommendations:")
            
            # Display recommendations in a selectbox for details
            selected_hotel_id = st.selectbox(
                "Select a hotel to view details:",
                options=recommendations['Hotel_ID'].values,
                format_func=lambda x: recommendations[recommendations['Hotel_ID'] == x]['Hotel_Name'].values[0]  # Display hotel name
            )
            
            if selected_hotel_id:
                # Display details of the selected hotel
                selected_hotel = recommendations[recommendations['Hotel_ID'] == selected_hotel_id]
                
                if not selected_hotel.empty:
                    st.write('#### Hotel Details:')
                    st.write('### Name:', selected_hotel['Hotel_Name'].values[0])
                    st.write('#### Address:', selected_hotel['Hotel_Address'].values[0])
                    st.write('#### Rank:', selected_hotel['Hotel_Rank'].values[0])
                    st.write('#### Total Score:', selected_hotel['Total_Score'].values[0])
                    st.write('#### Location:', selected_hotel['Location'].values[0])
                    st.write('#### Cleanliness:', selected_hotel['Cleanliness'].values[0])
                    st.write('#### Service:', selected_hotel['Service'].values[0])
                    st.write('#### Facilities:', selected_hotel['Facilities'].values[0])
                    st.write('#### Value for Money:', selected_hotel['Value_for_money'].values[0])
                    st.write('#### Comfort and Room Quality:', selected_hotel['Comfort_and_room_quality'].values[0])
                    st.write('#### Comments Count:', selected_hotel['comments_count'].values[0])
                    st.write('#### Description:', selected_hotel['Hotel_Description'].values[0])
        else:
            st.write("No recommendations available.")
    else:
        st.write("Please enter a Hotel ID.")

# Additional Features
st.sidebar.title("Filters")
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.0)

if st.sidebar.button("Apply Filters"):
    if 'recommendations' in locals():
        filtered_recommendations = recommendations[
            (recommendations['Total_Score'] >= min_rating)  # Adjusted for Total_Score
        ]
        st.write(f"Filtering recommendations with rating >= {min_rating}.")
        if not filtered_recommendations.empty:
            for i, row in filtered_recommendations.iterrows():
                st.write(f"Hotel ID: {row['Hotel_ID']}, Hotel Name: {row['Hotel_Name']}")
        else:
            st.write("No recommendations meet the filter criteria.")
    else:
        st.write("No recommendations to filter.")
