import streamlit as st
import pandas as pd
import pickle
from surprise import SVD


# Load hotel data
df_hotels = pd.read_csv('hotel_info.csv')

# Load similarity matrices and models
with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

with open('svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)

# Function to get content-based recommendations
def get_content_based_recommendations(hotel_id, nums=5):
    if hotel_id not in df_hotels['Hotel_ID'].values:
        st.write("Hotel ID not found.")
        return pd.DataFrame()
    
    idx = df_hotels[df_hotels['Hotel_ID'] == hotel_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums+1]
    hotel_indices = [i[0] for i in sim_scores]
    return df_hotels.iloc[hotel_indices]

# Function to get collaborative recommendations
def get_collaborative_recommendations(user_id, nums=5):
    user_id = int(user_id)
    pred_ratings = []
    for hotel_id in df_hotels['Hotel_ID'].unique():
        pred_rating = svd_model.predict(user_id, hotel_id).est
        pred_ratings.append((hotel_id, pred_rating))
    
    # Sort hotels by predicted rating
    pred_ratings = sorted(pred_ratings, key=lambda x: x[1], reverse=True)
    hotel_indices = [df_hotels[df_hotels['Hotel_ID'] == x[0]].index[0] for x in pred_ratings[:nums]]
    return df_hotels.iloc[hotel_indices]

# Streamlit GUI
st.title("Hotel Recommendation System")

# Sidebar menu for selecting the recommender system
menu_option = st.sidebar.selectbox("Choose Recommender System", ["Content-Based Recommender", "Collaborative Recommender"])

if menu_option == "Content-Based Recommender":
    st.subheader("Content-Based Recommender")
    st.write("Enter a Hotel ID to get similar hotel recommendations.")
    
    hotel_id = st.text_input("Hotel ID", "")
    
    # Initialize session_state if not already set
    if 'selected_hotel_id' not in st.session_state:
        st.session_state.selected_hotel_id = None
    if 'show_details' not in st.session_state:
        st.session_state.show_details = False
    
    if st.button("Get Recommendations"):
        st.session_state.show_details = False  # Reset detail view
        if hotel_id:
            recommendations = get_content_based_recommendations(hotel_id)
            if not recommendations.empty:
                st.write("Top Recommendations:")
                
                for i, row in recommendations.iterrows():
                    st.write(f"Hotel ID: {row['Hotel_ID']}, Hotel Name: {row['Hotel_Name']}")
                    if st.button(f"Chi tiết tại đây {row['Hotel_ID']}", key=f"details-{i}"):
                        st.session_state.selected_hotel_id = row['Hotel_ID']
                        st.session_state.show_details = True  # Set detail view
    
    if st.session_state.show_details:
        selected_hotel = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]
    
        if not selected_hotel.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_hotel['Hotel_Name'].values[0])
    
            hotel_description = selected_hotel['Hotel_Description'].values[0]
            truncated_description = ' '.join(hotel_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')
    
            st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
            more_recommendations = get_content_based_recommendations(st.session_state.selected_hotel_id, nums=5)
            for j, more_row in more_recommendations.iterrows():
                st.write(f"Hotel ID: {more_row['Hotel_ID']}, Hotel Name: {more_row['Hotel_Name']}")

elif menu_option == "Collaborative Recommender":
    st.subheader("Collaborative Recommender")
    st.write("Enter a User ID to get hotel recommendations.")
    
    user_id = st.text_input("User ID", "")
    
    if st.button("Get Recommendations"):
        if user_id:
            recommendations = get_collaborative_recommendations(user_id)
            if not recommendations.empty:
                st.write("Top Recommendations:")
                
                for i, row in recommendations.iterrows():
                    st.write(f"Hotel ID: {row['Hotel_ID']}, Hotel Name: {row['Hotel_Name']}")
                    
                    # Display detailed information if button clicked
                    if st.button(f"Chi tiết tại đây {row['Hotel_ID']}", key=f"details-colla-{i}"):
                        st.session_state.selected_hotel_id = row['Hotel_ID']
                        st.session_state.show_details = True  # Set detail view
    
    if st.session_state.show_details and st.session_state.selected_hotel_id:
        selected_hotel = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]
    
        if not selected_hotel.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_hotel['Hotel_Name'].values[0])
    
            hotel_description = selected_hotel['Hotel_Description'].values[0]
            truncated_description = ' '.join(hotel_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')
    
            st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
            more_recommendations = get_collaborative_recommendations(st.session_state.selected_hotel_id, nums=5)
            for j, more_row in more_recommendations.iterrows():
                st.write(f"Hotel ID: {more_row['Hotel_ID']}, Hotel Name: {more_row['Hotel_Name']}")
