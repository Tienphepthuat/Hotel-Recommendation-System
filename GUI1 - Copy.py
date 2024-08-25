import pandas as pd
import streamlit as st
import pickle

# Function to get recommendations
def get_recommendations(hotel_id, nums=5):
    if hotel_id not in data_hotel_comments['Hotel_ID'].values:
        st.write("Hotel ID not found.")
        return pd.DataFrame()
    
    idx = data_hotel_comments[data_hotel_comments['Hotel_ID'] == hotel_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums+1]
    hotel_indices = [i[0] for i in sim_scores]
    return data_hotel_comments.iloc[hotel_indices]

# Load hotel data and similarity matrix
data_hotel_comments = pd.read_csv('hotel_info.csv')
df_hotels = pd.read_csv('hotel_info.csv')

with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

# Streamlit GUI
st.title("Hotel Recommendation System")
st.write("Enter a Hotel ID to get similar hotel recommendations.")

hotel_id = st.text_input("Hotel ID", "")

# Initialize session_state if not already set
if 'selected_hotel_id' not in st.session_state:
    st.session_state.selected_hotel_id = None
if 'show_details' not in st.session_state:
    st.session_state.show_details = False

def show_details():
    # Display detailed information for selected hotel
    if st.session_state.selected_hotel_id:
        selected_hotel = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]

        if not selected_hotel.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_hotel['Hotel_Name'].values[0])

            hotel_description = selected_hotel['Hotel_Description'].values[0]
            truncated_description = ' '.join(hotel_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')

            st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
            more_recommendations = get_recommendations(st.session_state.selected_hotel_id, nums=5)
            for j, more_row in more_recommendations.iterrows():
                st.write(f"Hotel ID: {more_row['Hotel_ID']}, Hotel Name: {more_row['Hotel_Name']}")
        else:
            st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")

# When clicking "Get Recommendations"
if st.button("Get Recommendations"):
    if hotel_id:
        recommendations = get_recommendations(hotel_id)
        if not recommendations.empty:
            st.write("Top Recommendations:")
            
            # Display recommended hotels and "Chi tiết tại đây" button
            for i, row in recommendations.iterrows():
                st.write(f"Hotel ID: {row['Hotel_ID']}, Hotel Name: {row['Hotel_Name']}")
                
                # "Chi tiết tại đây" button for each hotel
                if st.button(f"Chi tiết tại đây {row['Hotel_ID']}", key=f"details-{i}"):
                    st.session_state.selected_hotel_id = row['Hotel_ID']
                    st.session_state.show_details = True
                    # Use `st.experimental_rerun` to refresh the app and show details
                    st.experimental_rerun()
        else:
            st.write("No recommendations available.")
    else:
        st.write("Please enter a Hotel ID.")

# Show details for selected hotel
if st.session_state.show_details:
    show_details()


# Additional Features
st.sidebar.title("Filters")
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.0)
max_distance = st.sidebar.slider("Maximum Distance (km)", 0, 50, 10)

if st.sidebar.button("Apply Filters"):
    filtered_recommendations = recommendations[
        (recommendations['Rating'] >= min_rating) & 
        (recommendations['Distance'] <= max_distance)
    ]
    st.write(f"Filtering recommendations with rating >= {min_rating} and distance <= {max_distance} km.")
    for i, row in filtered_recommendations.iterrows():
        st.write(f"Hotel ID: {row['Hotel_ID']}, Hotel Name: {row['Hotel_Name']}")
