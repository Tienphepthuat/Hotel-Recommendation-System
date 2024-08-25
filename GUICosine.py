import streamlit as st
import pandas as pd
import pickle

# Function to get recommendations
def get_recommendations(hotel_id, nums=5):
    if hotel_id not in df_hotels['Hotel_ID'].values:
        st.write("Hotel ID not found.")
        return pd.DataFrame()
    
    idx = df_hotels[df_hotels['Hotel_ID'] == hotel_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums+1]
    hotel_indices = [i[0] for i in sim_scores]
    return df_hotels.iloc[hotel_indices]

# Load hotel data and similarity matrix
df_hotels = pd.read_csv('hotel_info.csv')

with open(r'D:\Documents\TTTH - KHTN\Project\GUI\GUI_hotel_recommender system\word2vec_model.pkl', 'rb') as f:
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

# Khi nhấn "Get Recommendations"
if st.button("Get Recommendations"):
    st.session_state.show_details = False  # Reset detail view
    if hotel_id:
        recommendations = get_recommendations(hotel_id)
        if not recommendations.empty:
            st.write("Top Recommendations:")
            
            for i, row in recommendations.iterrows():
                st.write(f"Hotel ID: {row['Hotel_ID']}, Hotel Name: {row['Hotel_Name']}")
                if st.button(f"Chi tiết tại đây {row['Hotel_ID']}", key=f"details-{i}"):
                    st.session_state.selected_hotel_id = row['Hotel_ID']
                    st.session_state.show_details = True  # Set detail view

if st.session_state.show_details:
    show_details()
