import streamlit as st
import pandas as pd
import pickle



# Streamlit GUI
st.title("Hotel Recommendation System")
st.image('hotel.jpg', use_column_width=True)
st.write("Enter a Hotel ID to get similar hotel recommendations.")
hotel_id = st.text_input("Hotel ID", "")

# Initialize session_state if not already set
if 'selected_hotel_id' not in st.session_state:
    st.session_state.selected_hotel_id = None
if 'show_details' not in st.session_state:
    st.session_state.show_details = False

# Load hotel data and similarity matrix
df_hotels = pd.read_csv('hotel_info.csv')
with open('D:\Documents\TTTH - KHTN\Project\GUI\GUI_hotel_recommender system\cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

# Define function to get hotel description
def get_hotel_description(hotel_id):
    hotel_row = df_hotels[df_hotels['Hotel_ID'] == hotel_id]
    if not hotel_row.empty:
        return hotel_row.iloc[0]['Description']
    return 'No description available.'

# Define function to show hotel details
def show_details(hotel_id):
    description = get_hotel_description(hotel_id)
    st.write(f"Details for Hotel ID {hotel_id}:")
    st.write(description)

# Define function to get recommendations
def get_recommendations(df, hotel_id, cosine_sim, nums=5):
    matching_indices = df.index[df['Hotel_ID'] == hotel_id].tolist()
    if not matching_indices:
        st.write(f"No hotel found with ID: {hotel_id}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    idx = matching_indices[0]

    # Get the pairwise similarity scores of all hotels with that hotel
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the hotels based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the nums most similar hotels (Ignoring the hotel itself)
    sim_scores = sim_scores[1:nums+1]

    # Get the hotel indices
    hotel_indices = [i[0] for i in sim_scores]

    # Return the top n most similar hotels as a DataFrame
    return df.iloc[hotel_indices]

# Define function to display recommended hotels
def display_recommended_hotels(recommendations):
    if recommendations is not None and not recommendations.empty:
        st.write("Top Recommendations:")
        for i, row in recommendations.iterrows():
            st.write(f"Hotel ID: {row['Hotel_ID']}, Hotel Name: {row['Hotel_Name']}")
            if st.button(f"Chi tiết tại đây {row['Hotel_ID']}", key=f"details-{i}"):
                st.session_state.selected_hotel_id = row['Hotel_ID']
                st.session_state.show_details = True
                st.experimental_rerun()
    else:
        st.write("No recommendations available.")

# Initialize 'random_hotels' if not already in session state
if 'random_hotels' not in st.session_state:
    st.session_state.random_hotels = None

# Main logic
if hotel_id:
    recommendations = get_recommendations(df_hotels, hotel_id, cosine_sim, nums=5)
    display_recommended_hotels(recommendations)

# Show details for selected hotel
if st.session_state.show_details and st.session_state.selected_hotel_id:
    show_details(st.session_state.selected_hotel_id)



# Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
if 'selected_hotel_id' not in st.session_state:
    # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
    st.session_state.selected_hotel_id = None

# Theo cách cho người dùng chọn khách sạn từ dropdown
# Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.random_hotels.iterrows()]
st.session_state.random_hotels
# Tạo một dropdown với options là các tuple này
selected_hotel = st.selectbox(
    "Chọn khách sạn",
    options=hotel_options,
    format_func=lambda x: x[0]  # Hiển thị tên khách sạn
)

# Cập nhật session_state dựa trên lựa chọn hiện tại
st.session_state.selected_hotel_id = selected_hotel[1]

if st.session_state.selected_hotel_id:
    # Hiển thị thông tin khách sạn được chọn
    selected_hotel = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]

    if not selected_hotel.empty:
        st.write('#### Bạn vừa chọn:')
        st.write('### ', selected_hotel['Hotel_Name'].values[0])

        hotel_description = selected_hotel['Hotel_Description'].values[0]
        truncated_description = ' '.join(hotel_description.split()[:100])
        st.write('##### Thông tin:')
        st.write(truncated_description, '...')

        st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
        recommendations = get_recommendations(df_hotels, st.session_state.selected_hotel_id, cosine_sim=cosine_sim, nums=5) 
        display_recommended_hotels(recommendations, cols=3)
    else:
        st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")
