
import streamlit as st
import pandas as pd
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


# Streamlit GUI
st.title("Hotel Recommendation System")
st.write("Enter a Hotel ID to get similar hotel recommendations.")

hotel_id = st.text_input("Hotel ID", "")

if st.button("Get Recommendations"):
    if hotel_id:
        recommendations = get_recommendations(hotel_id)
        if not recommendations.empty:
            st.write("Top Recommendations:")
            for i, row in recommendations.iterrows():
                st.write(f"Hotel ID: {row['Hotel_ID']}, Hotel Name: {row['Hotel_Name']}")
                
                if st.button(f"View Details for {row['Hotel_Name']}", key=i):
                    st.write(f"Hotel Details:\nLocation: {row['Location']}\nDescription: {row['Description']}\nRating: {row['Rating']}")
        else:
            st.write("No recommendations available.")
    else:
        st.write("Please enter a Hotel ID.")

# Đọc dữ liệu khách sạn
data_hotel_comments = pd.read_csv('hotel_info.csv')
# Lấy 10 khách sạn
random_hotels = df_hotels.head(n=5)
print(random_hotels)

st.session_state.random_hotels = df_hotels

# Open and read file to cosine_sim_new
with open('word2vec_model.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)


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
# Display the selected hotel
st.write("Bạn đã chọn:", selected_hotel)

# Cập nhật session_state dựa trên lựa chọn hiện tại
st.session_state.selected_hotel_id = selected_hotel[1]

if st.session_state.selected_hotel_id:
    st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
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
        recommendations = get_recommendations(df_hotels, st.session_state.selected_hotel_id, cosine_sim=cosine_sim_new, nums=3) 
        display_recommended_hotels(recommendations, cols=3)
    else:
        st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")


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


