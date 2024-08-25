import pandas as pd
import streamlit as st
import pickle
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from folium import Marker
import seaborn as sns
from matplotlib import pyplot as plt



def get_recommendations2(hotel_id, nums=5):
    # Ép kiểu dữ liệu thành chuỗi nếu không phải là chuỗi
    if not isinstance(hotel_id, str):
        hotel_id = str(hotel_id)
    # Kiểm tra xem hotel_id có trong dữ liệu không
    if hotel_id not in df_hotels['Hotel_ID'].values:
        # st.write("Hotel ID không tồn tại trong dữ liệu.")
        # print(df_hotels['Hotel_ID'].values)
        return pd.DataFrame()
    # Khởi tạo 'indices' để ánh xạ hotel_id với index
    indices = pd.Series(df_hotels.index, index=df_hotels['Hotel_ID']).to_dict()
    # Lấy chỉ số của khách sạn
    idx = indices[hotel_id]
    # Tính toán điểm tương đồng cho khách sạn cụ thể
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Loại bỏ khách sạn hiện tại (vì nó có điểm tương đồng là 1 với chính nó)
    sim_scores = sim_scores[1:nums+1]
    # Lấy các chỉ số của các khách sạn tương tự
    hotel_indices = [i[0] for i in sim_scores]
    # Trả về thông tin các khách sạn tương tự
    return df_hotels.iloc[hotel_indices][['Hotel_ID', 'Hotel_Description']]

def recommend_hotels_by_user(algo, user_id, hotel_ids, num_recommendations=5):
    # Dự đoán đánh giá cho tất cả các khách sạn
    predictions = [algo.predict(user_id, iid) for iid in hotel_ids]
    # Sắp xếp các dự đoán theo điểm số ước lượng (est) từ cao đến thấp
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:num_recommendations]
    # Trích xuất danh sách các khách sạn đề xuất
    recommended_hotels = [(pred.iid, pred.est) for pred in top_predictions]
    # Chuyển kết quả thành DataFrame
    recommended_hotels_df = pd.DataFrame(recommended_hotels, columns=['Hotel_ID', 'Predicted Rating'])
    # Kết hợp với DataFrame df_hotels để lấy Hotel_Name
    recommended_hotels_df = recommended_hotels_df.merge(df_hotels[['Hotel_ID', 'Hotel_Name']], on='Hotel_ID', how='left')
    return recommended_hotels_df

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
            more_recommendations = get_recommendations2(st.session_state.selected_hotel_id, nums=5)
            for j, more_row in more_recommendations.iterrows():
                st.write(f"Hotel ID: {more_row['Hotel_ID']}, Hotel Name: {more_row['Hotel_Name']}")
        else:
            st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")

def geocode_address(address, geolocator):
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except GeocoderTimedOut:
        return geocode_address(address, geolocator)

# Khởi tạo đối tượng Geolocator toàn cục
geolocator = Nominatim(user_agent="hotel_recommender") 
def get_lat_lon(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            st.error("Address not found.")
            return None, None
    except Exception as e:
        st.error(f"An error occurred during geocoding: {str(e)}")
        return None, None

def show_hotel_info(hotel_id):
    # Load your hotel data here
    df_hotels = pd.read_csv('hotel_info.csv')
    # Find the hotel by ID
    hotel = df_hotels[df_hotels['Hotel_ID'] == hotel_id]
    if not hotel.empty:
        hotel_address = hotel['Hotel_Address'].values[0]
        hotel_description = hotel['Hotel_Description'].values[0]
        hotel_rank = hotel['Hotel_Rank'].values[0]
        # Geocoding address
        geolocator = Nominatim(user_agent="hotel_recommender")
        latitude, longitude = geocode_address(hotel_address, geolocator)
        if latitude and longitude:
            # Create a map
            map_center = [latitude, longitude]
            m = folium.Map(location=map_center, zoom_start=14)
            folium.Marker(location=map_center, popup=hotel_address).add_to(m)
            # Display the map
            st.write("**Map:**")
            st.components.v1.html(m._repr_html_(), height=500)
            st.write(f"**Rank:** {hotel_rank}")
            st.write(f"**Address:** {hotel_address}")
            st.write(f"**Description:** {hotel_description}")
        else:
            # st.write("Location not found.")
            st.write(f"**Rank:** {hotel_rank}")
            st.write(f"**Address:** {hotel_address}")
            st.write(f"**Description:** {hotel_description}")
    else:
        st.write("Hotel ID not found in the dataset.")



df_hotels = pd.read_csv('Hotel_infos_process.csv')
data_hotel_comments = pd.read_csv('Hotel_comments_process_svd.csv')

# Load similarity matrices and models
with open(r'D:\Documents\TTTH - KHTN\Project\GUI\GUI_hotel_recommender system\cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

with open(r'D:\Documents\TTTH - KHTN\Project\GUI\GUI_hotel_recommender system\svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)



# Streamlit GUI
st.title("Hotel Recommendation System")
# Initialize session_state if not already set
if 'selected_hotel_id' not in st.session_state:
    st.session_state.selected_hotel_id = None
if 'show_details' not in st.session_state:
    st.session_state.show_details = False



def run_cosine_similarity_model(hotel_id, cosine_sim, df_hotels):
    if hotel_id:
        try:
            # Chuyển đổi hotel_id thành chuỗi để đảm bảo tính nhất quán
            hotel_id = str(hotel_id)
            # Kiểm tra sự tồn tại của hotel_id trong DataFrame
            if hotel_id in df_hotels['Hotel_ID'].values:
                # Tạo Series ánh xạ từ Hotel_ID sang index
                indices = pd.Series(df_hotels.index, index=df_hotels['Hotel_ID']).to_dict()
                # Lấy chỉ số của khách sạn
                idx = indices[hotel_id]
                # Tính toán điểm tương đồng cho khách sạn đã chọn
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                # Lấy top 5 khách sạn tương tự (bỏ qua khách sạn hiện tại)
                sim_scores = sim_scores[1:6]  # Lấy top 6 và bỏ qua khách sạn hiện tại
                # Trả về danh sách các chỉ số của khách sạn tương tự
                similar_indices = [i[0] for i in sim_scores]
                st.write(f"Các khách sạn tương tự với Hotel ID {hotel_id}:")
                # Hiển thị kết quả
                for index in similar_indices:
                    similar_hotel_id = df_hotels.iloc[index]['Hotel_ID']
                    similar_hotel_name = df_hotels.iloc[index]['Hotel_Name']
                    score = sim_scores[similar_indices.index(index)][1]
                    st.write(f"Hotel ID: {similar_hotel_id}, Hotel Name: {similar_hotel_name}")
            else:
                st.error(f"Hotel ID {hotel_id} không tồn tại trong dữ liệu.")
        except ValueError:
            st.error("Vui lòng nhập Hotel ID hợp lệ dưới dạng số hoặc chuỗi.")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {str(e)}")



# Create a menu to choose between the models
option = st.sidebar.selectbox(
    'Select a recommendation method:',
    ('Collaborative Filtering (SVD)', 'Content-Based (Cosine Similarity)',  'Information visualization')
)

# Execute the corresponding model based on user selection
if option == 'Collaborative Filtering (SVD)':
    st.write("Enter a User ID to get similar hotel recommendations.")
    user_id = st.text_input("User ID", "")
    # Lấy danh sách các ID khách sạn duy nhất từ cột Hotel_ID của df_hotels
    hotel_ids = df_hotels['Hotel_ID'].unique()
        # Khi nhấn "Get Recommendations"
    if st.button("Get Recommendations"):
        recommendations = recommend_hotels_by_user(svd_model, user_id, hotel_ids=hotel_ids, num_recommendations=5)
        if not recommendations.empty:
            st.write("Top Recommendations:")
            # Display both Hotel ID and Hotel Name
            for i, row in recommendations.iterrows():
                st.write(f"Hotel ID: {row['Hotel_ID']}, Hotel Name: {row['Hotel_Name']}")
        else:
            st.write("No recommendations available.")

    # Theo cách cho người dùng chọn khách sạn từ dropdown
    # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    st.session_state.random_hotels = df_hotels
    hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.random_hotels.iterrows()]

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
        else:
            st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")
    show_hotel_info(st.session_state.selected_hotel_id)
elif option == 'Content-Based (Cosine Similarity)':
    st.write("Enter a Hotel ID to get similar hotel recommendations.")
    hotel_id = st.text_input("Hotel ID", "")
    # Khi nhấn "Get Recommendations"
    if st.button("Get Recommendations"):
        if hotel_id:
            st.write("Top Recommendations:")
            run_cosine_similarity_model(hotel_id, cosine_sim, df_hotels)
            recommendations = get_recommendations2([hotel_id])
            if not recommendations.empty:
                
                st.write("No recommendations available.")
            else:
                # Hiển thị khách sạn gợi ý
                for i, row in recommendations.iterrows():
                    st.write(f"Hotel ID: {row['Hotel_ID']}, Hotel Name: {row['Hotel_Name']}")
        else:
            st.write("Please enter a Hotel ID.")
        # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    st.session_state.random_hotels = df_hotels
    hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.random_hotels.iterrows()]

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

        else:
            st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")

    show_hotel_info(st.session_state.selected_hotel_id)

elif option == 'Information visualization':
    # Chuyển đổi cột 'Review Date' thành datetime với định dạng ngày/tháng/năm
    monthly_counts = pd.to_datetime(data_hotel_comments['Review Date'], format='%d/%m/%Y').dt.to_period('M').value_counts().sort_index()
    monthly_counts = monthly_counts[monthly_counts > 400]
    # Tạo biểu đồ tần suất của các đánh giá theo tháng
    plt.figure(figsize=(14, 8))
    # Vẽ biểu đồ chính
    sns.lineplot(x=monthly_counts.index.astype(str), y=monthly_counts.values, marker='o')
    plt.title('Tần suất đánh giá theo tháng')
    plt.xlabel('Tháng')
    plt.ylabel('Số lượng đánh giá')
    # Đặt trục x
    ax = plt.gca()
    ax.set_xticks(range(len(monthly_counts.index)))
    ax.set_xticklabels([date.strftime('%m') for date in monthly_counts.index.to_timestamp()], rotation=0, ha='center')
    # Thêm hàng năm phía dưới trục x
    years = [date.year for date in monthly_counts.index.to_timestamp()]
    unique_years = sorted(set(years))
    year_labels = [''] * len(years)
    for year in unique_years:
        indices = [i for i, x in enumerate(years) if x == year]
        for index in indices:
            year_labels[index] = str(year)
    # Thêm dòng năm phía dưới
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(year_labels, rotation=0, ha='center')
    ax2.set_xlabel('Năm')
    # Sử dụng Streamlit để hiển thị biểu đồ
    st.pyplot(plt)
    # Reset lại figure để tránh xung đột với các biểu đồ khác
    plt.clf()

    # Nhóm dữ liệu theo 'Score Level' và đếm số lượng, sau đó vẽ biểu đồ
    ax = data_hotel_comments.groupby('Score Level').size().plot(kind='barh', color=sns.color_palette('Dark2'))
    # Tùy chỉnh lại biểu đồ để ẩn đi các đường viền không cần thiết
    plt.gca().spines[['top', 'right']].set_visible(False)
    # Sử dụng Streamlit để hiển thị biểu đồ
    st.pyplot(plt)
    # Reset lại figure để tránh xung đột với các biểu đồ khác
    plt.clf()




