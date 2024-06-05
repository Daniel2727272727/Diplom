import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import joblib
from geopy.geocoders import Nominatim
import requests
import folium
from streamlit_folium import folium_static
import time
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
#from bd import save_in_data


# page_bg_img = '''
# <style>
# .stApp {
#     background-image: url("https://gas-kvas.com/uploads/posts/2023-02/1675497299_gas-kvas-com-p-fonovie-risunki-fotografii-18.jpg");  # Укажите правильный путь к изображению
#     background-size: cover;
# }
# </style>
# '''
# st.markdown(page_bg_img, unsafe_allow_html=True)

PATH_DATA = "data/moscow.csv"
PATH_UNIQUE_VALUES = 'data/unique_values.json'
PATH_MODEL = "models/lr_pipeline.sav"

@st.cache_data
def load_data(path):
    """Load data from path"""
    data = pd.read_csv(path)
    special_data = data[data['spec_id'] == 1]  # Извлекаем строку с специальным идентификатором
    rest_data = data[data['spec_id'] != 1]  # Остальные данные
    sample_size = min(4999, len(rest_data))  # Подгоняем размер выборки
    sampled_data = rest_data.sample(sample_size)  # Выбираем случайные данные
    final_data = pd.concat([special_data, sampled_data])  # Объединяем специальную строку с выборкой
    return final_data

@st.cache_data
def load_model(path):
    """Load model from path"""
    model = joblib.load(path)
    return model

@st.cache_data
def transform(data):
    """Transform data"""
    if data.empty:
        return data
    colors = sns.color_palette("coolwarm").as_hex()
    n_colors = len(colors)
    data["norm_price"] = data["price"] / data["area"]
    data["label_colors"] = pd.qcut(data["norm_price"], n_colors, labels=colors)
    data["label_colors"] = data["label_colors"].astype("str")
    return data

with open(PATH_UNIQUE_VALUES) as file:
    dict_unique = json.load(file)

df = load_data(PATH_DATA)
df = transform(df)
if df.empty:
    st.error("Нет данных для отображения.")

# Загрузка модели
model = load_model(PATH_MODEL)  # Здесь загружаем и сохраняем модель

# Переключение языка

def switch_language(language):
    if language not in ['EN', 'RU']:
        return
    st.session_state.language = language

# Размещение кнопок переключения языка в боковой панели
with st.sidebar:
    st.write("## Смена языка")
    col1, col2 = st.columns(2)
    with col1:
        st.button("EN", on_click=switch_language, args=("EN",))
    with col2:
        st.button("RU", on_click=switch_language, args=("RU",))
    # with col3:
    #     st.button("CN", on_click=switch_language, args=("CN",))

# # Вывод текста в зависимости от выбранного языка
# if st.session_state.language == "EN":
#     st.sidebar.write("Language is set to English.")
# elif st.session_state.language == "ZH":
#     st.sidebar.write("Language is set to Chinese.")
# else:
#     st.sidebar.write("Язык установлен на русский.")

if 'language' not in st.session_state:
    st.session_state.language = "RU"


# Вывод информации на выбранном языке
if st.session_state.language == "EN":
    st.header('House prices in Moscow')
    description = """
    ### Field Description
        - Building type: 1 - Panel, 2 - Monolithic, 3 - Brick
        - Property type: 1 - Secondary market, 11 - New construction
        - Floor: The floor on which the apartment is located
        - Levels: Total number of floors in the building
        - Rooms: Number of living rooms. If '-1', it means 'studio'
        - Area: Total area of the apartment in square meters
        - Kitchen area: Kitchen area in square meters
        - Price: Price in rubles
    """
    filter_labels = {
        'building_type': 'Building type',
        'object_type': 'Property type',
        'level': 'Floor',
        'levels': 'Total floors',
        'rooms': 'Rooms',
        'area': 'Area',
        'kitchen_area': 'Kitchen area',
        'address': 'Enter property address:',
        'predict_button': 'Predict'
    }
# elif st.session_state.language == "CN":
#     st.header('莫斯科房价')
#     description = """
#     ### 字段描述
#         - 建筑类型：0 - 其他，1 - 板块，2 - 整体，3 - 砖，4 - 块，5 - 木
#         - 物业类型：1 - 二手市场，11 - 新建
#         - 楼层：公寓所在的楼层
#         - 层数：建筑物的总层数
#         - 房间：生活室的数量。如果为'-1'，则表示 '一室户'
#         - 面积：公寓的总面积（平方米）
#         - 厨房面积：厨房面积（平方米）
#         - 价格：价格（卢布）
#     """
else:
    st.header('Цены на жилье в Москве')
    description = """
    ### Описание полей
        - Тип здания:  1 - Панельный, 2 - Монолитный, 3 - Кирпичный
        - Тип объекта: 1 - Вторичка, 11 - Новостройка
        - Этаж: Этаж, на котором находится квартира
        - Уровни: Общее количество этажей в здании
        - Комнаты: Количество жилых комнат. Если '-1', значит 'студия'
        - Площадь: Общая площадь квартиры в квадратных метрах
        - Площадь кухни: Площадь кухни в квадратных метрах
        - Цена: Цена в рублях
    """

    filter_labels = {
        'building_type': 'Тип здания',
        'object_type': 'Тип объекта',
        'level': 'Этаж',
        'levels': 'Общее количество этажей',
        'rooms': 'Комнаты',
        'area': 'Площадь',
        'kitchen_area': 'Площадь кухни',
        'address': 'Введите адрес недвижимости:',
        'predict_button': 'Предсказать'
    }

st.markdown(description)

# Словари для перевода типов домов и объектов недвижимости
building_types = {
    1: "Панельный",
    2: "Монолитный",
    3: "Кирпичный"
}

object_types = {
    1: "Вторичный рынок",
    11: "Новостройка"
}

building_keys = list(building_types.keys())
building_values = list(building_types.values())

object_keys = list(object_types.keys())
object_values = list(object_types.values())

@st.cache_data
def create_map(df):
    m = folium.Map(location=[55.7522, 37.6156], zoom_start=10, tiles='cartodbdark_matter')
    for idx, row in df.iterrows():
        # Формируем основной контент для попапа
        popup_content = f"""
        <div style="font-family: Arial, sans-serif; font-size:12px; color: darkslategray;">
            <b>Цена:</b> {row['price']} руб.<br>
            <b>Площадь:</b> {row['area']} м²<br>
            <b>Комнаты:</b> {row['rooms']}<br>
            <b>Этаж:</b> {row['level']}/{row['levels']}<br>
            <b>Тип дома:</b> {building_types.get(row['building_type'], 'Неопределено')}<br>
            <b>Тип объекта:</b> {object_types.get(row['object_type'], 'Неопределено')}
        </div>"""

        # Добавляем изображение, если оно есть
        if 'image_url' in row and row['image_url']:
            popup_content += f"<img src='{row['image_url']}' style='width:100%; max-height:100px; margin-top:5px;'>"

        popup = folium.Popup(popup_content, max_width=250)
        folium.CircleMarker(
            location=[row['geo_lat'], row['geo_lon']],
            radius=3,
            color=row['label_colors'],
            fill=True,
            fill_color=row['label_colors'],
            popup=popup
        ).add_to(m)
    return m


#font_path = 'data/Noto_Sans_TC/static/NotoSansTC-Regular.ttf'
font_path = 'data/Noto_Sans_Mono/static/NotoSansMono-Regular.ttf'
prop = FontProperties(fname=font_path)

def display_color_gradient():
    # Настройка названия и подписей в зависимости от языка
    if st.session_state.language == "EN":
        gradient = np.linspace(0, 1, 256).reshape(256, 1)
        fig, ax = plt.subplots(figsize=(0.2, 0.5))  # Уменьшенный размер блока
        ax.imshow(gradient, aspect='auto', cmap="coolwarm_r")

        title = 'Price m² ₽'
        labels = ['High', 'Low']

        ax.set_title(title, fontsize=4, fontweight='bold', color='white')
        ax.set_xticks([])
        ax.set_yticks([0, 255])
        ax.set_yticklabels(labels, fontdict={'fontsize': 4, 'fontweight': 'bold', 'color': 'white' })
        fig.patch.set_facecolor('#00000000')  # Прозрачный фон
        fig.subplots_adjust(top=0.85)  # Регулируем верхний отступ
        fig.tight_layout(pad=0.5)  
        st.pyplot(fig)
    # elif st.session_state.language == "CN":
    #     title = '每平米价格 ₽'
    #     labels = ['高', '低']
    else:
        gradient = np.linspace(0, 1, 256).reshape(256, 1)
        fig, ax = plt.subplots(figsize=(0.3, 0.8))  # Уменьшенный размер блока
        ax.imshow(gradient, aspect='auto', cmap="coolwarm_r")

        title = 'Цена за м² ₽'
        labels = ['Высокая', 'Низкая']

        ax.set_title(title, fontsize=5, fontweight='bold', color='white')
        ax.set_xticks([])
        ax.set_yticks([0, 255])
        ax.set_yticklabels(labels, fontdict={'fontsize': 5, 'fontweight': 'bold', 'color': 'white'})
        fig.patch.set_facecolor('#00000000')  # Прозрачный фон
        fig.subplots_adjust(top=0.85)  # Регулируем верхний отступ
        fig.tight_layout(pad=1.0)  
        st.pyplot(fig)
        


if not df.empty:
    col1, col2 = st.columns([3, 1])  # Указана пропорция колонок
    with col1:
        map_data = create_map(df)  # Создание карты
        folium_static(map_data)  # Отображение карты в Streamlit
        
    with col2:
        display_color_gradient() 
    
    
    button_css = """
    <style>
    div.stButton > button:first-child {
        width: 70%;
        margin: 10px auto;
        display: block;
        font-size: 20px;
        color: white;
        background-color: #696969;
    }
    </style>
    """

    st.markdown(button_css, unsafe_allow_html=True) 


if 'building_type' not in st.session_state:
    st.session_state.building_type = dict_unique['building_type'][0]

if 'object_type' not in st.session_state:
    st.session_state.object_type = dict_unique["object_type"][0]

if 'level' not in st.session_state:
    st.session_state.level = min(dict_unique["level"])

if 'levels' not in st.session_state:
    st.session_state.levels = min(dict_unique["levels"])

if 'rooms' not in st.session_state:
    st.session_state.rooms = dict_unique["rooms"][0]

if 'area' not in st.session_state:
    st.session_state.area = min(dict_unique["area"])

if 'kitchen_area' not in st.session_state:
    st.session_state.kitchen_area = min(dict_unique["kitchen_area"])


# Sidebar filters
# building_type = st.sidebar.selectbox(filter_labels['building_type'], (dict_unique['building_type']))
# object_type = st.sidebar.selectbox(filter_labels["object_type"], (dict_unique["object_type"]))
# level = st.sidebar.slider(filter_labels["level"], min_value=min(dict_unique["level"]), max_value=max(dict_unique["level"]))
# levels = st.sidebar.slider(filter_labels["levels"], min_value=min(dict_unique["levels"]), max_value=max(dict_unique["levels"]))
# rooms = st.sidebar.selectbox(filter_labels["rooms"], (dict_unique["rooms"]))
# area = st.sidebar.slider(filter_labels["area"], min_value=min(dict_unique["area"]), max_value=max(dict_unique["area"]))
# kitchen_area = st.sidebar.slider(filter_labels["kitchen_area"], min_value=min(dict_unique["kitchen_area"]), max_value=max(dict_unique["kitchen_area"]))

# Вывод selectbox с названиями типов зданий и выбранным значением
st.session_state.building_type = st.sidebar.selectbox(
    filter_labels['building_type'],
    options=building_values,  # Отображаем значения
    index=building_keys.index(st.session_state.building_type) if st.session_state.building_type in building_keys else 0
)

# Вывод selectbox с названиями типов объектов и выбранным значением
st.session_state.object_type = st.sidebar.selectbox(
    filter_labels["object_type"],
    options=object_values,  # Отображаем значения
    index=object_keys.index(st.session_state.object_type) if st.session_state.object_type in object_keys else 0 # Индекс для сохранения текущего выбранного значения
)

st.session_state.level = st.sidebar.slider(
    filter_labels["level"],
    min_value=min(dict_unique["level"]),
    max_value=max(dict_unique["level"]),
    value=st.session_state.level
)

st.session_state.levels = st.sidebar.slider(
    filter_labels["levels"],
    min_value=min(dict_unique["levels"]),
    max_value=max(dict_unique["levels"]),
    value=st.session_state.levels
)

st.session_state.rooms = st.sidebar.selectbox(
    filter_labels["rooms"],
    dict_unique["rooms"],
    index=dict_unique["rooms"].index(st.session_state.rooms)
)

st.session_state.area = st.sidebar.slider(
    filter_labels["area"],
    min_value=int(min(dict_unique['area'])),   # Преобразуем к int
    max_value=int(max(dict_unique['area'])),   # Преобразуем к int
    value=int(st.session_state.area),          # Текущее значение тоже преобразуем к int
    step=1                                     # Если используем целые числа, шаг тоже должен быть int
)

st.session_state.kitchen_area = st.sidebar.slider(
    filter_labels["kitchen_area"],
    min_value=min(dict_unique["kitchen_area"]),
    max_value=max(dict_unique["kitchen_area"]),
    value=st.session_state.kitchen_area
)

address = st.sidebar.text_input(filter_labels['address'], '')
button = st.button(filter_labels["predict_button"])

@st.cache_data
def get_location(address):
    """Get location information using geocoding"""
    geolocator = Nominatim(user_agent="geoapiExercises")
    return geolocator.geocode(address)


if button and address:
    if not address.lower().startswith('москва'):
        st.error("Пожалуйста, введите адрес в Москве.")
    else:
        location = get_location(address)
        if location:
            st.write(f"Адрес: {location.address}")
            st.write(f"Координаты: {location.latitude}, {location.longitude}")

            # Создание карты с использованием координат искомого адреса
            map = folium.Map(location=[location.latitude, location.longitude], tiles='cartodbdark_matter', zoom_start=14)
            # Добавление маркера для искомого адреса
            folium.Marker([location.latitude, location.longitude], popup=location.address).add_to(map)
            # Показ карты в Streamlit
            folium_static(map)

            # Используем значения из st.session_state
            dict_data = {
                "building_type": st.session_state.building_type,  #0
                "object_type": st.session_state.object_type,      #1
                "level": st.session_state.level,                  #2
                "levels": st.session_state.levels,                #3
                "rooms": st.session_state.rooms,                  #4
                "area": st.session_state.area,                    #5
                "kitchen_area": st.session_state.kitchen_area,    #6
                "geo_lat": location.latitude,                     #7
                "geo_lon": location.longitude                     #8
            }
            data_predict = pd.DataFrame([dict_data])
            print(data_predict)
            print(data_predict.columns)
            output = model.predict(data_predict)
            #save_in_data(data_predict, output)

            st.success(f"Предполагаемая стоимость: {round(output[0])} руб.")
        else:
            st.error("Не удалось найти введенный адрес.")
