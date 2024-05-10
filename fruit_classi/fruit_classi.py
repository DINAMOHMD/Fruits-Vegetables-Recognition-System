import streamlit as st
import tensorflow as tf
import numpy as np
import json
import requests
from streamlit_lottie import st_lottie 


def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()


# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("my_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(100, 100))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    input_arr = input_arr/255.
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element


# Sidebar
st.sidebar.title("Fruits Dashboard")

app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if (app_mode == 'Home'):
    st.header('FRUITS & VEGETABLES RECOGNITION SYSTEM')
    lottie_url = 'https://lottie.host/2d7158e2-719e-464d-b913-30330a0bd604/oH2vBjwOLz.json'
    lottie_j = load_lottieurl(lottie_url)
    st_lottie(lottie_j,height=300)
    image_path = "home_img.jpg"
    st.image(image_path)

# About project
if (app_mode == "About Project"):
    st.header("About Project")
    st.subheader('About Dataset 📊')
    st.text("This dataset Contains images of the following food items:")
    st.text(
        "Apples (different varieties: Crimson Snow, Golden, Golden-Red, Granny Smith, Pink Lady, Red, Red Delicious), Apricot, Avocado, Avocado,")
    st.text(
        "ripe, Banana (Yellow, Red, Lady Finger), Beetroot Red, Blueberry, Cactus fruit, Cantaloupe (2 varieties), Carambula, Cauliflower, Cherry,")
    st.text(
        "(different varieties, Rainier), Cherry Wax (Yellow, Red, Black), Chestnut, Clementine, Cocos, Corn (with husk), Cucumber (ripened), Dates,")
    st.text(
        "Eggplant, Fig, Ginger Root, Granadilla, Grape (Blue, Pink, White (different varieties)), Grapefruit (Pink, White), Guava, Hazelnut,")
    st.text(
        "Huckleberry, Kiwi, Kaki, Kohlrabi, Kumsquats, Lemon (normal, Meyer), Lime, Lychee, Mandarine, Mango (Green, Red), Mangostan,")
    st.text(
        "Maracuja, Melon Piel de Sapo, Mulberry, Nectarine (Regular, Flat), Nut (Forest, Pecan), Onion (Red, White), Orange, Papaya, Passion fruit,")
    st.text(
        "Peach (different varieties), Pepino, Pear (different varieties, Abate, Forelle, Kaiser, Monster, Red, Stone, Williams), Pepper (Red, Green,")
    st.text(
        "Orange, Yellow), Physalis (normal, with Husk), Pineapple (normal, Mini), Pitahaya Red, Plum (different varieties), Pomegranate, Pomelo")
    st.text(
        "Sweetie, Potato (Red, Sweet, White), Quince, Rambutan, Raspberry, Redcurrant, Salak, Strawberry (normal, Wedge), Tamarillo, Tangelo,")
    st.text("Tomato (different varieties, Maroon, Cherry Red, Yellow, not ripened, Heart), Walnut, Watermelon.")
    lottie_url = 'https://lottie.host/65d3104d-9ea2-4805-9239-b86bfff910b5/43MfWr51fn.json'
    lottie_j = load_lottieurl(lottie_url)
    st_lottie(lottie_j,height=300)

# About Prediction
elif (app_mode == "Prediction"):
    st.header("🤖Model Prediction")
    test_image = st.file_uploader("💻Choose an image:")
    if (st.button("Show Image")):
        st.image(test_image, width=4, use_column_width=True)

        # Predict Button
    if (st.button("Predict")):
        st.balloons()
        st.write("🎉Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success("Model is Predicting it's a {}".format(label[result_index]))
