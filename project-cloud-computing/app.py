from keras.models import load_model, save_model
import streamlit as st 
from streamlit_option_menu import option_menu
import numpy as np

# load models

price_sim_models = load_model("group3_model.h5")

# sidebar

with st.sidebar:
    selected = option_menu("Predict Systems",
                           ["Predict Sim Price"],
                           icons=["sim-fill"],
                           default_index=0)

if (selected == "Predict Sim Price"):
    st.title("Predict Sim Price using DL")

    phone_number = st.text_input("Number of your")
    digits = [int(d) for d in str(phone_number)]
    digits = digits[1:10]
    phone_number_array = np.array(digits)
    phone_number_array = np.reshape(phone_number_array, (-1, 9, 1))

    dianogis = ''

    if st.button("Price Predict Sim"):
        results = price_sim_models.predict(phone_number_array)
        x = np.argmax(results)
        
        if x == 0:
            dianogis = "giá khoảng từ 0 -> 500.000"
        if x == 1:
            dianogis = "giá khoảng từ 500.000 -> 700.000"
        if x == 2:
            dianogis = "giá khoảng từ 700.000 -> 900.000"
        if x == 3:
            dianogis = "giá khoảng từ 900.000 -> 1.000.000"
        if x == 4:
            dianogis = "giá khoảng từ 1.000.000 -> 1.200.000"
        if x == 5:
            dianogis = "giá khoảng từ 1.200.000 -> 1.500.000"
        if x == 6:
            dianogis = "giá khoảng từ 1.500.000 -> 3.000.000"
        if x == 7:
            dianogis = "giá khoảng từ 3.000.000 -> 6.000.000"
        if x == 8:
            dianogis = "giá lớn hơn 6.000.000"

    st.success(dianogis)



