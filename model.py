import streamlit as st
import pickle
import os
from datetime import date
import numpy as np

# Define the options class
class options:
    country_values = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 44.8930221, 77.0, 78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]
    product_ref_values = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407, 164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642, 1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026, 1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]
    item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']
    item_type_dict = {'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}
    status_values = ['Lost', 'Won']
    application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
    status_dict = {'Lost':0, 'Won':1}

# Regression Model
def regression_model():
    st.header('Regression Model')
    
    item_date = st.date_input('Item Date', value=date(2020, 7, 1), min_value=date(2020, 7, 1), max_value=date(2021, 5, 31), key='reg_item_date')
    delivery_date = st.date_input('Delivery Date', value=date(2020, 8, 1), min_value=date(2020, 8, 1), max_value=date(2022, 2, 28), key='reg_delivery_date')

    quantity_tons = st.text_input('Quantity Tons (Min: 0.00001 & Max: 1000000000)', value='1.0', key='reg_quantity_tons')
    quantity_tons_log = np.log(float(quantity_tons)) if quantity_tons else 0.0
    country = st.selectbox('Country', options.country_values, key='reg_country')
    item_type = st.selectbox('Item Type', options.item_type_values, key='reg_item_type')
    thickness = st.number_input('Thickness', min_value=0.1, max_value=2500000.0, value=1.0, key='reg_thickness')
    thickness_log = np.log(thickness)
    product_ref = st.selectbox('Product Ref', options.product_ref_values, key='reg_product_ref')
    customer = st.text_input('Customer ID (Min: 12458000 & Max: 2147484000)', value='12458000', key='reg_customer')
    application = st.selectbox('Application', options.application_values, key='reg_application')
    width = st.number_input('Width(Min=1.0, Max=2990000.0)', min_value=1.0, max_value=2990000.0, value=1.0, key='reg_width')
    width_log = np.log(width)
    status = st.selectbox(label='Status', options=options.status_values, key='reg_status')

    regression_model_path = 'd:\\project 6\\regression_model.pkl'
    with open(regression_model_path, 'rb') as f:
        model = pickle.load(f)

    input_data = np.array([[
        country,
        options.item_type_dict[item_type],
        customer,
        options.status_dict[status],
        application,
        width_log,
        product_ref,
        quantity_tons_log,
        thickness_log,
        item_date.day,
        item_date.month,
        item_date.year,
        delivery_date.day,
        delivery_date.month,
        delivery_date.year
    ]])

    if st.button('Predict Selling Price', key='reg_predict_button'):
        prediction = model.predict(input_data)
        selling_price = np.exp(prediction[0])
        selling_price = round(selling_price, 2)
        st.write(f'Predicted Selling Price: {selling_price}')

# Classification Model
def classification_model():
    st.header('Classification Model')
    
    item_date = st.date_input('Item Date', value=date(2020, 7, 1), min_value=date(2020, 7, 1), max_value=date(2021, 5, 31), key='class_item_date')
    delivery_date = st.date_input('Delivery Date', value=date(2020, 8, 1), min_value=date(2020, 8, 1), max_value=date(2022, 2, 28), key='class_delivery_date')

    quantity_tons = st.text_input('Quantity Tons (Min: 0.00001 & Max: 1000000000)', value='1.0', key='class_quantity_tons')
    quantity_tons_log = np.log(float(quantity_tons)) if quantity_tons else 0.0
    country = st.selectbox('Country', options.country_values, key='class_country')
    item_type = st.selectbox('Item Type', options.item_type_values, key='class_item_type')
    thickness = st.number_input('Thickness', min_value=0.1, max_value=2500000.0, value=1.0, key='class_thickness')
    thickness_log = np.log(thickness)
    product_ref = st.selectbox('Product Ref', options.product_ref_values, key='class_product_ref')
    customer = st.text_input('Customer ID (Min: 12458000 & Max: 2147484000)', value='12458000', key='class_customer')
    application = st.selectbox('Application', options.application_values, key='class_application')
    width = st.number_input('Width(Min=1.0, Max=2990000.0)', min_value=1.0, max_value=2990000.0, value=1.0, key='class_width')
    width_log = np.log(width)
    selling_price_log = st.text_input(label='Selling Price (Min: 0.1 & Max: 100001000)', key='class_selling_price')

    try:
        selling_price_log = np.log(float(selling_price_log)) if selling_price_log else 0.0
    except ValueError:
        st.error("Please enter a valid number for Selling Price.")
        return

    classification_model_path = 'd:\\project 6\\classification_model.pkl'
    with open(classification_model_path, 'rb') as f:
        model = pickle.load(f)

    input_data = np.array([[
        country,
        options.item_type_dict[item_type],
        customer,
        selling_price_log,
        application,
        width_log,
        product_ref,
        quantity_tons_log,
        thickness_log,
        item_date.day,
        item_date.month,
        item_date.year,
        delivery_date.day,
        delivery_date.month,
        delivery_date.year
    ]])

    if st.button('Predict Status', key='class_predict_button'):
        prediction = model.predict(input_data)[0]
        prediction_label = 'Won' if prediction == 1 else 'Lost'
        st.write(f'Predicted Status: {prediction_label}')

# Define the Streamlit app with radio buttons for model selection
def app():
    st.title('Industrial Copper Modeling')
    
    model_choice = st.radio('Choose Model', ['Regression Model', 'Classification Model'])
    
    if model_choice == 'Regression Model':
        regression_model()
    elif model_choice == 'Classification Model':
        classification_model()

if __name__ == '__main__':
    app()
