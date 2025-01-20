import streamlit as st
import pickle
import numpy as np

# Load data and model
df = pickle.load(open('df.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Set up the page title and layout
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")
st.title("💻 Laptop Price Predictor")

# Organize inputs into two columns
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('💼 Brand', df['Company'].unique(), help="Select the laptop brand.")
    type = st.selectbox('🖥️ Type', df['TypeName'].unique(), help="Select the type of laptop.")
    ram = st.selectbox('🛠️ RAM (GB)', df['Ram'].unique(), help="Select the amount of RAM.")
    weight = st.number_input('⚖️ Weight (kg)', min_value=0.5, max_value=5.0, step=0.1, help="Enter the weight of the laptop.")

with col2:
    touchscreen = st.selectbox('🖐️ TouchScreen', ['No', 'Yes'], help="Does the laptop have a touchscreen?")
    ips = st.selectbox('📺 IPS', ['No', 'Yes'], help="Does the laptop have an IPS display?")
    screen_size = st.slider('📏 Screen Size (inches)', min_value=10.0, max_value=18.0, step=0.1, help="Select the screen size.")
    resolution = st.selectbox('🖼️ Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160',
        '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ], help="Select the screen resolution.")

# Additional inputs
cpu = st.selectbox('🧠 CPU', df['CPU_brand'].unique(), help="Select the CPU brand.")
hdd = st.selectbox('📂 HDD (GB)', [0, 128, 256, 512, 1024, 2048], help="Select the HDD capacity.")
ssd = st.selectbox('💾 SSD (GB)', [0, 128, 256, 512, 1024, 2048], help="Select the SSD capacity.")
gpu = st.selectbox('🎮 GPU Brand', df['Gpu_brand'].unique(), help="Select the GPU brand.")
os = st.selectbox('🖥️ Operating System', df['OS'].unique(), help="Select the operating system.")

# Prediction logic
if st.button('💰 Predict Price'):
    with st.spinner('Predicting...'):
        # Process touchscreen and IPS inputs
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        # Calculate PPI
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        # Prepare input query
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, 12)

        # Predict and display the price
        predicted_price = np.exp(pipe.predict(query)[0])  # Reverse log transformation
        st.success(f"The predicted price of the laptop is **${int(predicted_price):,}**.")
