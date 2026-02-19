import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Set page configuration
st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide"
)

st.title('📱 Mobile Phone Price Predictor')
st.write('Enter the phone specifications below to get an estimated price.')
st.markdown("---")

# Initialize session state for reset 
#  used to track if the reset button was clicked and trigger a rerun to reset all values
if 'reset_clicked' not in st.session_state:
    st.session_state.reset_clicked = False

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "./notebook/models/model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("❌ Model file not found at ./notebook/models/model.pkl")
        return None

model = load_model()

if model is not None:
    # Define dropdown options with units
    weight_options = {f"{x} g": x for x in range(100, 251, 10)}
    screen_size_options = {f"{round(x, 1)}\"": round(x, 1) for x in np.arange(4.0, 7.1, 0.5)}
    ppi_options = {f"{x} PPI": x for x in range(200, 601, 50)}
    cpu_core_options = {f"{x} cores": x for x in range(2, 13)}
    cpu_freq_options = {f"{round(x, 1)} GHz": round(x, 1) for x in np.arange(1.0, 3.6, 0.3)}
    internal_mem_options = {f"{x} GB": x for x in [16, 32, 64, 128, 256, 512]}
    ram_options = {f"{x} GB": x for x in [1, 2, 3, 4, 6, 8, 10, 12, 16]}
    rear_cam_options = {f"{x} MP": x for x in [5, 12, 16, 20, 25, 30, 40, 48, 50, 64, 100, 108, 200]}
    front_cam_options = {f"{x} MP": x for x in range(2, 61, 2)}
    battery_options = {f"{x} mAh": x for x in range(2000, 6001, 250)}
    thickness_options = {f"{round(x, 1)} mm": round(x, 1) for x in np.arange(7.0, 12.1, 0.5)}
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Phone Specifications")
        weight_display = st.selectbox('Weight', list(weight_options.keys()), index=8, key='weight')
        weight = weight_options[weight_display]
        
        screen_display = st.selectbox('Screen Size', list(screen_size_options.keys()), index=3, key='screen')
        screen_size = screen_size_options[screen_display]
        
        ppi_display = st.selectbox('Pixels Per Inch', list(ppi_options.keys()), index=4, key='ppi')
        ppi = ppi_options[ppi_display]
        
        cpu_core_display = st.selectbox('CPU Cores', list(cpu_core_options.keys()), index=6, key='cpu_core')
        cpu_core = cpu_core_options[cpu_core_display]
        
        cpu_freq_display = st.selectbox('CPU Frequency', list(cpu_freq_options.keys()), index=6, key='cpu_freq')
        cpu_freq = cpu_freq_options[cpu_freq_display]
    
    with col2:
        st.subheader("💾 Memory & Camera")
        internal_display = st.selectbox('Internal Memory', list(internal_mem_options.keys()), index=3, key='internal')
        internal_mem = internal_mem_options[internal_display]
        
        ram_display = st.selectbox('RAM', list(ram_options.keys()), index=4, key='ram')
        ram = ram_options[ram_display]
        
        rear_display = st.selectbox('Rear Camera', list(rear_cam_options.keys()), index=8, key='rear')
        rear_cam = rear_cam_options[rear_display]
        
        front_display = st.selectbox('Front Camera', list(front_cam_options.keys()), index=7, key='front')
        front_cam = front_cam_options[front_display]
        
        battery_display = st.selectbox('Battery Capacity', list(battery_options.keys()), index=9, key='battery')
        battery = battery_options[battery_display]
        
        thickness_display = st.selectbox('Thickness', list(thickness_options.keys()), index=2, key='thickness')
        thickness = thickness_options[thickness_display]

    # Feature names in correct order
    feature_names = ['weight', 'screen_size', 'ppi', 'cpu core', 'cpu freq', 
                     'internal mem', 'ram', 'RearCam', 'Front_Cam', 'battery', 'thickness']
    
    # Create input array
    input_data = np.array([[
        weight, screen_size, ppi, cpu_core, cpu_freq,
        internal_mem, ram, rear_cam, front_cam, battery, thickness
    ]])
    
    # Prediction and Reset buttons
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn1:
        if st.button('🔄 Reset All', use_container_width=True):
            st.session_state.reset_clicked = True
            st.rerun()
    
    with col_btn2:
        predict_clicked = st.button('🔮 Predict Price', use_container_width=True)
    
    with col_btn3:
        pass  # Empty for alignment
    
    # Handle reset
    if st.session_state.reset_clicked:
        st.session_state.reset_clicked = False
        st.info("✅ All values have been reset to defaults. Adjust specificaitons above.")
        st.stop()
    
    # Prediction logic
    if predict_clicked:
        prediction = model.predict(input_data)[0]
        st.success(f"The predicted price is: **Rs.{prediction:,.2f}**")
        
        # Display specs summary
        st.subheader("📋 Specifications Summary")
        specs_summary = {
            'Weight': f"{weight} g",
            'Screen Size': f"{screen_size}\"",
            'PPI': f"{ppi}",
            'CPU Cores': f"{cpu_core}",
            'CPU Frequency': f"{cpu_freq} GHz",
            'Internal Memory': f"{internal_mem} GB",
            'RAM': f"{ram} GB",
            'Rear Camera': f"{rear_cam} MP",
            'Front Camera': f"{front_cam} MP",
            'Battery': f"{battery} mAh",
            'Thickness': f"{thickness} mm"
        }
        
        specs_df = pd.DataFrame(list(specs_summary.items()), columns=['Specification', 'Value'])
        st.table(specs_df)