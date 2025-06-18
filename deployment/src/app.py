import streamlit as st # type: ignore
import eda,prediction,home

st.set_page_config(page_title = "ANALISIS DAN PREDIKSI KARYAWAN YANG BERISIKO MENGUNDURKAN DIRI",
                   layout='centered',
                   initial_sidebar_state='expanded')

with st.sidebar:
    st.write('# Navigation')
    navigation = st.radio('Page', ['Home', 'Data Analysis', 'Predict Employee Attrition'])

if navigation == 'Data Analysis':
    eda.eda()
elif navigation == 'Home':
    home.home()
else:
    prediction.predict()