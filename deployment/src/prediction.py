import streamlit as st  # type: ignore
import pandas as pd
import pickle
import os
import numpy as np

def predict():
    model_path = os.path.join(os.path.dirname(__file__), "..", "src/AttritionBoosting.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    st.title("**Prediksi Attrition Karyawan**")
    st.markdown("---")

    st.markdown("### Menggunakan Machine Learning untuk Menilai Potensi Resign Karyawan")

    st.markdown("""
    _Perusahaan selalu berusaha mempertahankan karyawan terbaik mereka. Model ini membantu memprediksi potensi resign karyawan agar perusahaan dapat mengambil tindakan pencegahan yang diperlukan._
    """)

    with st.form('attrition', clear_on_submit=True):
        st.subheader("Masukkan Data Karyawan:")

        gender = st.radio('Jenis Kelamin', ['Female', 'Male'], horizontal=True, key="gender")
        age = st.slider('Masukkan Usia', 18, 65, 25, key="age")
        job_role = st.selectbox('Departemen Karyawan', ['Healthcare', 'Education', 'Media', 'Technology', 'Finance'], key="job_role")
        job_satisfaction = st.selectbox('Kepuasan Pekerjaan?', ['High', 'Very High', 'Medium', 'Low'], key="job_satisfaction")
        education_level = st.selectbox('Tingkat Pendidikan Terakhir', ["Master's Degree", 'Associate Degree', 'High School', 'Bachelor\'s Degree', 'PhD'], key="education_level")
        marital_status = st.selectbox('Status Pernikahan', ['Married', 'Single', 'Divorced'], key="marital_status")
        performance_rating = st.selectbox('Penilaian Kinerja Anda', ['Average', 'High', 'Below Average', 'Low'], key="performance_rating")
        work_life_balance = st.selectbox('Keseimbangan Kerja dan Kehidupan', ['Excellent', 'Good', 'Fair', 'Poor'], key="work_life_balance")
        number_of_promotions = st.number_input('Jumlah Promosi yang Diterima', 0, 5, 0, key="number_of_promotions")
        number_of_dependents = st.number_input('Jumlah Tanggungan', 0, 5, 0, key="number_of_dependents")
        job_level = st.selectbox('Tingkat Jabatan', ['Mid', 'Entry', 'Senior'], key="job_level")
        employee_recognition = st.selectbox('Penghargaan Karyawan', ['Medium', 'High', 'Low', 'Very High'], key="employee_recognition")

        remote_work = st.checkbox('Apakah Anda Bekerja Dari Rumah?', key="remote_work")
        overtime = st.checkbox('Apakah Anda Bekerja Lembur?', key="overtime")
        leadership_opportunities = st.checkbox('Kesempatan Kepemimpinan?', key="leadership_opportunities")
        innovation_opportunities = st.checkbox('Kesempatan Inovasi?', key="innovation_opportunities")
        company_tenure = st.slider('Masa Kerja di Perusahaan (tahun)', 1, 30, 5, key="company_tenure")
        monthly_income = st.slider('Pendapatan Bulanan Karyawan ($)', 2000, 15000, 5000, key="monthly_income")
        distance_from_home = st.slider('Jarak dari Rumah ke Kantor (mil)', 1, 100, 10, key="distance_from_home")
        years_at_company = st.slider('Jumlah Tahun di Perusahaan', 0, 40, 5, key="years_at_company")
        company_size = st.selectbox('Ukuran Perusahaan', ['Small', 'Medium', 'Large'], key="company_size")
        company_reputation = st.selectbox('Reputasi Perusahaan', ['Good', 'Poor', 'Fair', 'Excellent'], key="company_reputation")

        submitted = st.form_submit_button("Prediksi Attrition")

    if submitted:
        data_input = {
            'Age': age,
            'Gender': gender,
            'Job_Role': job_role,
            'Work_Life_Balance': work_life_balance,
            'Job_Satisfaction': job_satisfaction,
            'Performance_Rating': performance_rating,
            'Overtime': 'Yes' if overtime else 'No',
            'Number_of_Promotions': number_of_promotions,
            'Distance_from_Home': distance_from_home,
            'Education_Level': education_level,
            'Marital_Status': marital_status,
            'Number_of_Dependents': number_of_dependents,
            'Job_Level': job_level,
            'Company_Size': company_size,
            'Company_Tenure': company_tenure,
            'Remote_Work': 'Yes' if remote_work else 'No',
            'Leadership_Opportunities': 'Yes' if leadership_opportunities else 'No',
            'Innovation_Opportunities': 'Yes' if innovation_opportunities else 'No',
            'Company_Reputation': company_reputation,
            'Employee_Recognition': employee_recognition,
            'Monthly_Income': monthly_income,
            'Years_at_Company': years_at_company,
        }

        data = pd.DataFrame([data_input])
        reccured = model.predict(data)
        proba = model.predict_proba(data)

        hasil = 'Keluar dari Perusahaan' if reccured[0] == 1 else 'Tetap Bekerja di Perusahaan'
        probabilitas = proba[0][1] if reccured[0] == 1 else proba[0][0]

        warna = "#ffe6e6" if hasil == "Keluar dari Perusahaan" else "#e6ffed"
        garis = "red" if hasil == "Keluar dari Perusahaan" else "green"
        teks = "red" if hasil == "Keluar dari Perusahaan" else "green"

        st.markdown(f"""
        <div style="padding: 1rem; background-color: {warna}; border-left: 5px solid {garis}; border-radius: 5px;">
            <b style="color:{teks};">Hasil Prediksi Menyatakan Bahwa Karyawan: {hasil}</b>
            <p style="color:{teks};">Probabilitas: {probabilitas * 100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    predict()
