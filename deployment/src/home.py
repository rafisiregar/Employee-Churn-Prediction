import pandas as pd
import streamlit as st # type: ignore

def home():    
    # judul
    st.title("Deteksi Potensi Resign Karyawan Menggunakan Machine Learning")
    st.markdown("---")
    
    # deskripsi
    st.markdown("## Latar Belakang")
    st.markdown(''' 
    Attrition atau pengunduran diri karyawan merupakan tantangan besar bagi banyak perusahaan. Hal ini dapat meningkatkan biaya rekrutmen, hilangnya pengetahuan dan pengalaman, serta dapat menurunkan moral karyawan yang masih bertahan. Oleh karena itu, sangat penting untuk memprediksi potensi resign ini, agar perusahaan dapat melakukan langkah-langkah pencegahan yang tepat, seperti meningkatkan kebijakan retensi dan kepuasan kerja.
    
    Dalam proyek ini, saya sebagai seorang data scientist bertugas untuk membuat model machine learning yang dapat memprediksi apakah seorang karyawan berisiko untuk resign berdasarkan data historis yang ada di perusahaan. Model ini akan digunakan untuk mengidentifikasi karyawan yang berisiko resign dan membantu perusahaan untuk meningkatkan retensi mereka.
    ''')
    
    st.markdown("## Rumusan Masalah")
    st.markdown('''           
    Dalam waktu tiga bulan ini, saya harus mengembangkan sebuah model machine learning untuk memprediksi potensi resign karyawan berdasarkan data historis yang ada. Fokus utama adalah pada nilai recall yang tinggi untuk memastikan akurasi dalam memprediksi karyawan yang berisiko resign, dengan tingkat kepercayaan 99%. Hal ini sangat penting karena dalam konteks dunia kerja, akurasi tinggi sangat dibutuhkan agar perusahaan bisa mengambil langkah yang tepat.
    ''')
    
    st.markdown("---")

    # dataset
    st.markdown("## Dataset")
    data_karyawan = pd.read_csv('src/P1M2_rafi_siregar.csv')  # Sesuaikan nama file dataset
    st.dataframe(data_karyawan.head(5))

    st.markdown("---")

    # data overview
    st.markdown("## Data Overview", unsafe_allow_html=True)

    st.markdown("""
        <table>
        <thead>
        <tr><th>Kolom</th><th>Penjelasan</th></tr>
        </thead>
        <tbody>
        <tr><td>Age</td><td>Usia karyawan dalam tahun.</td></tr>
        <tr><td>Gender</td><td>Jenis kelamin karyawan: <code>Male</code> (Laki-laki), <code>Female</code> (Perempuan).</td></tr>
        <tr><td>Years at Company</td><td>Jumlah tahun yang telah dihabiskan karyawan di perusahaan.</td></tr>
        <tr><td>Monthly Income</td><td>Pendapatan bulanan karyawan dalam dolar.</td></tr>
        <tr><td>Job Role</td><td>Peran atau jabatan yang dijalani karyawan dalam perusahaan.</td></tr>
        <tr><td>Work-Life Balance</td><td>Penilaian terhadap keseimbangan kerja dan kehidupan pribadi karyawan.</td></tr>
        <tr><td>Job Satisfaction</td><td>Tingkat kepuasan karyawan terhadap pekerjaan mereka.</td></tr>
        <tr><td>Performance Rating</td><td>Penilaian kinerja karyawan oleh perusahaan.</td></tr>
        <tr><td>Overtime</td><td>Status apakah karyawan bekerja lembur (<code>Yes</code>/<code>No</code>).</td></tr>
        <tr><td>Number of Promotions</td><td>Jumlah promosi yang telah diterima oleh karyawan.</td></tr>
        <tr><td>Distance from Home</td><td>Jarak tempat tinggal karyawan dengan tempat kerja dalam mil.</td></tr>
        <tr><td>Education Level</td><td>Tingkat pendidikan terakhir karyawan.</td></tr>
        <tr><td>Marital Status</td><td>Status pernikahan karyawan: <code>Married</code>, <code>Single</code>, <code>Divorced</code>.</td></tr>
        <tr><td>Number of Dependents</td><td>Jumlah tanggungan yang dimiliki karyawan.</td></tr>
        <tr><td>Job Level</td><td>Tingkatan jabatan karyawan di perusahaan: <code>Mid</code>, <code>Entry</code>, <code>Senior</code>.</td></tr>
        <tr><td>Company Size</td><td>Ukuran perusahaan: <code>Large</code>, <code>Medium</code>, <code>Small</code>.</td></tr>
        <tr><td>Remote Work</td><td>Apakah karyawan bekerja dari jarak jauh: <code>Yes</code>/<code>No</code>.</td></tr>
        <tr><td>Leadership Opportunities</td><td>Kesempatan untuk memimpin dalam perusahaan: <code>Yes</code>/<code>No</code>.</td></tr>
        <tr><td>Innovation Opportunities</td><td>Kesempatan untuk berinovasi dalam perusahaan: <code>Yes</code>/<code>No</code>.</td></tr>
        <tr><td>Company Reputation</td><td>Reputasi perusahaan di mata karyawan: <code>Excellent</code>, <code>Good</code>, <code>Fair</code>, <code>Poor</code>.</td></tr>
        <tr><td>Employee Recognition</td><td>Penghargaan yang diberikan kepada karyawan atas pencapaian mereka: <code>Very High</code>, <code>High</code>, <code>Medium</code>, <code>Low</code>.</td></tr>
        <tr><td>Attrition</td><td>Status apakah karyawan tetap bertahan atau mengundurkan diri (<code>Yes</code>/<code>No</code>).</td></tr>
        </tbody>
        </table>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model yang digunakan
    st.markdown("## Algoritma yang Digunakan dalam Pemodelan")
    st.markdown(''' 
                Dalam penelitian ini, algoritma **Gradient Boosting** digunakan karena performanya yang sangat baik dalam mengatasi permasalahan klasifikasi. Dengan nilai recall train sebesar 77%, model ini dapat memprediksi potensi resign karyawan dengan akurasi yang tinggi, membantu perusahaan untuk meningkatkan kebijakan retensi mereka.
                ''')
    
    st.markdown("---")


if __name__ == "__main__":
    home()
