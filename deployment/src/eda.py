import streamlit as st  # type: ignore
import plot as ep  # type: ignore
import pandas as pd

def eda():
    # Title of the page
    st.title("Exploratory Data Analysis")

    # Selectbox to choose EDA type
    eda = st.selectbox('Pilih EDA', [
        'Bagaimana distribusi usia karyawan yang mengalami attrition dibandingkan dengan yang tidak?',
        'Apakah terdapat signifikansi tingkat kepuasan kerja antara karyawan yang bertahan dan yang keluar?',
        'Bagaimana persebaran lama bekerja karyawan dengan peluang memimpin divisi?',
        'Apakah ada perbedaan dalam tingkat attrition berdasarkan departemen atau peran pekerjaan?',
        'Sejauh mana faktor-faktor seperti jarak dari rumah ke tempat kerja mempengaruhi keputusan karyawan untuk bertahan?',
        'Bagaimana hubungan antara gaji bulanan dan tingkat attrition?',
        'Apakah terdapat hubungan antara pendapatan karyawan pada level pendidikan tertentu terhadap attrition?'
    ])

    st.markdown("---")

    # EDA based on the selected option
    if eda == 'Bagaimana distribusi usia karyawan yang mengalami attrition dibandingkan dengan yang tidak?':
        st.markdown("## Bagaimana distribusi usia karyawan yang mengalami attrition dibandingkan dengan yang tidak?")
        st.pyplot(ep.eda1())  # Visualisasi distribusi usia karyawan berdasarkan attrition
        st.write("Berdasarkan histogram distribusi usia karyawan yang mengalami attrition dan yang tidak, terlihat bahwa usia karyawan yang keluar cenderung lebih muda dibandingkan dengan yang bertahan. Ini menunjukkan bahwa perusahaan mungkin perlu lebih fokus pada karyawan muda untuk mencegah attrition.")
        st.markdown("---")
    
    elif eda == 'Apakah terdapat signifikansi tingkat kepuasan kerja antara karyawan yang bertahan dan yang keluar?':
        st.markdown("## Apakah terdapat signifikansi tingkat kepuasan kerja antara karyawan yang bertahan dan yang keluar?")
        st.pyplot(ep.eda2())  # Visualisasi perbedaan kepuasan kerja antara karyawan yang keluar dan bertahan
        st.markdown("### Hasil Chi-Square Analysis")
        st.write("""
			| Feature            | Chi2     | p_value | Significance |
			|--------------------|----------|---------|--------------|
			| Job_Satisfaction   | 388.237  | 0.0     | Signifikan   |
		""")
        st.markdown("### Hipotesis yang Diberikan")
        st.write("""
			- **H0**: Tidak terdapat signifikansi tingkat kepuasan kerja antara karyawan yang bertahan dan yang keluar
			- **H1**: Terdapat signifikansi tingkat kepuasan kerja antara karyawan yang bertahan dan yang keluar
		""")
        st.write("### Kesimpulan")
        st.write("""
			- **Karyawan yang Bertahan**: Mayoritas karyawan yang tetap bertahan di perusahaan memiliki tingkat kepuasan kerja yang **tinggi (High)**. Hal ini terlihat pada kolom "High" yang lebih tinggi pada karyawan yang bertahan dibandingkan dengan yang resign.
			- **Karyawan yang Resign**: Meskipun terdapat beberapa karyawan dengan kepuasan kerja tinggi yang resign, jumlahnya lebih rendah dibandingkan dengan yang bertahan. 
			- **Signifikansi**: Berdasarkan uji Chi-Square dengan **nilai Chi-Square 338** dan **p-value 0.0**, hubungan antara Job Satisfaction dan Attrition sangat signifikan. Hal ini menunjukkan bahwa tingkat kepuasan kerja berperan besar dalam keputusan karyawan untuk bertahan atau resign di perusahaan.
			- **Korelasi**: Meskipun kepuasan kerja tinggi berhubungan dengan keputusan bertahan, ada juga faktor lain yang perlu dipertimbangkan dalam keputusan karyawan untuk resign.
		""")
        st.markdown("---")
    
    elif eda == 'Bagaimana persebaran lama bekerja karyawan dengan peluang memimpin divisi?':
        st.markdown("## Bagaimana persebaran lama bekerja karyawan dengan peluang memimpin divisi?")
        st.pyplot(ep.eda3())  # Visualisasi hubungan lama bekerja dan peluang memimpin divisi
        st.write("Karyawan dengan lama bekerja lebih lama memiliki peluang lebih besar untuk memimpin divisi. Hal ini menunjukkan adanya hubungan positif antara pengalaman dan kesempatan untuk memimpin dalam perusahaan.")
        st.markdown("---")
    
    elif eda == 'Apakah ada perbedaan dalam tingkat attrition berdasarkan departemen atau peran pekerjaan?':
        st.markdown("## Apakah ada perbedaan dalam tingkat attrition berdasarkan departemen atau peran pekerjaan?")
        st.pyplot(ep.eda4())  # Visualisasi perbedaan attrition berdasarkan departemen atau peran pekerjaan
        st.write("Tingkat attrition pada departemen tertentu cenderung lebih tinggi, yang menunjukkan adanya masalah di dalam lingkungan kerja atau kepuasan kerja di departemen tersebut. Karyawan dengan peran pekerjaan tertentu lebih rentan untuk keluar.")
        st.markdown("---")
    
    elif eda == 'Sejauh mana faktor-faktor seperti jarak dari rumah ke tempat kerja mempengaruhi keputusan karyawan untuk bertahan?':
        st.markdown("## Sejauh mana faktor-faktor seperti jarak dari rumah ke tempat kerja mempengaruhi keputusan karyawan untuk bertahan?")
        st.pyplot(ep.eda5())  # Visualisasi hubungan jarak rumah dan keputusan bertahan atau keluar
        st.write("""
			### Hasil Point-Biserial Correlation
			| Feature             | r_pb   | p_value | Significance |
			|---------------------|--------|---------|--------------|
			| Distance_from_Home  | 0.094  | 0.0     | Signifikan   |
			
			### Hipotesis yang Diberikan
			- **H0**: Tidak ada hubungan antara Attrition dan Distance_from_Home.
			- **H1**: Ada hubungan antara Attrition dan Distance_from_Home.
		""")
        st.write("### Kesimpulan")
        st.write("""
			- Terdapat hubungan antara Attrition dan Distance_from_Home, meskipun korelasinya sangat kecil.
			- Grafik boxplot menunjukkan bahwa jarak dari rumah ke tempat kerja tidak membedakan signifikan antara karyawan yang resign dan yang bertahan.
			- Kelompok karyawan yang resign memiliki jarak tempuh yang lebih beragam, sementara kelompok yang bertahan memiliki jarak yang lebih seragam.
			- Hasil uji Point-Bisserial menunjukkan korelasi positif kecil sebesar **0.09**, yang mengindikasikan hubungan yang lemah antara jarak dan keputusan bertahan atau resign.
			- P-value yang sangat kecil menunjukkan hasil signifikansi yang kuat, tetapi hubungan yang ditemukan tetap lemah.
			- Jarak rumah ke tempat kerja bukanlah faktor utama dalam keputusan karyawan untuk resign atau bertahan di perusahaan.
		""")
        st.markdown("---")
    
    elif eda == 'Bagaimana hubungan antara gaji bulanan dan tingkat attrition?':
        st.markdown("## Bagaimana hubungan antara gaji bulanan dan tingkat attrition?")
        st.pyplot(ep.eda6())  # Visualisasi hubungan antara gaji bulanan dan attrition
        st.write("""
			### Hasil Point-Biserial Correlation
			| Feature          | r_pb   | p_value   | Significance |
			|------------------|--------|-----------|--------------|
			| Monthly_Income   | -0.011 | 0.003533  | Signifikan   |
			
			### Hipotesis yang Diberikan
			- **H0**: Tidak ada hubungan antara Attrition dan Monthly_Income.
			- **H1**: Ada hubungan antara Attrition dan Monthly_Income.
		""")
        st.write("""
			### Kesimpulan
			- **Hubungan antara Attrition dan Monthly Income**: Ada hubungan antara Attrition dan Monthly_Income, dengan korelasi yang sangat kecil **-0.011**, menunjukkan korelasi negatif.
			- **Visualisasi Boxplot**: Karyawan yang resign memiliki rentang gaji yang lebih besar, sementara karyawan yang bertahan memiliki rentang gaji yang lebih stabil dan lebih rendah.
			- **Outlier**: Persebaran karyawan dengan gaji sangat tinggi (outlier) terlihat pada kelompok resign.
			- **Signifikansi**: Hasil uji statistik menunjukkan signifikansi yang kuat, namun korelasi yang sangat kecil menunjukkan bahwa gaji bukanlah faktor utama dalam menentukan keputusan karyawan untuk resign atau bertahan.
			- **Asumsi Konteks Dunia Nyata**: Gaji seorang karyawan biasanya menjadi faktor penting dalam keputusan bertahan atau resign, namun berdasarkan hasil statistik ini, faktor lain mungkin lebih berpengaruh pada keputusan tersebut.
		""")
        st.markdown("---")
    
    elif eda == 'Apakah terdapat hubungan antara pendapatan karyawan pada level pendidikan tertentu terhadap attrition?':
        st.markdown("## Apakah terdapat hubungan antara pendapatan karyawan pada level pendidikan tertentu terhadap attrition?")
        st.pyplot(ep.eda7())  # Visualisasi hubungan antara pendidikan, pendapatan, dan attrition
        st.write("""
			### Hasil Chi-Square Analysis
			| Feature            | r_pb   | p_value | Significance |
			|--------------------|--------|---------|--------------|
			| Education Level    | 0.65   | 0.0     | Signifikan   |
			| Monthly Income     | 0.42   | 0.0     | Signifikan   |
		""")
        st.markdown("### Hipotesis yang Diberikan")
        st.write("""
			- **H0**: Tidak ada hubungan antara pendapatan dan tingkat pendidikan terhadap attrition.
			- **H1**: Ada hubungan antara pendapatan dan tingkat pendidikan terhadap attrition.
		""")
        st.write("### Kesimpulan")
        st.write("""
			Berdasarkan hasil analisis, terdapat hubungan signifikan antara level pendidikan dan pendapatan karyawan terhadap attrition. 
			Karyawan dengan pendidikan lebih tinggi cenderung memiliki pendapatan yang lebih tinggi dan lebih kecil kemungkinannya untuk keluar dari perusahaan. Hasil korelasi menunjukkan hubungan yang positif antara pendidikan dan pendapatan dengan tingkat attrition yang lebih rendah.
			
			Terdapat beberapa insight yang didapat:
			- Karyawan dengan tingkat pendidikan lebih tinggi cenderung memiliki pendapatan yang lebih besar.
			- Karyawan dengan pendidikan yang lebih tinggi juga menunjukkan kemungkinan yang lebih kecil untuk mengalami attrition.
			- Hal ini menunjukkan bahwa ada hubungan antara pendidikan dan pendapatan yang mempengaruhi keputusan karyawan untuk bertahan atau keluar dari perusahaan.
		""")
        st.markdown("---")

    
if __name__ == "__main__":
    eda()
