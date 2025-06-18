# Deteksi Potensi Resign Karyawan Menggunakan Machine Learning

## Repository Outline

1. P1M2_rafi_siregar.ipynb -- Notebook analisis dan pengembangan model.
2. P1M2_rafi_siregar_inf.ipynb -- Notebook untuk melakukan model inference.
3. P1M2_rafi_siregar.csv -- Dataset attrition yang berasal dari kaggle.
4. datainference.csv -- Dataset hasil model inference untuk melakukan analisa terhadap model prediksi.
5. P1M2_rafi_siregar_conceptual.txt -- Conceptual Problem.
6. eda_package.py -- Custom package untuk melakukan EDA.
7. README.md -- Instruksi Milestone 2
8. description.md -- Penjelasan gambaran umum project.
9. AttritionBoosting.pkl -- Model machine learning yang digunakan untuk memprediksi attrition karyawan.
10. Folder deployment -- Berisikan data-data model deployment proyek ini.

## Problem Background

Saya merupakan seorang data scientist yang ditugaskan untuk membuat model machine learning guna memprediksi potensi resign calon karyawan. Proyek ini di latar belakangi oleh tingginya laporan attrition yang diterima perusahaan selama satu tahun terakhir. Hal ini dapat berdampak pada stabilitas organisasi, seperti meningkatnya biaya rekrutmen, hilangnya pengetahuan dan pengalaman karyawan, serta penurunan moral karyawan yang bertahan. Dengan memprediksi potensi resign, perusahaan dapat mengambil langkah-langkah pencegahan, seperti meningkatkan kebijakan retensi dan kepuasan kerja, untuk menjaga kelangsungan dan kestabilan organisasi.

## Project Output

Membuat model machine learning yang dapat memprediksi potensi resign karyawan berdasarkan data historis, dengan evaluasi menggunakan metrik seperti akurasi, precision, recall, dan F1-score. Model ini akan memberikan wawasan mengenai faktor-faktor yang mempengaruhi keputusan attrition, seperti usia, kepuasan kerja, gaji, dan jarak rumah ke tempat kerja, serta menghasilkan visualisasi untuk menggambarkan distribusi dan hubungan antar variabel. Berdasarkan analisis ini, akan disarankan langkah-langkah untuk mengurangi tingkat attrition dan meningkatkan retensi karyawan, yang bisa diterapkan dalam kebijakan perusahaan.

## Method

Metode yang digunakan dalam membangun model ini adalah **machine learning supervised learning** dengan algoritma seperti  **KNN**,  **Boosting** atau  **random forest** . Data historis karyawan digunakan untuk melatih model, dengan fitur-fitur seperti usia, kepuasan kerja, gaji, dan lainnya sebagai input. Model kemudian dievaluasi menggunakan metrik klasifikasi, terutama pada **recall** untuk mengukur kemampuan model dalam memprediksi karyawan yang berisiko resign.

## Stacks

Menggunakan Python untuk mengelola data dan melakukan perhitungan statistika, serta **Streamlit** dan **HuggingFace** untuk menguji coba model secara realtime (online). Pada Python, digunakan library **pandas** untuk mengelola dataframe, **numpy** untuk operasi numerik, **scipy** untuk analisis statistika, serta **seaborn** dan **matplotlib** untuk visualisasi data. **sklearn** digunakan untuk preprocessing data, model algoritma, dan evaluasi metrik seperti  **recall** ,  **precision** , dan  **f1-score** , serta untuk tuning hyperparameter dengan  **RandomizedSearchCV** . **KNNImputer** digunakan untuk imputasi data yang hilang, dan berbagai teknik **scaling** diterapkan dengan **RobustScaler** untuk menormalkan fitur agar model lebih efektif. Untuk **encoding** , digunakan **OneHotEncoder** dan **OrdinalEncoder** untuk mengonversi fitur kategorikal menjadi format yang bisa diproses model. Model dapat disimpan dan dimuat menggunakan  **pickle** , sementara **eda_package** adalah paket kustom untuk eksplorasi data. **Streamlit** akan digunakan untuk membuat aplikasi web interaktif yang memungkinkan pengguna untuk melihat visualisasi dan hasil analisis secara langsung.

## Reference

1. [Employee Attrition Classification Dataset](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset): Dataset ini mencakup berbagai fitur seperti usia, kepuasan kerja, gaji, lama bekerja, dan lainnya, yang dapat digunakan untuk membangun model machine learning guna memprediksi potensi resign karyawan.
2. [A Comprehensive Guide to Data Imputation Techniques, Strategies, and Best Practices](https://medium.com/@tarangds/a-comprehensive-guide-to-data-imputation-techniques-strategies-and-best-practices-152a10fee543): Artikel ini memberikan panduan menyeluruh tentang teknik-teknik imputasi data, strategi yang tepat, serta praktik terbaik yang dapat diterapkan dalam menangani nilai yang hilang dalam dataset.
3. **[ANALISIS FAKTOR-FAKTOR YANG MEMPENGARUHI KARYAWAN RESIGN PADA PT. SUMBER ALFARIA TRIJAYA TBK KOTA BANJARMASIN](https://eprints.uniska-bjm.ac.id/8005/1/ARTIKEL%20RATNA%20SARI%20FIX.pdf)**: Artikel ini membahas topik terkait analisis dan prediksi dalam konteks data karyawan dan attrition.
4. [Streamlit - Aplikasi Web Interaktif](https://huggingface.co/spaces/egar1444/EmployeeAttritionPrediction): Streamlit digunakan untuk membuat aplikasi web interaktif yang memungkinkan pengguna untuk melihat visualisasi dan hasil analisis secara langsung dari model machine learning yang telah saya kembangkan untuk memprediksi attrition.

**Referensi tambahan:**

- [Basic Writing and Syntax on Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
- [Contoh readme](https://github.com/fahmimnalfrzki/Swift-XRT-Automation)
- [Another example](https://github.com/sanggusti/final_bangkit) (**Must read**)
- [Additional reference](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)
