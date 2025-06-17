import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr, kendalltau, spearmanr, pearsonr
from IPython.display import display
from sklearn.preprocessing import OrdinalEncoder # type: ignore
from scipy import stats
from sklearn.metrics import classification_report # type: ignore

# 1. Data Exploration
def data_explore(df):
    # Tampilkan info DataFrame
    print("=== Info DataFrame ===")
    df.info() 
    print()

    # Hitung duplicate dan total rows
    duplicates = df.duplicated().sum()
    total_rows = len(df)
    percentage_dup = duplicates / len(df) *100
    percentage_rows = 100

    # DataFrame jumlah missing per kolom
    missing = df.isnull().sum().reset_index()
    missing.columns = ['Kolom', 'Jumlah Missing Value']

    permiss_val = (missing['Jumlah Missing Value'] / total_rows) * 100
    permiss_val = permiss_val.reset_index(drop=True)
    missing['Persentase Missing Value'] = permiss_val
    missing['Jumlah Missing Value'] = missing.apply(
        lambda row: f"{row['Jumlah Missing Value']} ({row['Persentase Missing Value']:.2f}%)", axis=1)
    missing.drop(columns=['Persentase Missing Value'], inplace=True)

    # DataFrame jumlah nilai unik per kolom
    unique_counts = df.nunique().reset_index()
    unique_counts.columns = ['Kolom', 'Jumlah Nilai Unik']
    
    # DataFrame jumlah nilai unik per kolom
    listItemUnique = []
    for col in df.columns:
        listItemUnique.append([
            col, 
            df[col].unique().tolist()
        ])
    item_unik = pd.DataFrame(listItemUnique, columns=['Kolom', 'Item Unik'])

    # DataFrame untuk duplicate rows dan total rows
    dup = pd.DataFrame({
        'Kategori': ['Jumlah Duplicate Rows', 'Jumlah Total Baris'],
        'Jumlah': [duplicates, total_rows],
        'Persentase': [percentage_dup, percentage_rows]
    })

    # Gabungkan missing dan unique_counts berdasarkan kolom 'Kolom'
    summary = pd.merge(missing, unique_counts, on='Kolom')
    summary = pd.merge(summary, item_unik, on='Kolom')

    print("\n=== Missing & Unique Value  ===")
    display(summary)

    print("\n=== Duplicate Value & Total Rows ===")
    display(dup)

# 2. Statistika Deskriptif (Central Tendency)
def descriptive_statistics(df):
    """
    Menghitung statistik deskriptif seperti mean, median, standar deviasi, nilai maksimum, minimum, dan kuartil.
    Menambahkan informasi skewness dan kurtosis.
    """
    # Looping untuk setiap kolom numerik dalam dataframe
    for col in df.select_dtypes(exclude=object).columns:  # Hanya untuk kolom numerik
        print(f"\nStatistik Deskriptif untuk kolom ====> {col}")
        print(f"Rata-rata                   : {df[col].mean():,.2f}")
        print(f"Median                      : {df[col].median():,.2f}")
        print(f"Modus                       : {df[col].mode()[0]:,.2f}")
        print(f"Standar Deviasi             : {df[col].std():,.2f}")
        
        # Menghitung dan menampilkan range (max - min)
        range_value = df[col].max() - df[col].min()
        print(f"Range                       : {range_value:,.2f}")
        
        # Skewness dan Kurtosis
        print(f"Skewness                    : {df[col].skew():.2f}")
        print(f"Kurtosis                    : {df[col].kurt():.2f}")
        print(f"Nilai Terkecil (Min)        : {df[col].min():.2f}")
        print(f"Persebaran Quartil 1        : {df[col].quantile(0.25):.2f}")
        print(f"Persebaran Quartil 2        : {df[col].quantile(0.50):.2f}")
        print(f"Persebaran Quartil 3        : {df[col].quantile(0.75):.2f}")
        print(f"Nilai Terbesar (Max)        : {df[col].max():.2f}")


# 3. Plot Distribution
def plot_distribution(df):
    """
    Membuat visualisasi histogram dan boxplot untuk kolom yang ditentukan.
    """
    for col in df.select_dtypes(exclude=object).columns:  # Hanya untuk kolom numerik
        print(f"Visualisasi untuk kolom <==== {col} ====>")
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True, color='skyblue', bins=20)
        plt.title(f"Histogram of {col}")
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(f"Boxplot of {col}")
        
        plt.tight_layout()
        plt.show()

# 4. Check Outliers
def check_outlier(X_train_num, plot=True):
    """
    Menghitung batas bawah, batas atas, dan persentase outlier untuk fitur numerik.
    Juga menampilkan plot distribusi tiap fitur dengan batas outlier.

    Parameters:
    - X_train_num: DataFrame berisi fitur numerik dari data training.
    - plot: Boolean, jika True maka akan memunculkan plot distribusi setiap fitur.

    Returns:
    - DataFrame dengan kolom:
        'column', 'Skewness Value', 'Distribusi', 
        'lower_boundary', 'upper_boundary', 'percentage_total_outlier'
    """

    column = []
    skew_vals = []
    distribusi = []
    lower_bound = []
    upper_bound = []
    percent_total_outlier = []

    for i, col in enumerate(X_train_num.columns):
        skew_val = round(X_train_num[col].skew(), 1)
        distrib = 'normal' if -0.5 <= skew_val <= 0.5 else 'skewed'
        
        # Hitung batas bawah & atas berdasarkan distribusi
        if distrib == 'skewed':
            Q1 = X_train_num[col].quantile(0.25)
            Q3 = X_train_num[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
        else:
            mean = X_train_num[col].mean()
            std = X_train_num[col].std()
            lower = mean - 3 * std
            upper = mean + 3 * std

        # Hitung persentase outlier
        n_upper = (X_train_num[col] > upper).sum()
        n_lower = (X_train_num[col] < lower).sum()
        outlier_percent = ((n_upper + n_lower) / len(X_train_num)) * 100

        # Simpan semua nilai
        column.append(col)
        skew_vals.append(skew_val)
        distribusi.append(distrib)
        lower_bound.append(round(lower, 2))
        upper_bound.append(round(upper, 2))
        percent_total_outlier.append(outlier_percent)

        # Plot distribusi dan batas
        if plot:
            plt.figure(figsize=(8, 2))
            sns.boxplot(x=X_train_num[col], color='skyblue')
            plt.axvline(lower, color='green', linestyle='--', label='Lower Bound')
            plt.axvline(upper, color='red', linestyle='--', label='Upper Bound')
            plt.title(f'Boxplot Fitur: {col} (Skewness: {skew_val})')
            plt.xlabel(col)
            plt.legend()
            plt.tight_layout()
            plt.show()


    # Buat DataFrame hasil
    outliers = pd.DataFrame({
        'column': column,
        'Skewness Value': skew_vals,
        'Distribusi': distribusi,
        'lower_boundary': lower_bound,
        'upper_boundary': upper_bound,
        'percentage_total_outlier (%)': percent_total_outlier
    })

    return outliers

# 5. Correlation Analysis
def correlation_analysis(df, nilai_skew=0.5):
    """
    Menghitung dan memvisualisasikan korelasi antar fitur numerik.
    
    - Pearson: jika semua kolom numerik berskew rendah (< nilai_skew)
    - Spearman: jika ada kolom dengan skew tinggi
    - Point-Biserial: jika disediakan target_col biner
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset input
    target_col : str or None
        Nama kolom target biner (default None)
    nilai_skew : float
        Batas ambang skewness (default 0.5)
    alpha : float
        Level signifikansi untuk Point-Biserial correlation (default 0.05)
    """

    # Pilih kolom object
    df_obj = df.select_dtypes(include='object')    
    # Pilih kolom numerik saja
    df_num = df.select_dtypes(include='number')

    # Deteksi kolom normal dan skewed berdasarkan skewness
    normal_cols = []
    skewed_cols = []
    object_cols = []

    for col in df_num.columns:
        skew_val = df[col].skew()
        if abs(skew_val) < nilai_skew:
            normal_cols.append(col)
        else:
            skewed_cols.append(col)
    
    for col in df_obj.columns:
        object_cols.append(col)

    print(f"Normal Distribution Columns   : {normal_cols if normal_cols else '-- Tidak ada kolom normal --'}")
    print(f"Skewed Distribution Columns   : {skewed_cols if skewed_cols else '-- Tidak ada kolom skewed --'}")
    print(f"Object Columns                : {object_cols if object_cols else '-- Tidak ada kolom object --'}")
    print()

    # Tentukan metode korelasi utama
    if len(normal_cols) > 0:
        method = 'pearson'
        print(f"Using correlation method      : {method.upper()} ===> {normal_cols}")
        corr_matrix_pearson = df_num.corr(method=method)
        
        # Visualisasi korelasi antar fitur numerik
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix_pearson, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        # plt.xticks(rotation=45)
        plt.title(f"Correlation Matrix ({method.capitalize()})")
        plt.tight_layout()
        plt.show()
        pval_pearson = pd.DataFrame(np.ones_like(corr_matrix_pearson), columns=df_num.columns, index=df_num.columns)

        for i in normal_cols:
            for j in normal_cols:
                if i != j:
                    _, pval = pearsonr(df[i], df[j])
                    pval_pearson.loc[i, j] = pval
                else:
                    pval_pearson.loc[i, j] = 0.0  # diagonal

        # Menyertakan p-value pada matrix signifikansi
        signif_pearson = pval_pearson < 0.05
        print(f"\nSignificance Matrix (p < 0.05) - Pearson:")
        signif_pearson_with_pval = signif_pearson.astype(str) + ' (' + pval_pearson.round(4).astype(str) + ')'
        display(signif_pearson_with_pval)


    if len(skewed_cols) > 0:
        method = 'spearman'
        print(f"Using correlation method      : {method.upper()} ===> {skewed_cols}")
        corr_matrix_spearman = df_num.corr(method=method)

        # Visualisasi korelasi antar fitur numerik
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix_spearman, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        # plt.xticks(rotation=45)
        plt.title(f"Correlation Matrix ({method.capitalize()})")
        plt.tight_layout()
        plt.show()

        pval_spearman = pd.DataFrame(np.ones_like(corr_matrix_spearman), columns=df_num.columns, index=df_num.columns)

        for i in skewed_cols:
            for j in skewed_cols:
                if i != j:
                    _, pval = spearmanr(df[i], df[j])
                    pval_spearman.loc[i, j] = pval
                else:
                    pval_spearman.loc[i, j] = 0.0

        # Menyertakan p-value pada matrix signifikansi
        signif_spearman = pval_spearman < 0.05
        print(f"\nSignificance Matrix (p < 0.05) - Spearman:")
        signif_spearman_with_pval = signif_spearman.astype(str) + ' (' + pval_spearman.round(4).astype(str) + ')'
        display(signif_spearman_with_pval)

    if object_cols:   
        # Encoding
        encoder = OrdinalEncoder()
        df_obj_encoded = df_obj.copy()
        df_obj_encoded[:] = encoder.fit_transform(df_obj)

        method = 'kendall'
        print(f"Using correlation method      : {method.upper()}")
        # Hitung korelasi Kendall's tau manual karena pandas .corr() tidak mendukung kendall untuk DataFrame
        corr_matrix_kendalltau = df_obj_encoded.corr(method=method)
 
        # Visualisasi
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix_kendalltau.astype(float), annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        # plt.xticks(rotation=45)
        plt.title("Correlation Matrix (KENDALL) due to object columns")
        plt.tight_layout()
        plt.show()

        pval_kendall = pd.DataFrame(np.ones_like(corr_matrix_kendalltau), columns=df_obj.columns, index=df_obj.columns)

        for i in object_cols:
            for j in object_cols:
                if i != j:
                    _, pval = kendalltau(df_obj_encoded[i], df_obj_encoded[j])
                    pval_kendall.loc[i, j] = pval
                else:
                    pval_kendall.loc[i, j] = 0.0

        # Menyertakan p-value pada matrix signifikansi
        signif_kendall = pval_kendall < 0.05
        print(f"\nSignificance Matrix (p < 0.05) - Kendall:")
        signif_kendall_with_pval = signif_kendall.astype(str) + ' (' + pval_kendall.round(4).astype(str) + ')'
        display(signif_kendall_with_pval)

# 6. Point-Bisserial Correlation
def correlation_analysis_binary(df, target_col, alpha=0.05, h0=None, h1=None, show=True):
    """
    Fungsi ini melakukan analisis korelasi antara fitur numerik dan target biner dengan dua metode:
    1. Point-Biserial Correlation untuk data numerik dan target biner.
    2. Chi-Square Analysis untuk fitur kategorikal dan target biner.

    Argumen:
    - df: DataFrame yang berisi data yang ingin dianalisis
    - target_col: Kolom target (harus biner)
    - alpha: Tingkat signifikansi untuk pengujian hipotesis (default 0.05)
    - h0: Hipotesis Nol (H0), jika tidak diinput, akan menggunakan default
    - h1: Hipotesis Alternatif (H1), jika tidak diinput, akan menggunakan default
    - show_hypothesis_and_conclusion: Parameter untuk menampilkan atau menyembunyikan hipotesis dan kesimpulan (default True)
    """
    
    # Periksa apakah kolom target ada dalam DataFrame
    if target_col not in df.columns:
        print(f"Kolom target '{target_col}' tidak ditemukan.")
        return
    
    target = df[target_col]
    
    # Pastikan target adalah biner (0 dan 1)
    # if target.nunique() != 2:
    #     print(f"Kolom target '{target_col}' bukan biner. Target harus memiliki dua nilai unik.")
    #     return
    
    # Ubah kolom target menjadi tipe kategori
    df[target_col] = df[target_col].astype('category')
    
    print(f"\nAnalisis Korelasi terhadap target ===> '{target_col}'")
    
    # === 1. Analisis Point-Biserial ===
    df_num = df.select_dtypes(include='number')  # Memilih kolom numerik
    if df_num.empty:
        print("\nTidak ada kolom numerik untuk analisis Point-Biserial.")
    else:
        pb_results = []

        for col in df_num.columns:
            if col != target_col:
                series = df[col].dropna()  # Menghapus missing values
                aligned_target = target.loc[series.index]
                
                # Pastikan target adalah biner (0 dan 1)
                if aligned_target.nunique() == 2:
                    r_pb, p_val = pointbiserialr(series, aligned_target)
                    signif = "Signifikan" if p_val < alpha else "Tidak signifikan"

                    pb_results.append({
                        "Feature": col,
                        "r_pb": round(r_pb, 3),
                        "p_value": round(p_val, 10),
                        "Significance": signif
                    })
                else:
                    print(f"Kolom {col} tidak memenuhi kriteria target biner untuk Point-Biserial.")

        pb_df = pd.DataFrame(pb_results).sort_values(by='r_pb', ascending=False)

        # Tampilkan hasil Point-Biserial
        print("\n=== Hasil Point-Biserial Correlation ===")
        print(pb_df)

        # Menampilkan Hipotesis dan Kesimpulan jika diperlukan
        if show:
            if h0 is None or h1 is None:
                for col in df_num.columns:
                    if col != target_col:
                        # Jika hipotesis tidak diberikan, menggunakan default
                        print("\n=== Hipotesis ===")
                        print(f"H0: Tidak ada hubungan antara {target_col} dan {col}.")
                        print(f"H1: Ada hubungan antara {target_col} dan {col}.")
            else:
                print("\n=== Hipotesis yang Diberikan ===")
                print(f"H0: {h0}")
                print(f"H1: {h1}")

            # Menampilkan kesimpulan
            for index, row in pb_df.iterrows():
                for col in df_num.columns:
                    if col != target_col:
                        if row['p_value'] < alpha:
                            print(f"\nKesimpulan: Ada hubungan antara {target_col} dan {row[{col}]}")
                        else:
                            print(f"\nKesimpulan: Tidak ada hubungan antara {target_col} dan {row[{col}]}")

        # Heatmap untuk korelasi Point-Biserial
        heatmap_df = pb_df.set_index('Feature')[['r_pb']]
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap='coolwarm',  # colormap warna-warni seperti correlation matrix
            center=0,         # pusat warna di 0
            linewidths=0.5,
            fmt=".3f",        # format angka korelasi
            cbar_kws={'label': 'Point-Biserial Correlation (r_pb)'}
        )

        plt.title(f'Point-Biserial Correlation Matrix terhadap "{target_col}"', fontsize=12)
        plt.ylabel("Fitur")
        plt.xlabel("")
        plt.tight_layout()
        plt.show()

    # === 2. Analisis Chi-Square ===
    df_cat = df.select_dtypes(include='object')  # Memilih kolom kategorikal
    if df_cat.empty:
        print("\nTidak ada kolom kategorikal untuk analisis Chi-Square.")
    else:
        chi_results = []

        for col in df_cat.columns:
            # Ubah kolom menjadi tipe kategori jika belum
            df[col] = df[col].astype('category')
            
            contingency_table = pd.crosstab(df[col], target)
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                continue  # Skip kolom yang kontingensinya kurang dari 2x2
            chi2, p_val, dof, ex = chi2_contingency(contingency_table)
            signif = "Signifikan" if p_val < alpha else "Tidak signifikan"

            chi_results.append({
                "Feature": col,
                "Chi2": round(chi2, 3),
                "p_value": round(p_val, 10),
                "Significance": signif
            })

        chi_df = pd.DataFrame(chi_results).sort_values(by='Chi2', ascending=False)

        # Tampilkan hasil Chi-Square
        print("\n=== Hasil Chi-Square Analysis ===")
        print(chi_df)

        # Menampilkan Hipotesis dan Kesimpulan jika diperlukan
        if show:
            if h0 is None or h1 is None:
                for col in df_cat.columns:
                    if col != target_col:
                        # Jika hipotesis tidak diberikan, menggunakan default
                        print("\n=== Hipotesis ===")
                        print(f"H0: Tidak ada hubungan antara {target_col} dan {col}.")
                        print(f"H1: Ada hubungan antara {target_col} dan {col}.")
            else:
                print("\n=== Hipotesis yang Diberikan ===")
                print(f"H0: {h0}")
                print(f"H1: {h1}")

            # Menampilkan kesimpulan
            for index, row in chi_df.iterrows():
                for col in df_cat.columns:
                    if col != target_col:
                        if row['p_value'] < alpha:
                            print(f"\nKesimpulan: Ada hubungan antara {target_col} dan {row[{col}]}")
                        else:
                            print(f"\nKesimpulan: Tidak ada hubungan antara {target_col} dan {row[{col}]}")

        # Heatmap untuk statistik Chi-Square
        heatmap_df = chi_df.set_index('Feature')[['Chi2']]
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap='YlGnBu',
            fmt=".3f",
            linewidths=0.5,
            cbar_kws={'label': 'Chi-Square Statistic'}
        )
        plt.title(f'Chi-Square Analysis terhadap "{target_col}"', fontsize=12)
        plt.ylabel("Fitur")
        plt.xlabel("")
        plt.tight_layout()
        plt.show()

# 7. Cek persentase missing value pada fitur tertentu
def persentase_missing_value(df_train, df_test, fitur_list):
    # Hitung persentase null di train
    nilainulltrain = (df_train[fitur_list].isnull().sum() / len(df_train)) * 100
    nntr = nilainulltrain.reset_index().rename(columns={'index': 'kolom', 0: 'persentase null train'})
    
    # Hitung persentase null di test
    nilainulltest = (df_test[fitur_list].isnull().sum() / len(df_test)) * 100
    nnts = nilainulltest.reset_index().rename(columns={'index': 'kolom', 0: 'persentase null test'})
    
    # Gabung keduanya
    hasil = pd.merge(nntr, nnts, on='kolom')
    
    return hasil

# 8. Cek persentase dan value tiap kolom
def calculate_value_percentage(df, column, plot=None):
    # Memeriksa apakah kolom ada dalam DataFrame
    if column not in df.columns:
        raise ValueError(f"Kolom '{column}' tidak ditemukan dalam DataFrame.")
    
    # Menghitung jumlah nilai unik untuk kolom yang dipilih
    val_counts = df[column].value_counts()
    
    # Menghitung persentase untuk nilai unik
    per = (val_counts / len(df)) * 100
    
    # Menyimpan hasil perhitungan dalam list
    per_val_col = []
    for value, count in val_counts.items():
        per_val_col.append({
            'Nilai': value,
            'Jumlah': count,
            'Persentase (%)': per[value]
        })
    
    # Membuat DataFrame dari list hasil perhitungan
    pervalcolsum = pd.DataFrame(per_val_col)

    display(pervalcolsum)
    # Visualisasi bar chart jika diinginkan
    if plot:
        plt.figure(figsize=(10, 6))
        plt.bar(pervalcolsum['Nilai'].astype(str), pervalcolsum['Jumlah'], color='skyblue')
        plt.title(f'Distribusi Nilai untuk Kolom: {column}')
        plt.xlabel('Nilai')
        plt.ylabel('Jumlah')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    



# 9. Uji Hipotesis t-test (unknown sample)
def t_test_analysis_with_input(df, target_col, feature_col, alpha=0.05, h0=None, h1=None):
    """
    Fungsi ini melakukan analisis t-test untuk membandingkan rata-rata antara dua kelompok (biner) pada fitur numerik dan target biner,
    dengan inputan manual untuk hipotesis H0 dan H1.
    
    Argumen:
    - df: DataFrame yang berisi data yang ingin dianalisis
    - target_col: Kolom target (harus biner)
    - feature_col: Kolom fitur numerik yang akan diuji
    - alpha: Tingkat signifikansi untuk pengujian hipotesis (default 0.05)
    - h0: Hipotesis Nol (H0), jika tidak diinput, akan menggunakan default
    - h1: Hipotesis Alternatif (H1), jika tidak diinput, akan menggunakan default
    """
    
    # Periksa apakah kolom target dan fitur ada dalam DataFrame
    if target_col not in df.columns or feature_col not in df.columns:
        print(f"Kolom '{target_col}' atau '{feature_col}' tidak ditemukan.")
        return
    
    target = df[target_col]
    feature = df[feature_col]
    
    # Pastikan target adalah biner
    if target.nunique() != 2:
        print(f"Kolom target '{target_col}' bukan biner. Tidak bisa hitung t-test.")
        return

    print(f"\nAnalisis t-test untuk '{feature_col}' terhadap target '{target_col}'")

    # Memisahkan data berdasarkan nilai target (0 atau 1)
    group1 = feature[target == target.unique()[0]]
    group2 = feature[target == target.unique()[1]]

    # Uji t-test antara dua kelompok
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)  # Menggunakan asumsi varian yang tidak sama

    # Menentukan signifikansi
    signif = "Signifikan" if p_val < alpha else "Tidak signifikan"

    # Menampilkan hasil uji t-test
    print("\n=== Hasil t-test ===")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"p-value: {p_val:.10f}")
    print(f"Signifikansi: {signif}")

    # Hipotesis: Gunakan input manual jika ada, jika tidak akan menggunakan default
    if h0 is None or h1 is None:
        print("\n=== Hipotesis Default ===")
        print(f"H0: Tidak ada perbedaan rata-rata antara kedua kelompok pada fitur '{feature_col}'")
        print(f"H1: Ada perbedaan rata-rata antara kedua kelompok pada fitur '{feature_col}'")
    else:
        print("\n=== Hipotesis yang Diberikan ===")
        print(f"H0: {h0}")
        print(f"H1: {h1}")

    if p_val < alpha:
        print("\nKesimpulan: Ada hubungan antara", target_col, "dan", feature_col)
    else:
        print("\nKesimpulan: Tidak ada hubungan antara", target_col, "dan", feature_col)
    

# 10. plot_line_relationship
def plot_relationship(dataset, x_col, target_cols, kind='line', figsize=(10, 7), custom_colors=None):
    """
    Fungsi fleksibel untuk memvisualisasi hubungan antara satu kolom X dengan satu atau lebih kolom target Y,
    dalam berbagai jenis plot seaborn: 'line', 'scatter', 'bar', 'hist', 'box', 'violin', 'kde'.
    
    Argumen:
    - dataset: pandas DataFrame
    - x_col: kolom yang akan jadi sumbu X (bisa None untuk plot seperti hist/kde)
    - target_cols: list kolom target Y
    - kind: jenis plot: 'line', 'scatter', 'bar', 'hist', 'box', 'violin', 'kde'
    - figsize: ukuran grafik (default (17, 15))
    - custom_colors: dictionary untuk mengubah warna manual, format {nilai_target: warna}
    """
    # Jika custom_colors diberikan, gunakan warna tersebut, jika tidak, gunakan Set1
    fig, axs = plt.subplots(len(target_cols), 1, figsize=figsize)

    if len(target_cols) == 1:
        axs = [axs]

    for i, target_col in enumerate(target_cols):
        ax = axs[i]
        
        # Jika custom_colors diberikan, gunakan itu, jika tidak, tentukan warna berdasarkan nilai unik pada kolom target
        if custom_colors:
            value_to_color = custom_colors
        else:
            unique_values = dataset[target_col].unique()
            colors = sns.color_palette("Set1", len(unique_values))  # Warna untuk setiap kategori unik
            value_to_color = {value: colors[j] for j, value in enumerate(unique_values)}
        
        if kind == 'line':
            for value in dataset[target_col].unique():
                subset = dataset[dataset[target_col] == value]
                sns.lineplot(x=subset[x_col], y=subset[target_col], ax=ax, color=value_to_color.get(value, 'gray'), label=value)
        elif kind == 'scatter':
            for value in dataset[target_col].unique():
                subset = dataset[dataset[target_col] == value]
                sns.scatterplot(x=subset[x_col], y=subset[target_col], ax=ax, color=value_to_color.get(value, 'gray'), label=value)
        elif kind == 'bar':
            for value in dataset[target_col].unique():
                subset = dataset[dataset[target_col] == value]
                sns.barplot(x=subset[x_col], y=subset[target_col], ax=ax, color=value_to_color.get(value, 'gray'), label=value)
        elif kind == 'hist':
            for value in dataset[target_col].unique():
                subset = dataset[dataset[target_col] == value]
                sns.histplot(subset[target_col], ax=ax, color=value_to_color.get(value, 'gray'), kde=False, label=value)
        elif kind == 'box':
            sns.boxplot(x=dataset[x_col] if x_col else None, y=dataset[target_col], ax=ax, palette=value_to_color)
        elif kind == 'violin':
            sns.violinplot(x=dataset[x_col] if x_col else None, y=dataset[target_col], ax=ax, palette=value_to_color)
        elif kind == 'kde': 
            for value in dataset[target_col].unique():
                subset = dataset[dataset[target_col] == value]
                sns.kdeplot(data=subset, x=target_col, ax=ax, fill=True, color=value_to_color.get(value, 'gray'), label=value)
        elif kind == 'count':
            sns.countplot(data=dataset, x=target_col, ax=ax, palette=value_to_color)
        else:
            raise ValueError("Jenis plot tidak dikenali. Gunakan 'line', 'scatter', 'bar', 'hist', 'box', 'violin', atau 'kde'.")

        title = f'{target_col}' if kind in ['hist', 'kde'] else f'{x_col} vs {target_col}'
        ax.set_title(f'{kind.title()} Plot: {title}')
        ax.set_xlabel(x_col if x_col else '')
        ax.set_ylabel(target_col)

        # Menambahkan legend
        ax.legend()

    plt.tight_layout()
    plt.show()


# 11. Annova
def anova_analysis_with_input(df, target_col, feature_col, alpha=0.05, h0=None, h1=None):
    """
    Fungsi ini melakukan analisis ANOVA untuk membandingkan rata-rata antara lebih dari dua kelompok pada fitur numerik dan target kategorikal,
    dengan inputan manual untuk hipotesis H0 dan H1.
    
    Argumen:
    - df: DataFrame yang berisi data yang ingin dianalisis
    - target_col: Kolom target (harus kategorikal dengan lebih dari dua kategori)
    - feature_col: Kolom fitur numerik yang akan diuji
    - alpha: Tingkat signifikansi untuk pengujian hipotesis (default 0.05)
    - h0: Hipotesis Nol (H0), jika tidak diinput, akan menggunakan default
    - h1: Hipotesis Alternatif (H1), jika tidak diinput, akan menggunakan default
    """
    
    # Periksa apakah kolom target dan fitur ada dalam DataFrame
    if target_col not in df.columns or feature_col not in df.columns:
        print(f"Kolom '{target_col}' atau '{feature_col}' tidak ditemukan.")
        return
    
    target = df[target_col]
    feature = df[feature_col]
    
    # Pastikan target memiliki lebih dari dua kategori
    if target.nunique() <= 2:
        print(f"Kolom target '{target_col}' harus memiliki lebih dari dua kategori. Tidak bisa hitung ANOVA.")
        return

    print(f"\nAnalisis ANOVA untuk '{feature_col}' terhadap target '{target_col}'")

    # Memisahkan data berdasarkan kategori pada kolom target
    groups = [feature[target == category] for category in target.unique()]

    # Uji ANOVA antara kelompok-kelompok berdasarkan target
    f_stat, p_val = stats.f_oneway(*groups)

    # Menentukan signifikansi
    signif = "Signifikan" if p_val < alpha else "Tidak signifikan"

    # Menampilkan hasil uji ANOVA
    print("\n=== Hasil ANOVA ===")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"p-value: {p_val:.10f}")
    print(f"Signifikansi: {signif}")

    # Hipotesis: Gunakan input manual jika ada, jika tidak akan menggunakan default
    if h0 is None or h1 is None:
        print("\n=== Hipotesis Default ===")
        print(f"H0: Tidak ada perbedaan rata-rata antara kelompok-kelompok pada fitur '{feature_col}'")
        print(f"H1: Ada perbedaan rata-rata antara kelompok-kelompok pada fitur '{feature_col}'")
    else:
        print("\n=== Hipotesis yang Diberikan ===")
        print(f"H0: {h0}")
        print(f"H1: {h1}")

    if p_val < alpha:
        print("\nKesimpulan: Ada hubungan antara", target_col, "dan", feature_col)
    else:
        print("\nKesimpulan: Tidak ada hubungan antara", target_col, "dan", feature_col)

# 12. Chi-Square Test
def chi_square_analysis(df, target_col, feature_col, alpha=0.05, h0=None, h1=None):
    """
    Fungsi ini melakukan uji Chi-Square untuk menguji apakah ada hubungan antara dua variabel kategorikal
    (misalnya, 'Attrition' dan 'Job Satisfaction').
    
    Argumen:
    - df: DataFrame yang berisi data yang ingin dianalisis
    - target_col: Kolom target (harus kategorikal)
    - feature_col: Kolom fitur kategorikal
    - alpha: Tingkat signifikansi untuk pengujian hipotesis (default 0.05)
    - h0: Hipotesis Nol (H0), jika tidak diinput, akan menggunakan default
    - h1: Hipotesis Alternatif (H1), jika tidak diinput, akan menggunakan default
    """
    
    # Periksa apakah kolom target dan fitur ada dalam DataFrame
    if target_col not in df.columns or feature_col not in df.columns:
        print(f"Kolom '{target_col}' atau '{feature_col}' tidak ditemukan.")
        return
    
    target = df[target_col]
    feature = df[feature_col]
    
    # Membuat tabel kontingensi
    contingency_table = pd.crosstab(target, feature)

    # Uji Chi-Square
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

    # Menentukan signifikansi
    signif = "Signifikan" if p_val < alpha else "Tidak signifikan"

    # Menampilkan hasil uji Chi-Square
    print("\n=== Hasil Uji Chi-Square ===")
    print(f"Chi-Square Test Statistic: {chi2_stat:.3f}")
    print(f"p-value: {p_val:.10f}")
    print(f"Signifikansi: {signif}")

    # Menampilkan hipotesis
    if h0 is None or h1 is None:
        # Jika hipotesis tidak diberikan, menggunakan default
        print("\n=== Hipotesis ===")
        print(f"H0: Tidak ada hubungan antara {target_col} dan {feature_col}.")
        print(f"H1: Ada hubungan antara {target_col} dan {feature_col}.")
    else:
        print("\n=== Hipotesis yang Diberikan ===")
        print(f"H0: {h0}")
        print(f"H1: {h1}")

    # Menampilkan kesimpulan
    if p_val < alpha:
        print("\nKesimpulan: Ada hubungan antara", target_col, "dan", feature_col)
    else:
        print("\nKesimpulan: Tidak ada hubungan antara", target_col, "dan", feature_col)
    
    # Visualisasi heatmap Chi-Square (perbandingan observed dan expected)
    plt.figure(figsize=(8, 10))

    # Heatmap Observed Frequencies (Atas)
    plt.subplot(2, 1, 1)
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1, linecolor='black')
    plt.title("Observed Frequencies (Tabel Kontingensi)", fontsize=14)
    plt.xlabel(feature_col, fontsize=12)
    plt.ylabel(target_col, fontsize=12)

    # Heatmap Expected Frequencies (Bawah)
    plt.subplot(2, 1, 2)
    sns.heatmap(expected, annot=True, fmt=".2f", cmap="Oranges", cbar=False, linewidths=1, linecolor='black')
    plt.title("Expected Frequencies (Dihitung dari Chi-Square)", fontsize=14)
    plt.xlabel(feature_col, fontsize=12)
    plt.ylabel(target_col, fontsize=12)

    plt.tight_layout()
    plt.show()

def evaluate_model_class_report(model, X_train, y_train, X_test, y_test):
    """
    Parameters:
    - model: model yang sudah dilatih (KNN, SVC, Decision Tree, Random Forest, Gradient Boost)
    - X_train: Data fitur untuk training
    - y_train: Data label untuk training
    - X_test: Data fitur untuk testing
    - y_test: Data label untuk testing
    
    Returns:
    - None: Mencetak hasil evaluasi
    """
    # Prediksi hasil model pada data uji dan data latih
    y_pred_tuning_train = model.predict(X_train)
    y_pred_tuning_test = model.predict(X_test)

    # 3. Classification report
    print("=============== Classification Report ===============\n")
    print("Train Data:")
    print(classification_report(y_train, y_pred_tuning_train))
    print('------------------------------------------------------')
    print("Test Data:")
    print(classification_report(y_test, y_pred_tuning_test))