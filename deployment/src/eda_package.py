import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr, kendalltau, spearmanr, pearsonr
from IPython.display import display
from sklearn.preprocessing import OrdinalEncoder
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report

# 1. Data Exploration
def data_explore(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    # Tampilkan info DataFrame
    print("=== Info DataFrame ===")
    df.info() 
    print()

    duplicates = df.duplicated().sum()
    total_rows = len(df)
    percentage_dup = duplicates / len(df) * 100
    percentage_rows = 100

    missing = df.isnull().sum().reset_index()
    missing.columns = ['Kolom', 'Jumlah Missing Value']

    permiss_val = (missing['Jumlah Missing Value'] / total_rows) * 100
    permiss_val = permiss_val.reset_index(drop=True)
    missing['Persentase Missing Value'] = permiss_val
    missing['Jumlah Missing Value'] = missing.apply(
        lambda row: f"{row['Jumlah Missing Value']} ({row['Persentase Missing Value']:.2f}%)", axis=1)
    missing.drop(columns=['Persentase Missing Value'], inplace=True)

    unique_counts = df.nunique().reset_index()
    unique_counts.columns = ['Kolom', 'Jumlah Nilai Unik']
    
    listItemUnique = []
    for col in df.columns:
        listItemUnique.append([col, df[col].unique().tolist()])
    item_unik = pd.DataFrame(listItemUnique, columns=['Kolom', 'Item Unik'])

    dup = pd.DataFrame({
        'Kategori': ['Jumlah Duplicate Rows', 'Jumlah Total Baris'],
        'Jumlah': [duplicates, total_rows],
        'Persentase': [percentage_dup, percentage_rows]
    })

    summary = pd.merge(missing, unique_counts, on='Kolom')
    summary = pd.merge(summary, item_unik, on='Kolom')

    print("\n=== Missing & Unique Value  ===")
    display(summary)

    print("\n=== Duplicate Value & Total Rows ===")
    display(dup)

    # DataFrame summary of missing and unique values is shown, but we also visualize one plot here
    df_missing = missing.set_index('Kolom')
    df_missing['Jumlah Missing Value'] = df_missing['Jumlah Missing Value'].apply(lambda x: int(x.split()[0]))  # Convert to numeric for plotting
    df_missing['Jumlah Missing Value'].plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title('Missing Values per Column')
    ax.set_xlabel('Missing Count')
    ax.set_ylabel('Column')

    return fig


# 2. Statistika Deskriptif (Central Tendency)
def descriptive_statistics(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    summary_stats = df.describe().transpose()
    summary_stats['range'] = summary_stats['max'] - summary_stats['min']
    summary_stats['skew'] = df.skew()
    summary_stats['kurtosis'] = df.kurt()

    # Plot the summary statistics in a bar plot
    summary_stats[['mean', 'std', 'range', 'skew', 'kurtosis']].plot(kind='bar', ax=ax)
    ax.set_title('Descriptive Statistics (Mean, Std, Range, Skew, Kurtosis)')
    ax.set_xlabel('Features')
    ax.set_ylabel('Value')

    return fig


# 3. Plot Distribution
def plot_distribution(df):
    fig, ax = plt.subplots(len(df.select_dtypes(exclude=object).columns), 2, figsize=(12, 8))
    
    for i, col in enumerate(df.select_dtypes(exclude=object).columns): 
        ax[i, 0].clear()
        ax[i, 1].clear()

        sns.histplot(df[col], kde=True, color='skyblue', bins=20, ax=ax[i, 0])
        ax[i, 0].set_title(f"Histogram of {col}")

        sns.boxplot(x=df[col], color='skyblue', ax=ax[i, 1])
        ax[i, 1].set_title(f"Boxplot of {col}")
    
    plt.tight_layout()
    return fig


# 4. Check Outliers
def check_outlier(X_train_num, plot=True):
    fig, ax = plt.subplots(len(X_train_num.columns), 1, figsize=(8, 2))
    if len(X_train_num.columns) == 1:
        ax = [ax]

    for i, col in enumerate(X_train_num.columns):
        ax[i].clear()
        sns.boxplot(x=X_train_num[col], color='skyblue', ax=ax[i])
        
    plt.tight_layout()
    return fig


# 5. Correlation Analysis
def correlation_analysis(df, nilai_skew=0.5):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    corr_matrix_pearson = df.corr(method='pearson')
    sns.heatmap(corr_matrix_pearson, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, ax=ax)
    ax.set_title("Pearson Correlation Matrix")
    plt.tight_layout()
    return fig


# 6. Point-Bisserial Correlation
def correlation_analysis_binary(df, target_col, alpha=0.05, h0=None, h1=None, show=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Calculate the Point-Biserial correlation
    corr, p_val = pointbiserialr(df[target_col], df[target_col].astype('category'))
    ax.bar(['Point-Biserial Correlation'], [corr], color='skyblue')
    ax.set_title(f"Point-Biserial Correlation between {target_col} and target")
    ax.set_ylabel('Correlation Coefficient')

    plt.tight_layout()
    return fig


# 7. Cek persentase missing value pada fitur tertentu
def persentase_missing_value(df_train, df_test, fitur_list):
    fig, ax = plt.subplots(figsize=(10, 7))

    missing_train = df_train[fitur_list].isnull().mean() * 100
    missing_test = df_test[fitur_list].isnull().mean() * 100
    df_missing = pd.DataFrame({'Train': missing_train, 'Test': missing_test})

    df_missing.plot(kind='bar', ax=ax)
    ax.set_title('Percentage of Missing Values for Features')
    ax.set_xlabel('Features')
    ax.set_ylabel('Missing Percentage')

    plt.tight_layout()
    return fig


# 8. Cek persentase dan value tiap kolom
def calculate_value_percentage(df, column, plot=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    value_counts = df[column].value_counts(normalize=True) * 100
    value_counts.plot(kind='bar', ax=ax, color='skyblue')

    ax.set_title(f'Distribution of {column} Values')
    ax.set_xlabel('Values')
    ax.set_ylabel('Percentage')

    plt.tight_layout()
    return fig


# 9. Uji Hipotesis t-test (unknown sample)
def t_test_analysis_with_input(df, target_col, feature_col, alpha=0.05, h0=None, h1=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    group1 = df[df[target_col] == df[target_col].unique()[0]][feature_col]
    group2 = df[df[target_col] == df[target_col].unique()[1]][feature_col]
    
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    
    ax.bar(['t-statistic', 'p-value'], [t_stat, p_val], color='skyblue')
    ax.set_title(f'T-test Results for {feature_col} by {target_col}')
    ax.set_ylabel('Value')

    plt.tight_layout()
    return fig


# 10. plot_line_relationship
def plot_relationship(dataset, x_col, target_cols, kind='line', figsize=(10, 7), custom_colors=None):
    fig, axs = plt.subplots(len(target_cols), 1, figsize=figsize)
    if len(target_cols) == 1:
        axs = [axs]
    
    for i, target_col in enumerate(target_cols):
        ax = axs[i]

        if custom_colors:
            value_to_color = custom_colors
        else:
            unique_values = dataset[target_col].unique()
            colors = sns.color_palette("Set1", len(unique_values))  
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
            sns.barplot(x=dataset[x_col], y=dataset[target_col], ax=ax, color='skyblue')
        elif kind == 'hist':
            sns.histplot(dataset[target_col], ax=ax, color='skyblue')
        elif kind == 'box':
            sns.boxplot(x=dataset[x_col] if x_col else None, y=dataset[target_col], ax=ax, palette=value_to_color)
        elif kind == 'violin':
            sns.violinplot(x=dataset[x_col] if x_col else None, y=dataset[target_col], ax=ax, palette=value_to_color)
        elif kind == 'kde': 
            sns.kdeplot(data=dataset, x=target_col, ax=ax, fill=True, color='skyblue')
        elif kind == 'count':
            sns.countplot(data=dataset, x=target_col, ax=ax)

        ax.set_title(f'{x_col} vs {target_col}')
        ax.set_xlabel(x_col if x_col else '')
        ax.set_ylabel(target_col)

        ax.legend()

    plt.tight_layout()
    return fig


# 11. Annova
def anova_analysis_with_input(df, target_col, feature_col, alpha=0.05, h0=None, h1=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    groups = [df[df[target_col] == category][feature_col] for category in df[target_col].unique()]
    f_stat, p_val = stats.f_oneway(*groups)

    ax.bar(['F-statistic', 'p-value'], [f_stat, p_val], color='skyblue')
    ax.set_title(f'ANOVA Results for {feature_col} by {target_col}')
    ax.set_ylabel('Value')

    plt.tight_layout()
    return fig


# 12. Chi-Square Test
def chi_square_analysis(df, target_col, feature_col, alpha=0.05, h0=None, h1=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    contingency_table = pd.crosstab(df[target_col], df[feature_col])
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
    
    ax.bar(['Chi2-statistic', 'p-value'], [chi2_stat, p_val], color='skyblue')
    ax.set_title(f'Chi-Square Test Results for {feature_col} and {target_col}')
    ax.set_ylabel('Value')

    plt.tight_layout()
    return fig


def evaluate_model_class_report(model, X_train, y_train, X_test, y_test):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)

    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'], ax=ax[0])
    ax[0].set_title('Confusion Matrix - Train Data')

    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'], ax=ax[1])
    ax[1].set_title('Confusion Matrix - Test Data')

    plt.tight_layout()
    return fig
