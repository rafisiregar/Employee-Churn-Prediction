import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import eda_package as edap  # Importing the custom eda package
from scipy import stats

# Load the dataset
dataattrition = pd.read_csv("src/P1M2_rafi_siregar.csv")  # Sesuaikan nama file dataset

def eda1() -> plt.Figure:
    """Histogram & Boxplot distribusi usia karyawan berdasarkan attrition, dengan warna kustom."""
    
    # Custom colors based on Attrition status
    warna = {'Stayed': '#799FCB', 'Left': '#F9665E'}

    # Create the plot using the custom function from eda_package
    fig = edap.plot_relationship(dataattrition, 'Attrition', ['Age'], kind='box', custom_colors=warna)
    
    return fig

def eda2() -> plt.Figure:
    """Plotting the countplot for Job Satisfaction based on Attrition."""
    
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the countplot on the axis
    sns.countplot(data=dataattrition, x='Job Satisfaction', hue='Attrition', ax=ax)

    # Set the title
    ax.set_title('Distribusi Job Satisfaction Berdasarkan Attrition')
    
    # Return the figure object to be used in Streamlit
    return fig

def eda3() -> plt.Figure:
    """Boxplot untuk distribusi lama bekerja dengan peluang memimpin divisi."""
    
    warna1 = {'Yes': '#799FCB', 'No': '#F9665E'}
    fig = edap.plot_relationship(dataattrition, 'Leadership Opportunities', target_cols=['Years at Company'], kind='box', custom_colors=warna1)
    
    return fig

def eda4() -> plt.Figure:
    """Countplot untuk distribusi attrition berdasarkan departemen atau peran pekerjaan."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.countplot(data=dataattrition, x='Job Role', hue='Attrition', palette='Set1', ax=ax)
    ax.set_title('Distribusi Attrition Berdasarkan Job Role')
    
    # Return the figure object
    return fig


def eda5() -> plt.Figure:
    """Boxplot untuk jarak rumah ke tempat kerja berdasarkan attrition."""
    
    warna2 = {'Stayed': '#799FCB', 'Left': '#F9665E'}
    fig = edap.plot_relationship(dataattrition, 'Distance from Home', target_cols=['Attrition'], kind='box', custom_colors=warna2)
    
    return fig

def eda6() -> plt.Figure:
    """Boxplot untuk gaji bulanan berdasarkan attrition."""
    
    warna3 = {'Stayed': '#799FCB', 'Left': '#F9665E'}
    fig = edap.plot_relationship(dataattrition, 'Attrition', target_cols=['Monthly Income'], kind='box', custom_colors=warna3)
    return fig

def eda7() -> plt.Figure:
    """Countplot untuk distribusi job satisfaction berdasarkan level pendidikan."""
    
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the countplot on the axis
    sns.countplot(data=dataattrition, x='Education Level', hue='Job Satisfaction', ax=ax)
    
    # Set the title
    ax.set_title('Distribusi Job Satisfaction Berdasarkan Education Level')
    
    # Return the figure object to be used in Streamlit
    return fig

