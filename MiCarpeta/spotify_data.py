import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

file_path="/Users/solrac_201/Desktop/Código SRC/spotify.csv"
df = pd.read_csv(file_path)
df.head()

print("Info del DataFrame")
df.info()
print("\nValores nulos por columna:")
print(df.isnull().sum())

def tracks_by_genre(df, genre):
    return df[df['track_genre'].str.contains(genre, case=False, na=False)][['track_name', 'artists', 'track_genre']]

def top_artists(df, n=10):
    return df['artists'].value_counts().head(n)

def search_titles(df, keyword):
    return df[df['track_name'].str.contains(keyword, case=False, na=False)][['track_name', 'artists', 'track_genre']]

plt.figure()
df['track_genre'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 géneros")
plt.xlabel("Género")
plt.ylabel("Número de tracks")
plt.tight_layout()
plt.show()

columna = 'popularity'

media = df[columna].mean()
moda = df[columna].mode()[0]
mediana = df[columna].median()
desviacion = df[columna].std()

print(f"Media de {columna}: {media}")
print(f"Moda de {columna}: {moda}")
print(f"Mediana de {columna}: {mediana}")
print(f"Desviación estándar de {columna}: {desviacion}")