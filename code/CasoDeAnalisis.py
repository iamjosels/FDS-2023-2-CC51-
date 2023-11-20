import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm  
tqdm.pandas()

# Cargar el conjunto de datos desde el archivo CSV
df = pd.read_csv('GB_videos.csv')

#A. ¿Qué categorías de videos son las de mayor tendencia?
# Convertir la columna 'publish_time' a tipo datetime
df['publish_time'] = pd.to_datetime(df['publish_time'])

# Asegurarse de que 'publish_time' no tenga información de zona horaria
df['publish_time'] = df['publish_time'].dt.tz_localize(None)

# Calcular las visitas totales por categoría
visitas_por_categoria = df.groupby('category_id')['views'].sum()

# Calcular la cantidad de videos por categoría
cantidad_videos_por_categoria = df.groupby('category_id').size()

# Calcular el promedio de visitas por categoría
promedio_visitas_por_categoria = visitas_por_categoria/cantidad_videos_por_categoria

# Calcular la diferencia de días entre 'publish_time' y '1/11/2023'
df['diferencia_dias'] = (pd.to_datetime('2023-11-01') - df['publish_time']).dt.days

# Calcular la diferencia promedio de días por categoría
promedio_diferencia_dias_por_categoria = df.groupby('category_id')['diferencia_dias'].mean()

# Calcular el promedio de visitas diarias por categoría
promedio_visitas_diarias = promedio_visitas_por_categoria / promedio_diferencia_dias_por_categoria

# Crear un DataFrame con los resultados
resultados = pd.DataFrame({
    'Categoria': promedio_visitas_diarias.index,
    'Promedio_Visitas_Diarias': promedio_visitas_diarias.values
})

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.bar(resultados['Categoria'], resultados['Promedio_Visitas_Diarias'], color='blue')
plt.xlabel('Categoría')
plt.ylabel('Visitas Diarias')
plt.title('Promedio de Visitas Diarias ')

plt.tight_layout()
plt.show()

#B. ¿Qué categorías de videos son los que más gustan? ¿Y las que menos  gustan?
# Calcular el promedio de likes por categoría
promedio_likes_por_categoria = df.groupby('category_id')['likes'].mean().sort_values(ascending=False)
# Crear el gráfico de barras
plt.bar(promedio_likes_por_categoria.index, promedio_likes_por_categoria.values, color='lightcoral')
plt.xlabel('Categoría')
plt.ylabel('Promedio de Likes')
plt.title('Promedio de Likes por Categoría')
plt.show()
print(promedio_likes_por_categoria)

#C.¿Qué categorías de videos tienen la mejor proporción (ratio) de “Me gusta” / “No me gusta”?

# Calcular el ratio de likes por visita
df['likes_per_view'] = df['likes'] / df['views']
# Calcular el promedio del ratio de likes por categoría
ratio_likes_por_categoria = df.groupby('category_id')['likes_per_view'].mean().sort_values(ascending=False)
# Crear el gráfico de barras
plt.bar(ratio_likes_por_categoria.index, ratio_likes_por_categoria.values, color='lightgreen')
plt.xlabel('Categoría')
plt.ylabel('Ratio de Likes por Visita')
plt.title('Ratio de Likes por Categoría')
plt.show()

#D. ¿Qué categorías de videos tienen la mejor proporción (ratio) de “Vistas” /“Comentarios”?

# Agrupa por categoría y calcula las sumas
grouped_data = df.groupby('category_id').agg({'views': 'sum', 'comment_count': 'sum'})
# Calcula el ratio de Vistas/Comentarios
grouped_data['views_to_comments_ratio'] = grouped_data['views'] / grouped_data['comment_count']
# Ordena las categorías por el ratio en orden descendente
sorted_data = grouped_data.sort_values(by='views_to_comments_ratio', ascending=False)
# Muestra las categorías con los mejores ratios
best_categories = sorted_data.head(5) 
# Imprime los resultados en la consola
print("Mejores Categorías de Videos en función del Ratio Vistas/Comentarios:")
print(best_categories)
# Utilizando los datos calculados anteriormente (best_categories)
categories = best_categories.index
ratios = best_categories['views_to_comments_ratio']
# Crear un gráfico de barras horizontal
plt.figure(figsize=(10, 6))
bars = plt.barh(categories, ratios, color='skyblue')
plt.xlabel('Ratio de Vistas / Comentarios')
plt.title('Mejores Categorías de Videos en función del Ratio Vistas/Comentarios')
plt.grid(axis='x')
# Agregar los ratios junto a cada barra
for bar, ratio in zip(bars, ratios):
plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{ratio:.2f}', ha='left', va='center')
# Establecer las etiquetas del eje y como los IDs de las categorías
plt.yticks(categories, labels=["Category ID " + str(category) for category in categories])
# Mostrar el gráfico
plt.show()

#E. ¿Cómo ha cambiado el volumen de los videos en tendencia a lo largo del tiempo?
df = pd.read_csv('GB_videos.csv', parse_dates=['trending_date'], dayfirst=True)
# Convertir trending_date solo a la fecha (sin la hora) para la agregación diaria
df['trending_date'] = df['trending_date'].dt.date
# Contar el número de video_ids únicos para cada trending_date
daily_trending_counts = df.groupby('trending_date')['video_id'].nunique().reset_index()
# Renombrar columnas para mayor claridad
daily_trending_counts.rename(columns={'video_id': 'unique_videos'}, inplace=True)
# Mostrar las primeras filas de los datos agregados
daily_trending_counts.head()
# Graficar el número de videos únicos en tendencia con el tiempo
plt.figure(figsize=(14,7))
plt.plot(daily_trending_counts['trending_date'], daily_trending_counts['unique_videos'], marker='o', linestyle='-', color='blue')
plt.title('Recuento diario de videos únicos en tendencia con el tiempo', fontsize=16)
plt.xlabel('Fecha de tendencia', fontsize=14)
plt.ylabel('Recuento de videos únicos', fontsize=14)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.tight_layout()
plt.legend(['Videos Únicos en Tendencia'])
plt.show()

#F. ¿Qué canales de YouTube son tendencia más frecuentemente? ¿Y cuáles con menos frecuencia?

print(df.head())
# Calculando la frecuencia de cada canal en tendencia
channel_trending_frequency = df['channel_title'].value_counts().reset_index()
channel_trending_frequency.columns = ['channel_title', 'trending_frequency']
# Mostrando los 5 principales canales que tienden con más frecuencia
most_trending_channels = channel_trending_frequency.head(5)
# Mostrando los 5 principales canales que tienden con menos frecuencia
least_trending_channels = channel_trending_frequency.tail(5)
# Mostrando los resultados
print('Channels that trend more frequently:')
print(most_trending_channels)
print('\nChannels that trend less frequently:')
print(least_trending_channels)
# Creando una visualización para comparar la frecuencia de tendencia de los canales
# Seleccionando los 10 principales y los 10 inferiores canales para la visualización
top_channels = channel_trending_frequency.head(10)
bottom_channels = channel_trending_frequency.tail(10)

# Combinando los datos para graficar
combined_channels = pd.concat([top_channels, bottom_channels])
plt.figure(figsize=(10, 8))
sns.barplot(x='trending_frequency', y='channel_title', data=combined_channels, palette='coolwarm')
plt.title('Principales y Menores Canales de YouTube en Tendencia')
plt.xlabel('Frecuencia de Tendencia')
plt.ylabel('Título del Canal')
plt.tight_layout()
plt.show()

#G. ¿En qué Estados se presenta el mayor número de “Vistas”, “Me gusta” y “No me gusta”?

print(df.head())
# Agregando los datos por estado y calculando la suma de Vistas, Me gusta y No me gusta.
state_aggregated = df.groupby('state')['views', 'likes', 'dislikes'].sum().reset_index()
# Encontrando los estados con el mayor número de Vistas, Me gusta y No me gusta
highest_views_state = state_aggregated.loc[state_aggregated['views'].idxmax()]
highest_likes_state = state_aggregated.loc[state_aggregated['likes'].idxmax()]
highest_dislikes_state = state_aggregated.loc[state_aggregated['dislikes'].idxmax()]
# Mostrando los resultados
print('State with the highest number of views:\n', highest_views_state)
print('\nState with the highest number of likes:\n', highest_likes_state)
print('\nState with the highest number of dislikes:\n', highest_dislikes_state)

# Configurando el estilo de las gráficas
sns.set_style('whitegrid')
# Creando un gráfico de barras para los principales estados en cada categoría
fig, axes = plt.subplots(3, 1, figsize=(10, 15))
# Views
sns.barplot(x='views', y='state', data=state_aggregated.nlargest(5, 'views'), ax=axes[0], palette='viridis')
axes[0].set_title('Principales 5 estados con más Vistas')
# Likes
sns.barplot(x='likes', y='state', data=state_aggregated.nlargest(5, 'likes'), ax=axes[1], palette='magma')
axes[1].set_title('Principales 5 estados con más Me gusta')
# Dislikes
sns.barplot(x='dislikes', y='state', data=state_aggregated.nlargest(5, 'dislikes'), ax=axes[2], palette='cubehelix')
axes[2].set_title('Principales 5 estados con más No me gusta')
plt.tight_layout()
plt.show()

#H. ¿Es factible predecir el número de “Vistas” o “Me gusta” o “No me gusta”?

# Selecciona las variables geográficas y la métrica de interés
features = ['lat', 'lon']
target = 'dislikes'  # Cambiar el target(views, like, dislikes)
# Filtra el DataFrame para seleccionar solo las columnas relevantes
data = df[['lat', 'lon', target]]
# Elimina filas con valores faltantes
data = data.dropna()
# Divide los datos en conjuntos de entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
# Prepara los datos de entrenamiento
X_train = train_data[features]
y_train = train_data[target]
# Prepara los datos de prueba
X_test = test_data[features]
y_test = test_data[target]
# Crea un modelo de regresión lineal
model = LinearRegression()
# Entrena el modelo
model.fit(X_train, y_train)
# Realiza predicciones en los datos de prueba
predictions = model.predict(X_test)
# Evalúa el rendimiento del modelo utilizando el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, predictions)
print(f'Error Cuadrático Medio (MSE) para "{target}": {mse}')
# Visualiza las predicciones vs. los valores reales
plt.scatter(y_test, predictions)
plt.xlabel(f'{target} Reales')
plt.ylabel(f'{target} Predichos')
plt.title(f'Predicciones vs. Valores Reales ({target})')
plt.show()

#I. ¿Los videos en tendencia son los que mayor cantidad de comentarios positivos reciben?

# Selecciona las columnas relevantes
relevant_columns = ['views', 'likes']
# Filtra el DataFrame para seleccionar solo las columnas relevantes
data = df[relevant_columns]
# Elimina filas con valores faltantes
data = data.dropna()
# Calcula la correlación
correlation = data['views'].corr(data['likes'])# Visualiza la relación usando un gráfico de dispersión
sns.scatterplot(x='views', y='likes', data=data)
plt.title(f'Correlación: {correlation:.2f}')
plt.xlabel('Número de Vistas')
plt.ylabel('Número de Likes')
plt.show()
# Muestra la correlación en la consola
print(f'Correlación entre "Número de Vistas" y "Número de Likes": {correlation:.2f}')

#(ADICIONALES)
#J. ¿La hora del día en que se publica un video afecta su popularidad?

# Convierte 'publish_time' a formato de fecha y hora
df['publish_time'] = pd.to_datetime(df['publish_time'])
# Extrae la hora del día (solo la hora) en formato de 24 horas
df['hour'] = df['publish_time'].dt.hour
# Agrupa por hora y calcula el promedio de vistas
hourly_stats = df.groupby('hour')['views'].mean()
# Ajusta la configuración de visualización de pandas
pd.set_option('display.float_format', '{:.3f}'.format)
# Imprime los resultados en la consola
print("Promedio de Vistas por Hora del Día de Publicación:")
print(hourly_stats)
# Restaura la configuración de visualización predeterminada
pd.reset_option('display.float_format')
# Visualiza los resultados
plt.figure(figsize=(10, 6))
hourly_stats.plot(kind='bar', color='skyblue')
plt.title('Promedio de Vistas por Hora del Día de Publicación')
plt.xlabel('Hora del Día')
plt.ylabel('Promedio de Vistas')
plt.show()

#K. ¿La presencia de comentarios desactivados o ratings deshabilitados afecta las vistas?

# Carga el conjunto de datos desde el archivo CSV
df = pd.read_csv('GBvideos_cc50_202101.csv') 
# Filtra el DataFrame para excluir videos con datos faltantes
df_clean = df.dropna(subset=['views', 'comments_disabled', 'ratings_disabled'])
# Agrupa por la presencia/ausencia de comentarios desactivados y ratings deshabilitados
grouped_data = df_clean.groupby(['comments_disabled', 'ratings_disabled']).agg({'views': 'mean'}).reset_index()
# Redondea los valores en la columna 'views'
grouped_data['views'] = grouped_data['views'].round()
# Renombra la columna 'ratings_disabled' para que solo diga 'Ratings'
grouped_data.rename(columns={'ratings_disabled': 'Ratings'}, inplace=True)
# Visualiza los resultados con un gráfico de barras apiladas
plt.figure(figsize=(10, 6))
sns.barplot(x='comments_disabled', y='views', hue='Ratings', data=grouped_data, palette="Blues")
plt.title('Promedio de Vistas por Estado de Comentarios y Ratings')
plt.xlabel('Comentarios')
plt.ylabel('Promedio de Vistas')
plt.show()
# Imprime los resultados en el terminal
print("Promedio de Vistas por Estado de Comentarios y Ratings:")
print(grouped_data)

#L. ¿Cuál es el patrón de crecimiento o declive en la cantidad de vistas a medida que avanza el tiempo?

#Asegurar conversión del tipo de dato a datatime
df['trending_date'] = pd.to_datetime(df['trending_date'])

#Gráfico de línea:
plt.figure(figsize=(10, 6))
sns.lineplot(data=df,x='trending_date', y='views')
plt.title('Patrón de crecimiento o declive en vistas a lo largo del tiempo')
plt.xlabel('Fecha de Tendencia')
plt.ylabel('Número de Vistas')
plt.show()

# Agrupar por mes y año y sumar las vistas
views_by_year = df.groupby(df['trending_date'].dt.to_period("M"))['views'].sum().reset_index()
#Imprimir datos por meses
print(views_by_year)

#M. ¿Existe una relación entre la eliminación de videos y el ratio de "No me gusta"? 
# Crear una nueva columna para el ratio de "No me gusta" por visita
df['dislikes_ratio'] = df['dislikes'] / df['views']
# Crear un DataFrame con videos eliminados y el ratio de "No me gusta"
df_elimination_dislikes_ratio = df[['video_error_or_removed', 'dislikes_ratio']]
# Filtrar solo los videos eliminados
videos_eliminados = df_elimination_dislikes_ratio[df_elimination_dislikes_ratio['video_error_or_removed'] == True]
# Filtrar solo los videos no eliminados
videos_no_eliminados = df_elimination_dislikes_ratio[df_elimination_dislikes_ratio['video_error_or_removed'] == False]

# Visualizar la relación usando un gráfico de caja (boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='video_error_or_removed', y='dislikes_ratio', data=df_elimination_dislikes_ratio)
plt.title('Relación entre Eliminación de Videos y Ratio de "No me gusta"')
plt.xlabel('Video Eliminado')
plt.ylabel('Ratio de "No me gusta" por visita')
plt.show()

    