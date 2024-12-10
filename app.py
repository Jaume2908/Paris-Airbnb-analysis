# Importamos librerías
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json
import statsmodels.api as sm


#VARIABLES PARA QUE EL USUARIO PUEDA PONER LAS RUTAS DE DONDE SE ENCUENTRA EL DATASET PRINCIPAL, EL ENLACE A POWERBI Y EL ARCHIVO GEOJSON
#Ruta del dataset principal
ruta_dataset = r"C:\Users\spide\Desktop\Bootcamp\mi_entorno\Moodulo_2\Proyecto_final_modulo_2\Airbnb_Paris.csv"

#Ruta del archivo GeoJSON
ruta_geojson = r"C:\Users\spide\Desktop\Bootcamp\mi_entorno\Moodulo_2\Proyecto_final_modulo_2\\Data_paris_septiembre\neighbourhoods.geojson"

#Ruta del enlace a PowerBI
ruta_powerbi = "https://app.powerbi.com/view?r=eyJrIjoiNGU0MmJiNmMtODQ2Ni00MDI3LThhNjMtZjA5ZDQxNjdmMjY3IiwidCI6IjhhZWJkZGI2LTM0MTgtNDNhMS1hMjU1LWI5NjQxODZlY2M2NCIsImMiOjl9"


#IMPORTAMOS LIBRERÍAS PARA REALIZAR PRUEBAS DE HIPÓTESIS
from scipy.stats import anderson, levene, ttest_ind, mannwhitneyu, kruskal, spearmanr

#FUNCIÓN PARA REALIZAR PRUEBAS DE HIPOTESIS CON NORMALIDAD DETERMINADA POR ANDERSON-DARLING

def diagrama_flujo_test_hipótesis_Anderson_Darling(conjunto1, conjunto2, alpha=0.05):
    """
    Realiza una serie de pruebas estadísticas comenzando con Anderson-Darling
    y devuelve las conclusiones en formato de texto.

    Parámetros:
        conjunto1, conjunto2: listas o arrays unidimensionales de datos.
        alpha: nivel de significancia para las pruebas.

    Retorna:
        conclusiones: lista de strings con las conclusiones.
    """
    # Convertir los conjuntos a arrays unidimensionales si son DataFrames o Series
    if isinstance(conjunto1, pd.DataFrame) or isinstance(conjunto1, pd.Series):
        conjunto1 = conjunto1.values.flatten()
    if isinstance(conjunto2, pd.DataFrame) or isinstance(conjunto2, pd.Series):
        conjunto2 = conjunto2.values.flatten()
        
    # Inicializar lista de conclusiones
    conclusiones = []

    # Prueba de Anderson-Darling
    stat_1, crit_1, sig_1 = anderson(conjunto1)
    stat_2, crit_2, sig_2 = anderson(conjunto2)
    conclusiones.append(f"Estadístico Anderson-Darling (Conjunto 1): {stat_1:.3f}")
    conclusiones.append(f"Estadístico Anderson-Darling (Conjunto 2): {stat_2:.3f}")
    conclusiones.append(f"Valor crítico (5%): {crit_1[2]:.3f}")

    if stat_1 < crit_1[2] and stat_2 < crit_2[2]:  # Usamos el valor crítico para 5%
        conclusiones.append("Ambos conjuntos siguen una distribución normal.")

        # Prueba de Levene para igualdad de varianzas
        stat_levene, p_levene = levene(conjunto1, conjunto2)
        conclusiones.append(f"Prueba de Levene: Estadístico={stat_levene:.3f}, P-valor={p_levene:.3f}")
        
        if p_levene > alpha:
            conclusiones.append("Varianzas iguales. Realizamos prueba t de Student con varianzas iguales.")

            # Prueba t de Student con varianzas iguales
            stat_t, p_t = ttest_ind(conjunto1, conjunto2, equal_var=True)
            conclusiones.append(f"Prueba t de Student: Estadístico={stat_t:.3f}, P-valor={p_t:.3f}")
            if p_t < alpha:
                conclusiones.append("Se rechaza H0: Existen diferencias significativas entre los dos conjuntos.")
            else:
                conclusiones.append("No se rechaza H0: No hay diferencias significativas entre los dos conjuntos.")
        else:
            conclusiones.append("Varianzas diferentes. Realizamos una prueba t de Student con varianzas diferentes.")

            # Prueba t de Student con varianzas diferentes
            stat_t, p_t = ttest_ind(conjunto1, conjunto2, equal_var=False)
            conclusiones.append(f"Prueba t de Student: Estadístico={stat_t:.3f}, P-valor={p_t:.3f}")
            if p_t < alpha:
                conclusiones.append("Se rechaza H0: Existen diferencias significativas entre los dos conjuntos.")
            else:
                conclusiones.append("No se rechaza H0: No hay diferencias significativas entre los dos conjuntos.")
    else:
        conclusiones.append("Ambos conjuntos no siguen una distribución normal. Realizamos una prueba de Mann-Whitney.")

        # Prueba de Mann-Whitney
        stat_mw, p_mw = mannwhitneyu(conjunto1, conjunto2)
        conclusiones.append(f"Prueba de Mann-Whitney: Estadístico={stat_mw:.3f}, P-valor={p_mw:.3f}")
        if p_mw < alpha:
            conclusiones.append("Se rechaza H0: Existen diferencias significativas entre los dos conjuntos.")
        else:
            conclusiones.append("No se rechaza H0: No hay diferencias significativas entre los dos conjuntos.")

    return conclusiones

# FUNCIÓN PARA GENERAR QQ PLOTS
def generar_qqplot(conjunto, titulo, ax):
    sm.qqplot(conjunto, line='s', ax=ax)
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.legend([titulo], loc='upper left', fontsize=10)

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Análisis de datos de alojamientos Airbnb en París",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

#CARGA DEL DATASET
# Función para cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv(ruta_dataset)
    return df

# Cargar datos
df = load_data()

# DF PARA EL ANÁLISIS DE RESEÑAS 
df_reviews = df[df['Review?']=='Con review'][['id','listing_url','review_scores_rating', 'review_scores_accuracy', 
                                                       'review_scores_cleanliness', 'review_scores_checkin', 
                                                       'review_scores_communication', 'review_scores_location', 
                                                       'review_scores_value','price','neighbourhood_cleansed','latitude','longitude',
                                                       'host_is_superhost','host_since','host_id']]

# MENÚ LATERAL
menu_options = [
    "Introducción",
    "Visualización General | Histogramas y Boxplots",
    "Análisis de Precios",
    "Análisis de Reseñas",
    "Análisis de Anfitriones",
    "Panel de control | PowerBI"
]

# Estilo personalizado para el menú lateral
st.markdown(
    """
    <style>
    /* Cambiar el fondo del menú lateral */
    [data-testid="stSidebar"] {
        background-color: #FF5A5F;
    }
    /* Cambiar el color y el estilo de la fuente */
    [data-testid="stSidebar"] * {
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True)
selected_option = st.sidebar.radio("Menú:", menu_options)

# PÁGINA INTRODUCCIÓN
if selected_option == "Introducción":
    # PORTADA CON TÍTULO E IMAGEN
    image_path = r"C:\Users\spide\Desktop\Bootcamp\Paris_airbnb.webp"
    image = Image.open(image_path)
    
    col1, col2, col3 = st.columns([1, 6, 1])  # Centrar contenido
    with col2:
        st.image(image, use_column_width=True)
        st.markdown(
            """
            <h1 style='text-align: center; color: #FF5A5F; font-family: Times New Roman;'>
            Análisis de datos de alojamientos Airbnb en París
            </h1>
            """, 
            unsafe_allow_html=True
        )

    # Subtítulo "Introducción"
    st.markdown(
        """
        ## Introducción
        """, 
        unsafe_allow_html=True
    )

    st.write("""
        En esta aplicación se analizan los datos de los alojamientos de Airbnb en París.
        El conjunto de datos contiene información sobre los alojamientos, como el precio, la ubicación, el número de habitaciones, el número de baños, etc.
        Se ha dividido el análisis en las siguientes secciones:
        - **Visualización General:** Una descripción inicial para explorar características y tendencias generales.
        - **Análisis de Precios:** Exploración detallada de cómo los precios varían según el barrio y otras variables.
        - **Análisis de Reseñas:** Evaluación de la satisfacción de los huéspedes basada en las reseñas.
        - **Análisis de Anfitriones:** Comparación entre superanfitriones y anfitriones regulares.
        -  **Panel de control | PowerBI**: Enlace a un panel de control interactivo en PowerBI en el que el usuario puede visualizar los alojamientos que mejor se adapten a sus necesidades.

    """)
    # Subtítulo "Información del dataset"
    st.markdown("### Información del dataset")

    # Párrafo introductorio del dataset
    st.write("""
        El siguiente dataset se ha obtenido de la página web *insideairbnb.com* y se ha hecho una selección de las siguientes variables:
    """)

    # Lista de variables en formato Markdown
    st.markdown("""
    1. **id**: Identificador único de la propiedad.
    2. **listing_url**: Enlace a la página del alojamiento en Airbnb.
    3. **property_type**: Tipo de propiedad (ej. apartamento, casa, etc.).
    4. **latitude**: Coordenada geográfica de latitud del alojamiento.
    5. **longitude**: Coordenada geográfica de longitud del alojamiento.
    6. **name**: Nombre del alojamiento.
    7. **host_id**: Identificador único del anfitrión.
    8. **host_name**: Nombre del anfitrión.
    9. **host_url**: Enlace al perfil del anfitrión.
    10. **has_availability**: Indica si la propiedad está disponible para reservar.
    11. **host_since**: Año en que el anfitrión se unió a Airbnb.
    12. **neighbourhood_cleansed**: Barrio en el que se encuentra el alojamiento (después de limpieza de datos).
    13. **room_type**: Tipo de habitación (entera, privada, compartida).
    14. **accommodates**: Número de personas que puede alojar.
    15. **host_is_superhost**: Indica si el anfitrión es un superanfitrión.
    16. **bedrooms**: Número de habitaciones en el alojamiento.
    17. **beds**: Número de camas en el alojamiento.
    18. **price**: Precio por noche del alojamiento.
    19. **minimum_nights**: Número mínimo de noches para hacer una reserva.
    20. **maximum_nights**: Número máximo de noches para hacer una reserva.
    21. **number_of_reviews**: Número total de reseñas recibidas.
    22. **instant_bookable**: Indica si el alojamiento es instantáneamente reservable.
    23. **review_scores_rating**: Puntuación general del alojamiento.
    24. **review_scores_accuracy**: Puntuación sobre la precisión de la descripción del alojamiento.
    25. **review_scores_cleanliness**: Puntuación sobre la limpieza del alojamiento.
    26. **review_scores_checkin**: Puntuación sobre el proceso de registro/check-in.
    27. **review_scores_communication**: Puntuación sobre la comunicación con el anfitrión.
    28. **review_scores_location**: Puntuación sobre la ubicación del alojamiento.
    29. **review_scores_value**: Puntuación sobre la relación calidad-precio.
    30. **host_total_listings_count**: Número total de anuncios que tiene el anfitrión.
    31. **bathrooms**: Número de baños en el alojamiento.
    """)

# PÁGINA VISUALIZACIÓN GENERAL
elif selected_option == "Visualización General | Histogramas y Boxplots":
    st.header("Visualización General")
    st.write("""
        En esta sección se generan gráficos de **Boxplots** e **Histogramas** para todas las variables numéricas del conjunto de datos. 
        Esto permite analizar cómo se distribuyen las diferentes características de los alojamientos.
    """)
    st.markdown("""
    <p><span style="color: red; font-weight: bold;">NOTA IMPORTANTE:</span> 
    En las variables de reseñas, los valores 0 indican casos SIN RESEÑA. 
    Estos originalmente eran nulos y fueron reemplazados por 0.
    </p>
    """, unsafe_allow_html=True)

    # Selección del tipo de gráfico
    plot_type = st.radio("Selecciona el tipo de gráfico:", ["Boxplot", "Histograma"])

    # Generar gráficos dinámicos según la selección
    num_vars = df.select_dtypes(include=['float64', 'int64']).columns
    n_cols = 3
    n_rows = (len(num_vars) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
    axes = axes.flatten()

    sns.set_theme(style="whitegrid")

    # Renderizar gráficos de Boxplot
    if plot_type == "Boxplot":
        st.subheader("Gráficos de Boxplot")

        # Preguntar al usuario si desea visualizar los outliers
        show_outliers = st.radio(
            "¿Quieres visualizar los outliers en los gráficos?",
            options=["Sí", "No"],
            index=1  # Por defecto, se selecciona "No"
        )
        
        # Convertir la respuesta del usuario en un valor booleano
        showfliers = True if show_outliers == "Sí" else False

        for i, var in enumerate(num_vars):
            sns.boxplot(data=df, y=var, ax=axes[i], color='#FF5A5F', showfliers=showfliers)
            axes[i].set_title(var, fontsize=14, fontweight='bold')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            axes[i].grid(True, linestyle='--', alpha=0.7)

        # Remover gráficos vacíos
        for j in range(len(num_vars), len(axes)):
            fig.delaxes(axes[j])

        st.pyplot(fig)

    # Renderizar gráficos de Histograma
    elif plot_type == "Histograma":
        st.subheader("Gráficos de Histogramas")
        for i, var in enumerate(num_vars):
            sns.histplot(df[var], kde=True, bins=100, ax=axes[i], color='#FF5A5F', edgecolor='black')
            axes[i].set_title(var, fontsize=14, fontweight='bold')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            axes[i].grid(True, linestyle='--', alpha=0.7)

        # Remover gráficos vacíos
        for j in range(len(num_vars), len(axes)):
            fig.delaxes(axes[j])
        st.pyplot(fig)


    st.markdown("""**IMPORTANTE** en nuestras variables de *number of reviews, minimum_nights 
                ,bahtrooms y host_total_listings_count*, tenemos valores extremadamente altos,
                lo que indica que tenemos outliers en nuestros datos.""")

    #Selección de variable para mostrar el describe() de pandas:
    st.subheader("Resumen estadístico de las variables numéricas (noo aparecen las variables de review, coordenadas y id)")  
    describe_variable = st.selectbox("Selecciona una variable para mostrar su descripción:", ['accommodates','bedrooms','beds','bathrooms','price','minimum_nights','maximum_nights'])

    if describe_variable:
        st.write(f"### Resumen Estadístico para la variable: **{describe_variable}**")
    
    # Obtener la tabla de describe() sin transponer y redondear a 2 decimales
    describe_table = df[describe_variable].describe().round(2)
    
    # Crear tabla en formato HTML centrado
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            {describe_table.to_frame().to_html(classes='table table-bordered', index=True, border=0)}
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.write("""
        ### Conclusiones
        - **Accommodates:** Se observan valores de hasta 30 personas pero muy poco frecuentes, la gran mayoría se encuentran entre 1 y 6.
        - **Bedrooms:** Se observa que la mayoría de alojamientos tienen entre 1 y 2 habitaciones. Los casos amyores a 4 habitaciones són menos frecuentes.
        - **Beds:** La mayoría de alojamientos tienen entre 1 y 2 camas. Los casos con más de 5 camas son menos frecuentes.
        - **Bathrooms:** La mayoría de alojamientos tienen 1 baño, pero también hay muchos alojamientos con 2 baños, la mayoría de casos se distribuyen antes de 5 baños. 
        - **Price:** La mayoría de alojamientos tienen precios menores a 200€, sin embargo el valor máximo al que llegan los alojamientos más caros es a los 2000 $/noche.
        - **Minimum_nights:** Se observan algunas frecuencias cerca de las 400 noches, pero la mayoría se ubican en valores cercanos a 0. No obstante, se observa una gran cantidad de *outliers* lo que indica que hay mucha variabiidad en cuanto a las noches mínimas que se puede alquilar.
        - **Maximum_nights:** La mayoría de alojamientos se ubican entorno a las 400 y 1000 noches.
        - **Price:** El histograma muestra una *apariencia asimetrica positiva* lo que indica que la mayoría de alojamientos tienen precios bajos, pero hay algunos alojamientos con precios muy altos.
        - **Number of reviews**: La myoría de casos tienen entre 1 y 20 reviews (en los alojamientos que tengan review).
        - **Instant_bookable**: La mayoría de alojamientos no son instantáneamente reservables.
        - **Variables de reseñas:** Los alojamientos por lo general tienen más reseñas entre 4 y 5 por lo que a nivel general son reseñas positivas.                 """)


#ÁNALISIS DE PRECIOS

elif selected_option == "Análisis de Precios":
    st.header("Análisis de Precios")
    st.write("""El precio es un factor determinante a la hora de reservar un alojamiento, por lo que en esta sección
             se podrá encontrar el análisis que se ha realizado sobre esta variable y como 
             cambia según el barrio y otras variables.En el siguiente menú deplegable se podrá seleccionar el tipo de análisis realizado:""")


    #PRECIO PROMEDIO POR TIPO DE HABITACIÓN

    tipo_analisis_precio = st.selectbox("Selecciona el tipo de análisis que desee visualizar:",["Precio promedio por tipo de habitación","Variación del precio promedio por barrio", "Impacto de la reserva instantánea en los precios","Factores clave que influyen en el precio"])

    if tipo_analisis_precio == "Precio promedio por tipo de habitación":
        st.subheader("Precio promedio por tipo de habitación")
        st.write("En este análisis se realiza una comparativa de los precios promedio de los alojamientos de Airbnb en París según el tipo de habitación mediante un gráfico de barras.")

        # Cálculo del precio promedio por tipo de habitación
        avg_price_by_room_type = df.groupby('room_type')['price'].mean().reset_index()
        
        # Creamos el gráfico de barras
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x='room_type',
            y='price',
            data=avg_price_by_room_type,
            color='#FF5A5F',
            ax=ax1
        )
        ax1.set_title('Precio promedio por tipo de habitación', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Tipo de habitación', fontsize=12)
        ax1.set_ylabel('Precio promedio', fontsize=12)
        ax1.tick_params(axis='x', labelsize=10)
        ax1.tick_params(axis='y', labelsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Etiquetas de datos
        for index, value in enumerate(avg_price_by_room_type['price']):
            ax1.text(index, value, f'${value:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Mostrar el gráfico en Streamlit
        st.pyplot(fig1)

        st.markdown("Podemos observar que la opción más económica es la habitación compartida (91.10), seguida de la habitación privada (154.59). Curiosamente, la habitación de hotel es más cara que la casa/apartamento entero. Echemos un vistazo a la distribución del precio de la habitación de hotel.")
        
        # Filtramos el df para las habitaciones de hotel.
        hotel_room_prices = df[df['room_type'] == 'Hotel room']['price']
        # Filtramos el df para los apartamentos enteros.
        entire_apt_prices = df[df['room_type'] == 'Entire home/apt']['price']  

        # Creamos los subplots
        fig2, axes = plt.subplots(1, 2, figsize=(20, 6))

        # Histograma para los precios de habitaciones de hotel
        sns.histplot(hotel_room_prices, kde=True, bins=50, ax=axes[0], color='#FF5A5F')
        axes[0].set_title('Distribución de precios para la categoría Hotel room', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Precio ($)', fontsize=12)
        axes[0].set_ylabel('Frecuencia', fontsize=12)
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # Histograma para los precios de apartamentos enteros
        sns.histplot(entire_apt_prices, kde=True, bins=50, ax=axes[1], color='#FF5A5F')
        axes[1].set_title('Distribución de precios para la categoría Entire home/apt', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Precio ($)', fontsize=12)
        axes[1].set_ylabel('Frecuencia', fontsize=12)
        axes[1].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        
        # Mostramos los subplots en Streamlit
        st.pyplot(fig2)

        st.write("Se observa que hay alrededor de 80 apartamentos con un precio superior a los 1000$, lo que hace aumentar el promedio del precio en esta categoría. Por lo que es un indicativo de que, en general las habitaciones de hotel pueden llegar a ser más costosas que los apartamentos enteros.")
        st.markdown("""
        ### Conclusiones
        - La habitación compartida es la opción más económica, seguida de la habitación privada.
        - La habitación de hotel es más cara que la casa/apartamento entero.
        - La distribución de precios para la categoría **Hotel room** muestra que hay alojamientos con precios superiores a los 1000$.
        - La distribución de precios para la categoría **Entire home/apt** muestra que la mayoría de alojamientos tienen precios inferiores a los 500 $.
        """)

    #PRECIO PROMEDIO POR BARRIO

    elif tipo_analisis_precio == "Variación del precio promedio por barrio":
        st.subheader("Variación del precio promedio por barrio")
        st.write("En este análisis se realiza una comparativa de los precios promedio de los alojamientos de Airbnb en París según el barrio mediante un treemap.")
        # Calculamos el precio promedio por barrio
        avg_price_by_neighbourhood = df.groupby('neighbourhood_cleansed')['price'].mean().reset_index()

        # Creamos el treemap con etiquetas de datos
        fig = px.treemap(avg_price_by_neighbourhood, 
                        path=['neighbourhood_cleansed'], 
                        values='price', 
                        title='Precio promedio por barrio', 
                        color='price', 
                        color_continuous_scale=px.colors.sequential.Reds,  # Escala de colores basada en #FF5A5F
                        hover_data={'neighbourhood_cleansed': False, 'price': ':.2f'},
                        width=1200, height=800)  # Ajustar el tamaño de la figura

        # Actualizamos las trazas para mostrar el texto
        fig.update_traces(hovertemplate="<b>%{label}</b><br>Precio promedio: €%{value:.2f}<extra></extra>")

        # Mostramos el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### Conclusiones
        - **Élysée** es el más caro con diferencia, seguido de **Palais-Bourbon** y **Passy**.
        - Entre **Observatoire**, **Popincourt** y **Reulli**, tenemos precios similares rondando los 170-160 $ la noche.
        - Los dos más baratos son **Buttes-Chaumont** y **Menilmontant**.
        """)

    #PRECIO SEGÚN SI ES INSTANTÁNEAMENTE RESERVABLE O NO

    elif tipo_analisis_precio == "Impacto de la reserva instantánea en los precios":
        st.subheader("Impacto de la reserva instantánea en los precios")
        st.write("En este análisis se realiza una comparativa de los precios promedio de los alojamientos de Airbnb en París según si son instantáneamente reservables o no.")
        st.write("En primer lugar visualizaremos mediante un boxplot cómo se distribuyen los precios según si son instantáneamente reservables o no.")
        
        # Crear un boxplot para comparar el precio según si es instantáneamente reservable o no
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='instant_bookable', y='price', data=df, showfliers=False, color='#FF5A5F', ax=ax4)
        ax4.set_xlabel('Instantáneamente Reservable', fontsize=14)
        ax4.set_ylabel('Precio ($)', fontsize=14)
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['No', 'Sí'], fontsize=12)
        ax4.tick_params(axis='y', labelsize=12)
        ax4.grid(True, linestyle='--', alpha=0.7)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig4)

        st.write("""
            A priori, parece que la mediana en los alojamientos que son instantáneamente reservables 
            es ligeramente superior a los que no lo son, al igual que su tercer cuartil, por lo que 
            es esperable que su precio promedio sea un poco superior (sin tener en cuenta los outliers, 
            ya que se estableció `showfliers=False` en el boxplot). 

            Sin embargo, ambas categorías presentan una gran cantidad de outliers, por lo que para tener 
            más precisión en la comparación procederemos a analizar sus promedios mediante pruebas estadísticas.
            En primer lugar con una prueba de Anderson-Daring para comprobar si siguen una distribución normal 
            y posteriormente con un T-test o Mann-Whitney según el resultado de la prueba. Usamos un nivel de
            significancia del 5%.
        """)

        st.markdown("""
            **<span style="color: red;">NOTA:</span>** 
            La función para la prueba de *Anderson-Darling* devuelve un valor estadístico que se compara con su valor crítico para rechazar o no su hipótesis nula, 
            no devuelve un *p-valor*. Por este motivo se puede ver *NaN* en la columna correspondiente al *p-valor*.
            """, unsafe_allow_html=True)

        
        # Creamos los dos conjuntos de datos para comparar los precios
        instant_bookable_false = df[df['instant_bookable'] == 0]['price']
        instant_bookable_true = df[df['instant_bookable'] == 1]['price']

        # Llamar a la función para realizar las pruebas estadísticas
        conclusiones = diagrama_flujo_test_hipótesis_Anderson_Darling(
            instant_bookable_false, 
            instant_bookable_true, 
            alpha=0.05
        )

        # Mostrar las conclusiones en Streamlit
        st.write("### Conclusiones de las Pruebas de Hipótesis")
        for conclusion in conclusiones:
            st.write("- " + conclusion)



        # Introducción
        st.write("""
            ### Análisis de la Normalidad con QQ Plots
            Para evaluar cómo se desvía la distribución de los datos de una distribución normal, 
            utilizamos gráficos QQ (Quantile-Quantile). Estos gráficos comparan los cuantiles de los datos 
            observados con los cuantiles esperados de una distribución normal.

            - **Conjunto 1:** Alojamientos que **no son instantáneamente reservables**.
            - **Conjunto 2:** Alojamientos que **sí son instantáneamente reservables**.

            Si los datos siguen una distribución normal, los puntos deberían alinearse aproximadamente sobre la línea diagonal.
        """)

        # Función para generar QQ plot
        import statsmodels.api as sm

        def generar_qqplot(conjunto, titulo, ax):
            sm.qqplot(conjunto, line='s', ax=ax)
            ax.set_title(titulo, fontsize=12, fontweight='bold')
            ax.legend([titulo], loc='upper left', fontsize=10)

        # Preparar los datos
        instant_bookable_false = df[df['instant_bookable'] == 0]['price'].values
        instant_bookable_true = df[df['instant_bookable'] == 1]['price'].values

        # Crear figura y subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # Generar QQ plots para ambos conjuntos
        generar_qqplot(instant_bookable_false, 'QQ plot para el conjunto 1 (No instantáneamente reservable)', axes[0])
        generar_qqplot(instant_bookable_true, 'QQ plot para el conjunto 2 (Instantáneamente reservable)', axes[1])

        plt.tight_layout()

        # Mostrar el QQ plot en Streamlit
        st.pyplot(fig)

        st.write("Se puede observar que los puntos se distribuyen diferente de la diagonal por lo que nuestros datos no siguen una distribución normal.")

        st.write("Ahora veamos como queda el promedio de los precios según si son instantáneamente reservables o no en un gráfico de barras.")

        # Calculamos el precio promedio según si es instantáneamente reservable o no
        avg_price_instant_bookable = df.groupby('instant_bookable')['price'].mean().reset_index()

        # Creamos un gráfico de barras con Seaborn
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(x='instant_bookable', y='price', data=avg_price_instant_bookable, palette=['#FF5A5F'], ax=ax3)
        ax3.set_title('Precio promedio según si es instantáneamente reservable', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Instantáneamente Reservable', fontsize=14)
        ax3.set_ylabel('Precio Promedio ($)', fontsize=14)
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['No', 'Sí'], fontsize=12)
        ax3.tick_params(axis='y', labelsize=12)
        ax3.grid(True, linestyle='--', alpha=0.7)

        # Añadimos etiquetas de datos
        for index, value in enumerate(avg_price_instant_bookable['price']):
            ax3.text(index, value, f'${value:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig3)

        st.write("""
            ### Conclusión
            Los resultados obtenidos muestran que los apartamentos que son **instantáneamente reservables** tienen precios 
            más altos en promedio que los que no lo son. Esta diferencia puede deberse a la conveniencia adicional que 
            ofrecen a los usuarios, permitiendo una reserva inmediata sin necesidad de confirmación adicional por parte del anfitrión. 
            """)
        
    #FACTORES CLAVE QUE INFLUYEN EN EL PRECIO

    elif tipo_analisis_precio == "Factores clave que influyen en el precio":
        st.subheader("Factores clave que influyen en el precio")
        
        # Introducción
        st.write("""
            En esta sección analizamos los factores que más influyen en el precio de los alojamientos. 
            Para ello, calculamos la **matriz de correlación** de la variable `price` con otras variables numéricas del conjunto de datos.
            Esto nos permite identificar qué variables tienen una relación más fuerte con el precio.
            
            A continuación, se presenta un **mapa de calor** con las correlaciones de las variables numéricas respecto al precio.
        """)

        # Crear una copia del dataframe y procesar
        df_copy = df.copy()
        df_copy = pd.get_dummies(df_copy, columns=['neighbourhood_cleansed'], drop_first=True, dtype=int)

        # Seleccionar columnas numéricas
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

        # Calcular matriz de correlación
        correlation_matrix = df_copy[numeric_cols].corr()
        correlation_matrix_price = correlation_matrix[['price']]

        # Crear mapa de calor
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix_price, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title('Matriz de Correlación para la Variable Price', fontsize=16, fontweight='bold')
        
        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

        # Conclusiones
        st.markdown("""
            ### Conclusiones
            - Los factores que más influyen en el precio son el número de **viajeros** (`accommodates`), **habitaciones** (`bedrooms`), 
            **camas** (`beds`) y **baños** (`bathrooms`). Estas variables tienen una correlación positiva y significativa con el precio.
            - Aunque los barrios no tienen una correlación muy fuerte con el precio, aquellos con precios promedio más elevados, 
            como **Passy** y **Élysée**, presentan una correlación más notable en comparación con otros barrios.
            - Este análisis sugiere que tanto las características del alojamiento como la ubicación contribuyen de manera significativa 
            al precio, aunque las primeras parecen tener un impacto más directo.
        """)

#ÁNALISIS DE RESEÑAS
elif selected_option == "Análisis de Reseñas":
    st.header("Análisis de Reseñas")
    st.write("""
    Las reseñas son un factor crucial para evaluar la satisfacción de los huéspedes y la calidad del servicio ofrecido por los alojamientos. 
    En esta sección se podrá encontrar el análisis que se ha realizado sobre esta variable y como cambia según el barrio y otras variables.En el siguiente menú deplegable se podrá seleccionar el tipo de análisis realizado:
    """)
    #Selección de tipo de análisis de reseñas:
    tipo_analisis_resenas = st.selectbox("Selecciona el tipo de análisis que desee visualizar:",["Barrios con mejores puntuaciones promedio","Factores relacionados con las puntuaciones generales","Top 20 alojamientos con mejor relación calidad-precio","Relación entre antigüedad del anfitrión y promedio de reseñas"])

    #BARRIOS CON MEJORES PUNTUACIONES PROMEDIO
    if tipo_analisis_resenas == "Barrios con mejores puntuaciones promedio":
        st.subheader("Barrios con mejores puntuaciones promedio")
        
        # Introducción
        st.write("""
            En esta sección exploramos cuáles son los barrios con las mejores puntuaciones promedio en las reseñas. 
            Este análisis permite identificar las áreas donde los huéspedes han tenido experiencias más satisfactorias, 
            basándonos en las valoraciones otorgadas. 

            A continuación, se presenta un gráfico que muestra los barrios ordenados de mayor a menor según su puntuación promedio.
        """)

        # Calcular la puntuación promedio de reviews por barrio
        avg_reviews_by_neighbourhood = df_reviews.groupby('neighbourhood_cleansed')['review_scores_rating'].mean().reset_index()

        # Ordenar los barrios por la puntuación promedio de reviews
        avg_reviews_by_neighbourhood = avg_reviews_by_neighbourhood.sort_values(by='review_scores_rating', ascending=False)

        # Crear el gráfico de barras horizontales
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x='review_scores_rating', 
            y='neighbourhood_cleansed', 
            data=avg_reviews_by_neighbourhood, 
            palette='magma', 
            ax=ax
        )
        ax.set_title('Puntuación promedio de reviews por barrio', fontsize=16, fontweight='bold')
        ax.set_xlabel('Puntuación promedio de reviews', fontsize=12)
        ax.set_ylabel('Barrio', fontsize=12)

        # Añadir etiquetas de datos
        for index, value in enumerate(avg_reviews_by_neighbourhood['review_scores_rating']):
            ax.text(value, index, f'{value:.2f}', va='center', fontsize=10)

        plt.tight_layout()

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
        st.markdown("""
            No se observan diferencias claras a simple vista entre el promedio de reseñas de los diferentes barrios. 
            Sin embargo, para justificar esta observación sería necesario realizar una prueba estadística. Esto plantea la pregunta: 
            ***¿Existe una diferencia significativa entre el promedio de reseñas de los diferentes barrios?***

            El promedio de críticas más elevado se encuentra en **Menilmontant**, lo que indica que los alojamientos en este barrio 
            han brindado una experiencia satisfactoria en general.

            Por otro lado, resulta curioso que los alojamientos en **Élysée**, siendo de los más caros, tengan una peor crítica promedio. 
            Esto plantea otra cuestión interesante: ***¿Existe una relación entre el precio y las reseñas?***
        """)

        # Subtítulo
        st.subheader("¿Existe una diferencia significativa entre el promedio de reseñas de los diferentes barrios?")

        # Introducción
        st.write("""
            Para determinar si existe una diferencia significativa entre el promedio de reseñas de los diferentes barrios, 
            primero evaluamos si la variable `review_scores_rating` sigue una distribución normal. 
            Para ello, aplicamos la prueba estadística de **Anderson-Darling**.

            **Hipótesis:**
            - H0: La variable `review_scores_rating` sigue una distribución normal.
            - H1: La variable `review_scores_rating` no sigue una distribución normal.

            A continuación, se muestran los resultados de la prueba.
        """)

        # Comprobación de normalidad con Anderson-Darling
        stat, crit, sig = anderson(df['review_scores_rating'].dropna())

        # Crear una tabla para los resultados con solo las métricas necesarias
        resultados_normalidad = pd.DataFrame([
            ["Estadístico", round(stat, 3)],
            ["Valor Crítico (5%)", round(crit[2], 3)],
            ["Conclusión", "Se acepta H0: Distribución Normal" if stat < crit[2] else "Se rechaza H0: No sigue una Distribución Normal"]
        ], columns=["Métrica", "Resultado"])  # Definir encabezados directamente en el DataFrame

        # Estilizar la tabla en HTML
        st.markdown("""
            <style>
                .table-container {
                    display: flex;
                    justify-content: center;
                }
                table {
                    border-collapse: collapse;
                    width: 50%;
                    text-align: center;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: center;
                    vertical-align: middle;  /* Forzar que el contenido esté alineado verticalmente */
                }
                th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                td {
                    height: 50px;  /* Ajustar la altura mínima para todas las celdas */
                }
            </style>
        """, unsafe_allow_html=True)

        # Mostrar la tabla en Streamlit sin índices
        st.markdown(f"""
        <div class="table-container">
            {resultados_normalidad.to_html(index=False, header=True, escape=False)}
        </div>
        """, unsafe_allow_html=True)


        # Interpretación
        if stat < crit[2]:
            st.markdown("""
                **Interpretación:**
                - La variable `review_scores_rating` sigue una distribución normal al nivel de significancia del 5%. 
                - Por lo tanto, podríamos considerar utilizar pruebas paramétricas para evaluar diferencias significativas entre barrios.
            """)
        else:
            st.markdown("""
                **Interpretación:**
                - La variable `review_scores_rating` no sigue una distribución normal al nivel de significancia del 5%. 
                - Esto sugiere la necesidad de utilizar pruebas no paramétricas para evaluar diferencias significativas entre barrios.
            """)

        st.write("""
            Para evaluar si existen diferencias significativas entre los barrios en términos de las puntuaciones promedio de reseñas, 
            utilizamos la prueba no paramétrica de **Kruskal-Wallis**. Esta prueba es adecuada ya que las puntuaciones de reseñas no siguen una distribución normal.

            **Hipótesis:**
            - H0: Las puntuaciones promedio de reseñas son iguales entre todos los barrios.
            - H1: Al menos un barrio tiene una puntuación promedio de reseñas diferente.
        """)

        # Creamos listas de reseñas por barrio
        reviews_por_barrio = []
        for barrio in df['neighbourhood_cleansed'].unique():
            reviews_por_barrio.append(df[df['neighbourhood_cleansed'] == barrio]['review_scores_rating'].dropna())

        # Realizar la prueba de Kruskal-Wallis
        stat, p_value = kruskal(*reviews_por_barrio)

        # Crear una tabla para los resultados
        resultados_kruskal = pd.DataFrame([
            ["Estadístico", round(stat, 3)],
            ["p-valor", round(p_value, 3)],
            ["Conclusión", "Se rechaza H0: Hay diferencias significativas" if p_value < 0.05 else "No se rechaza H0: No hay diferencias significativas"]
        ], columns=["Métrica", "Resultado"])

        # Estilizar la tabla en HTML
        st.markdown("""
            <style>
                .table-container {
                    display: flex;
                    justify-content: center;
                }
                table {
                    border-collapse: collapse;
                    width: 50%;
                    text-align: center;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: center;
                    vertical-align: middle;  /* Forzar que el contenido esté alineado verticalmente */
                }
                th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                td {
                    height: 50px;  /* Ajustar la altura mínima para todas las celdas */
                }
            </style>
        """, unsafe_allow_html=True)

        # Mostrar la tabla en Streamlit sin índices
        st.markdown(f"""
        <div class="table-container">
            {resultados_kruskal.to_html(index=False, header=True, escape=False)}
        </div>
        """, unsafe_allow_html=True)

        # Interpretación de los resultados
        if p_value < 0.05:
            st.markdown("""
                **Interpretación:**
                - Hay diferencias significativas en las puntuaciones promedio de reseñas entre los barrios.
            """)
        else:
            st.markdown("""
                **Interpretación:**
                - No hay diferencias significativas en las puntuaciones promedio de reseñas entre los barrios.
            """)
        # Subtítulo
        st.subheader("¿Existe una relación entre el precio y las reseñas?")

        # Introducción
        st.write("""
            Para analizar si existe una relación significativa entre el precio de los alojamientos y las puntuaciones promedio de las reseñas, 
            hemos generado un gráfico de dispersión para observar tendencias iniciales. También se ha calculado un modelo de **regresión lineal** 
            para representar la relación y el coeficiente de determinación (R²) para evaluar la proporción de variabilidad explicada por el modelo.

            A continuación, se muestra el gráfico de dispersión con la línea de tendencia:
        """)
        # Variables para el análisis
        X = df_reviews[['price']]
        y = df_reviews['review_scores_rating']

        # Crear y ajustar el modelo de regresión lineal
        model = LinearRegression()
        model.fit(X, y)

        # Predicciones y cálculo de R2
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        # Crear el gráfico de dispersión
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df_reviews['price'], df_reviews['review_scores_rating'], alpha=0.5, color='#FF5A5F', label='Datos')
        ax.plot(df_reviews['price'], y_pred, color='blue', linewidth=2, label=f'Línea de tendencia (R²={r2:.2f})')
        ax.set_title('Relación entre Precio y Puntuación de Reviews', fontsize=16, fontweight='bold')
        ax.set_xlabel('Precio ($)', fontsize=14)
        ax.set_ylabel('Puntuación de Reviews', fontsize=14)
        ax.legend(fontsize=12)

        # Ajustar los límites de los ejes
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        # Estilizar el gráfico
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
        st.markdown("""
            **Observaciones iniciales:**
            - La mayoría de las puntuaciones altas (4-5) se concentran en alojamientos con precios por debajo de 1000 $.
            - Precios superiores a 1000 $ tienden a estar asociados con puntuaciones de reseñas por encima de 3, que es cercano al promedio.
            - No parece haber una relación lineal clara entre el precio y las reseñas. 

            Para confirmar esta observación, realizaremos una **prueba de correlación de Spearman**, ya que las variables no siguen una distribución normal.
        """)


        # Realizar la prueba de correlación de Spearman
        corr, p_value = spearmanr(df_reviews['price'], df_reviews['review_scores_rating'])

        st.markdown("""
            **Hipótesis de la prueba de correlación de Spearman:**
            - **H0:** No existe una correlación significativa entre el precio y las reseñas.
            - **H1:** Existe una correlación significativa entre el precio y las reseñas.
        """)

        # Crear una tabla para los resultados
        resultados_spearman = pd.DataFrame([
            ["Coeficiente de Correlación (Spearman)", round(corr, 3)],
            ["p-valor", round(p_value, 3)],
            ["Conclusión", "Se acepta H0: No hay relación significativa" if p_value > 0.05 else "Se rechaza H0: Existe una relación significativa"]
        ], columns=["Métrica", "Resultado"])

        # Estilizar la tabla
        st.markdown("""
            <style>
                .table-container {
                    display: flex;
                    justify-content: center;
                }
                table {
                    border-collapse: collapse;
                    width: 50%;
                    text-align: center;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: center;
                    vertical-align: middle;
                }
                th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                td {
                    height: 50px;
                }
            </style>
        """, unsafe_allow_html=True)

        # Mostrar la tabla en Streamlit
        st.markdown(f"""
        <div class="table-container">
            {resultados_spearman.to_html(index=False, header=True, escape=False)}
        </div>
        """, unsafe_allow_html=True)

        # Interpretación
        if p_value > 0.05:
            st.markdown("""
                **Interpretación:**
                - No se encontró una relación significativa entre el precio y las reseñas.
                - Esto sugiere que otros factores, como la ubicación o la calidad del alojamiento, podrían ser más determinantes en las puntuaciones.
            """)
        else:
            st.markdown("""
                **Interpretación:**
                - Existe una relación significativa entre el precio y las reseñas.
                - Aunque la relación no es lineal, puede indicar que el precio tiene algún impacto en las puntuaciones de reseñas.
            """)

        st.subheader("Clasificación del Precio y su Relación con las Reseñas")
        st.write("""
            Para visualizar mejor la relación entre el precio y las puntuaciones de reseñas, hemos clasificado los alojamientos 
            en diferentes rangos de precio, y calculado el promedio de puntuaciones en cada rango. A continuación, se presenta un gráfico 
            de líneas que ilustra cómo las puntuaciones promedio de reseñas varían según la clasificación de precios.
        """)

        # Clasificación de precios
        bins = [0, 50, 90, 139, 230, 500, float('inf')]
        labels = ['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto', 'Extremadamente alto']

        df_reviews['price_classification'] = pd.cut(df_reviews['price'], bins=bins, labels=labels, right=False)

        # Calcular el promedio de review_scores_rating por price_classification
        avg_review_by_price_class = df_reviews.groupby('price_classification')['review_scores_rating'].mean().reset_index()

        # Crear el gráfico de líneas
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            x='price_classification',
            y='review_scores_rating',
            data=avg_review_by_price_class,
            marker='o',
            color='#FF5A5F',
            ax=ax
        )
        ax.set_xlabel('Clasificación de Precio', fontsize=14)
        ax.set_ylabel('Promedio de Puntuación de Reviews', fontsize=14)

        # Añadir etiquetas de datos al gráfico
        for index, value in enumerate(avg_review_by_price_class['review_scores_rating']):
            ax.text(index, value, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

        # Ajustar el diseño del gráfico
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

        # Observación final
        st.write("""
            - Podemos observar que a precios mayores, las reviews promedio tienden a ser más positivas. 
            - Esto podría indicar que los alojamientos más costosos ofrecen una mejor experiencia o servicios que los huéspedes valoran más.
        """)

    #FACTORES RELACIONADOS CON LAS PUNTUACIONES GENERALES
    elif tipo_analisis_resenas == "Factores relacionados con las puntuaciones generales":
        st.subheader("Factores relacionados con las puntuaciones generales")

        # Introducción
        st.write("""
            En este análisis exploramos qué factores están más relacionados con la puntuación general (`review_scores_rating`) 
            de los alojamientos en Airbnb. Utilizamos una matriz de correlación calculada con el método de Kendall para 
            identificar la fuerza de relación entre las puntuaciones generales y otros factores como la limpieza, la comunicación, 
            la ubicación y la relación calidad-precio.

            A continuación, se presenta un mapa de calor que muestra las correlaciones entre las diferentes columnas de reseñas.
        """)

        # Crear lista de columnas de review
        review_columns = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 
                        'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 
                        'review_scores_value']

        # Calcular matriz de correlación de Kendall
        correlation_matrix = df_reviews[review_columns].corr(method='kendall')

        # Crear el mapa de calor
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

        # Conclusión
        st.markdown("""
            ### Conclusión
            - Los **factores más influyentes** en la puntuación general (`review_scores_rating`) de los alojamientos son:
            - **Precisión del anuncio** (`review_scores_accuracy`): 0.64
            - **Relación calidad-precio** (`review_scores_value`): 0.61
            - **Limpieza** (`review_scores_cleanliness`): 0.57
            - Esto sugiere que los huéspedes valoran especialmente que la descripción del alojamiento sea precisa, que el precio 
            esté acorde con lo que se ofrece, y que el alojamiento esté limpio.
        """)
    elif tipo_analisis_resenas == "Top 20 alojamientos con mejor relación calidad-precio":
        st.subheader("Top 20 alojamientos con mejor relación calidad-precio")

        # Descripción introductoria
        st.write("""
            En esta sección analizamos los alojamientos con la mejor relación calidad-precio en París. 
            Hemos seleccionado los 20 alojamientos con la **puntuación más alta en relación calidad-precio (`review_scores_value`)**
            y el **precio más bajo** (`price`). Estos alojamientos se muestran en un mapa de dispersión junto con una tabla descriptiva 
            que incluye información relevante.
        """)

        # Ordenar el DataFrame y filtrar el top 20
        df_reviews_sorted = df_reviews.sort_values(by=['review_scores_value', 'price'], ascending=[False, True])
        top_20_reviews = df_reviews_sorted.head(20)

        # Crear el mapa de dispersión
        fig = px.scatter_mapbox(
            top_20_reviews,
            lat='latitude',
            lon='longitude',
            size='price',
            hover_name='neighbourhood_cleansed',
            hover_data={'latitude': False, 'longitude': False, 'price': True},
            labels={'review_scores_value': 'Relación Calidad-Precio', 'price': 'Precio ($)'},
            size_max=15,
            zoom=10,
            mapbox_style="carto-positron",
            title='Top 20 alojamientos según la relación calidad-precio más alta y precio más bajo',
            color_discrete_sequence=['#FF5A5F']
        )

        # Mostrar el mapa en Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Crear la tabla descriptiva
        tabla_top_20 = top_20_reviews[['id', 'listing_url', 'neighbourhood_cleansed', 'price', 'review_scores_value']]
        tabla_top_20.columns = ['ID', 'URL del Anuncio', 'Barrio', 'Precio ($)', 'Relación Calidad-Precio']

        # Estilizar la tabla en HTML
        st.markdown("""
            <style>
                .table-container {
                    display: flex;
                    justify-content: center;
                }
                table {
                    border-collapse: collapse;
                    width: 80%;
                    text-align: center;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: center;
                    vertical-align: middle;
                }
                th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                td {
                    height: 50px;
                }
            </style>
        """, unsafe_allow_html=True)

        # Mostrar la tabla en Streamlit sin índices
        st.markdown(f"""
        <div class="table-container">
            {tabla_top_20.to_html(index=False, escape=False)}
        </div>
        """, unsafe_allow_html=True)

    #RELACIÓN ENTRE ANTIGÜEDAD DEL ANFITRIÓN Y PROMEDIO DE RESEÑAS

    elif tipo_analisis_resenas == "Relación entre antigüedad del anfitrión y promedio de reseñas":
        st.subheader("Relación entre antigüedad del anfitrión y promedio de reseñas")

        # Introducción
        st.write("""
            En este análisis exploramos la relación entre la antigüedad de los anfitriones en Airbnb y las puntuaciones promedio 
            de reseñas que reciben. Utilizamos el año en que cada anfitrión se unió a Airbnb (`host_since`) para calcular tanto 
            el número de anfitriones únicos por año como el promedio de puntuaciones de reseñas en cada año. 
            A continuación, se presenta un gráfico combinado para visualizar estas tendencias.
        """)

        # Preparación de los datos
        df_reviews['host_since'] = pd.to_datetime(df_reviews['host_since'])
        df_reviews['host_since_year'] = df_reviews['host_since'].dt.year

        # Calcular la cantidad de hosts únicos por año
        unique_hosts_by_year = df_reviews.groupby('host_since_year')['host_id'].nunique().reset_index()
        unique_hosts_by_year.rename(columns={'host_id': 'unique_hosts_count'}, inplace=True)

        # Calcular el promedio de review_scores_rating por año
        avg_review_by_host_since_year = df_reviews.groupby('host_since_year')['review_scores_rating'].mean().reset_index()

        # Crear el gráfico combinado
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Gráfico de líneas (Promedio de review_scores_rating)
        color_line = 'tab:blue'
        ax1.set_xlabel('Año que se hizo host', fontsize=14)
        ax1.set_ylabel('Promedio de reseñas', color=color_line, fontsize=14)
        ax1.plot(
            avg_review_by_host_since_year['host_since_year'], 
            avg_review_by_host_since_year['review_scores_rating'], 
            marker='o', color=color_line, label='Promedio de reseñas', linewidth=2
        )
        ax1.tick_params(axis='y', labelcolor=color_line)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Crear un segundo eje Y para la cantidad de hosts únicos
        ax2 = ax1.twinx()  # Comparte el eje X con ax1
        color_bar = '#FF5A5F'
        ax2.set_ylabel('Cantidad de hosts', color=color_bar, fontsize=14)
        ax2.bar(
            unique_hosts_by_year['host_since_year'], 
            unique_hosts_by_year['unique_hosts_count'], 
            color=color_bar, alpha=0.6, label='Cantidad de hosts'
        )
        ax2.tick_params(axis='y', labelcolor=color_bar)

        # Añadir leyendas
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), fontsize=12)

        # Ajustar el diseño
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
        st.markdown(""" 
                    ### Conclusión: 
                    El promedio de reseñas tiende a ser más alto para los anfitriones con mayor antigüedad, disminuyendo a medida que los anfitriones son más recientes.""")
        
#ÁNALISIS DE ANFITRIONES

elif selected_option == "Análisis de Anfitriones":
    st.header("Análisis de Anfitriones")
    st.write("""
            Los anfitriones son una parte fundamental de la plataforma de Airbnb, ya que son quienes ofrecen sus alojamientos y servicios a los huéspedes. 
             En el presente análisis se estudiarán los **propietarios con más propiedades en Airbnb en París** y se realizará un estudio a los **super anfitriones**. """)
    #Selección de tipo de análisis de anfitriones:
    tipo_analisis_anfitriones = st.selectbox("Selecciona el tipo de análisis que desee visualizar:",["Top 20 anfitriones con más propiedades","Ánalisis de los super anfitriones"])

    #TOP 20 ANFITRIONES CON MÁS PROPIEDADES

    if tipo_analisis_anfitriones == "Top 20 anfitriones con más propiedades":
        st.subheader("Top 20 anfitriones con más propiedades")
        # Introducción
        st.write("""
            En esta sección se presentan los 20 anfitriones con más propiedades en Airbnb en París. 
            Se ha calculado el número total de propiedades que cada anfitrión posee y se han seleccionado los 20 anfitriones con más propiedades. 
            A continuación, se muestra un gráfico de barras que ilustra el número de propiedades de los 20 anfitriones seleccionados.
        """)
        # GRÁFICO DE BARRAS CON LOS 20 ANFITRIONES CON MÁS PROPIEDADES
        # Agrupamos los datos por 'host_id' y contamos el número de propiedades para cada anfitrión
        top_20_anfitriones_con_mas_propiedades = df.groupby(['host_id', 'host_name']).agg({
            'id': 'count',
            'host_since': 'first'
        }).reset_index()

        # Renombramos la columna 'id' a 'total_listings_count' para mayor claridad
        top_20_anfitriones_con_mas_propiedades.rename(columns={'id': 'Número de Propiedades'}, inplace=True)

        # Ordenamos los datos por 'total_listings_count' de forma descendente y seleccionamos los 20 principales
        top_20_anfitriones_con_mas_propiedades = top_20_anfitriones_con_mas_propiedades.sort_values(
            by='Número de Propiedades', ascending=False
        ).head(20)

        # Crear un gráfico de barras para visualizar los 20 anfitriones con más propiedades
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x='Número de Propiedades', 
            y='host_name', 
            data=top_20_anfitriones_con_mas_propiedades, 
            color='#FF5A5F'
        )
        ax.set_title('Top 20 propietarios con más propiedades', fontsize=16, fontweight='bold')
        ax.set_xlabel('Número total de propiedades', fontsize=14)
        ax.set_ylabel('Nombre del anfitrión', fontsize=14)
        plt.tight_layout()

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

        #TABLA DESCRIPTIVA DE LOS 20 ANFITRIONES CON MÁS PROPIEDADES
        # Seleccionar las columnas clave para la tabla
        tabla_top_20 = top_20_anfitriones_con_mas_propiedades[['host_id', 'host_name', 'Número de Propiedades', 'host_since']]
        tabla_top_20.rename(columns={
            'host_id': 'ID del Anfitrión',
            'host_name': 'Nombre del Anfitrión',
            'host_since': 'Fecha de Registro'
        }, inplace=True)

        # Estilizar la tabla en HTML
        st.markdown("""
            <style>
                .table-container {
                    display: flex;
                    justify-content: center;
                }
                table {
                    border-collapse: collapse;
                    width: 80%;
                    text-align: center;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: center;
                    vertical-align: middle;
                }
                th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                td {
                    height: 50px;
                }
            </style>
        """, unsafe_allow_html=True)

        # Mostrar la tabla en Streamlit sin índices
        st.markdown(f"""
        <div class="table-container">
            {tabla_top_20.to_html(index=False, escape=False)}
        </div>
        """, unsafe_allow_html=True)

        st.write("Se puede observar que hay una gran cantidad de organizaciones de gestión de alojamientos entre los 20 principales anfitriones, lo que sugiere que estas organizaciones gestionan múltiples propiedades en nombre de los propietarios.")

    #ÁNALISIS DE LOS SUPER ANFITRIONES

    elif tipo_analisis_anfitriones == "Ánalisis de los super anfitriones":
        st.subheader("Ánalisis de los super anfitriones")
        # Introducción
        st.write("""
                 El análisis de los super anfitriones es crucial para entender el impacto que tienen 
                 en la plataforma de Airbnb. Es esta sección veremos el porcentaje de ellos que hay
                 según el barrio, y si afectan al precio y a las reseñas de los alojamientos.
                 """)
        
        # OPCIONES DE SELECCIÓN PARA VER EL TIPO DE ANÁLISIS DE LOS SUPER ANFITRIONES CON UN ST.RADIO
        tipo_analisis_super_anfitriones = st.radio("Selecciona el tipo de análisis que desee visualizar:", ["Porcentaje de super anfitriones por barrio", "Relación entre super anfitriones y precio", "Relación entre super anfitriones y reseñas"])

        # PORCENTAJE DE SUPER ANFITRIONES POR BARRIO
        if tipo_analisis_super_anfitriones == "Porcentaje de super anfitriones por barrio":
            st.subheader("Porcentaje de super anfitriones por barrio")
            # Introducción
            st.write("""
                    En esta sección veremos el porcentaje de super anfitriones por barrio en París mediante un visual de mapa cloroplético.
                    """)


            # Cargar el archivo GeoJSON
            with open(ruta_geojson, encoding='utf-8') as f:
                geojson = json.load(f)

            # Calcular el número total de anfitriones por barrio
            total_anfitriones_por_barrio = df.groupby('neighbourhood_cleansed')['host_id'].nunique().reset_index()
            total_anfitriones_por_barrio.rename(columns={'host_id': 'total_anfitriones'}, inplace=True)

            # Calcular el número de superanfitriones por barrio
            superanfitriones_por_barrio = df[df['host_is_superhost'] == 't'].groupby('neighbourhood_cleansed')['host_id'].nunique().reset_index()
            superanfitriones_por_barrio.rename(columns={'host_id': 'superanfitriones'}, inplace=True)

            # Unir los dos DataFrames
            anfitriones_por_barrio = pd.merge(total_anfitriones_por_barrio, superanfitriones_por_barrio, on='neighbourhood_cleansed', how='left')
            anfitriones_por_barrio['superanfitriones'].fillna(0, inplace=True)

            # Calcular el porcentaje de superanfitriones por barrio
            anfitriones_por_barrio['porcentaje_superanfitriones'] = (anfitriones_por_barrio['superanfitriones'] / anfitriones_por_barrio['total_anfitriones']) * 100

            # Crear el mapa cloroplético
            fig = px.choropleth_mapbox(
                anfitriones_por_barrio,
                geojson=geojson,  # Archivo GeoJSON cargado
                locations='neighbourhood_cleansed',
                featureidkey="properties.neighbourhood",  # Ajusta esto según la estructura de tu archivo GeoJSON
                color='porcentaje_superanfitriones',
                color_continuous_scale=px.colors.sequential.Pinkyl,  # Escala de color con más tonalidades
                mapbox_style="carto-positron",
                zoom=10,
                center={"lat": 48.8566, "lon": 2.3522},  # Coordenadas del centro de París
                opacity=0.5,
                labels={'porcentaje_superanfitriones': ''},
            )

            # Actualizar las trazas para mostrar el texto con dos decimales
            fig.update_traces(hovertemplate='%{location}<br>Porcentaje de Superanfitriones: %{z:.2f}%')

            # Mostrar el mapa en Streamlit
            st.plotly_chart(fig, use_container_width=True)

            #Comentario del mapa cloroplético
            st.write("""
                    ### **CONCLUSIÓN** 
                    El mapa cloroplético muestra el porcentaje de superanfitriones por barrio en París. 
                    Los barrios con un tono más oscuro tienen un mayor porcentaje de superanfitriones. 
                    Como se puede observar en el mapa, hay más cantidad de superanfitriones en los barrios
                    más céntricos de la ciudad.  
                    """)
        elif tipo_analisis_super_anfitriones == "Relación entre super anfitriones y precio":
            st.subheader("Relación entre super anfitriones y precio")
            # Introducción
            st.write(""" 
                    El objetivo de este análisis en encontrar si existe una relación entre el precio
                    de los alojamientos y el hecho de que el anfitrión sea superanfitrión. En primer lugar
                    veremos si existen diferencias significativas entre los dos grupos, los que son superanfitriones
                    y los que no lo son. En primer lugar se realizará una prueba de Anderson-Darling para 
                    comprobar si la variable precio sigue una distribución normal, posteriormente se realizará
                    una prueba de Mann-Whitney para comparar los precios entre los dos grupos, en el caso de que 
                    la variable no siga una distribución normal, o una prueba t para comparar los precios en el caso
                    de que la variable siga una distribución normal.
                    """)
            st.write("En primer lugar veremos como se distribuyen los precios de los que son super anfitriones y los que no")

            #BOXPLOT PARA COMPARAR LOS PRECIOS DE LOS SUPERANFITRIONES Y LOS QUE NO LO SON
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='host_is_superhost', y='price', data=df, showfliers=False, ax=ax, color = "#FF5A5F")  # Crear el boxplot
            ax.set_xlabel('¿Es superanfitrión?', fontsize=12)
            ax.set_ylabel('Precio ($)', fontsize=12)
            ax.set_xticklabels(['No', 'Sí'], fontsize=10)  # Cambiar etiquetas del eje x
            ax.grid(True, linestyle='--', alpha=0.7)

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)

            #COMENTARIO DEL BOXPLOT
            st.write("""
                     Se puede observar una distribución en ambos conjuntos similar, presentando una mediana superior
                     en el caso de los superanfitriones. A priori parece que no haya diferencias entre el 
                     precio de los alojamientos de los superanfitriones y los que no lo son, pero para confirmar
                        esto realizaremos una prueba estadística.
                     """)

            st.markdown("""
                        **<span style="color: red;">NOTA:</span>** 
                        La función para la prueba de *Anderson-Darling* devuelve un valor estadístico que se compara con su valor crítico para rechazar o no su hipótesis nula, 
                        no devuelve un *p-valor*. Por este motivo se puede ver *NaN* en la columna correspondiente al *p-valor*.
                        """, unsafe_allow_html=True)
            
            #PRUEBA DE NORMALIDAD DE LOS PRECIOS Y DE COMPARACIÓN DE MEDIAS
            # Mostrar los resultados de la prueba de hipótesis en formato de tabla


            # Creamos los dos conjuntos de datos para comparar los precios
            superhost_false = df[df['host_is_superhost'] == 'f']['price']
            superhost_true = df[df['host_is_superhost'] == 't']['price']
            # Llamamos a la función para realizar las pruebas estadísticas
            conclusiones = diagrama_flujo_test_hipótesis_Anderson_Darling(
                superhost_false,
                superhost_true,
                alpha=0.05
            )

            # Mostramos las conclusiones en Streamlit
            st.write("### Resultados de las Pruebas de Hipótesis")
            for conclusion in conclusiones:
                st.write(f"- {conclusion}")

            # Introducción
            st.write("""
                ### Análisis de la Normalidad con QQ Plots
                Para evaluar cómo se desvía la distribución de los datos de una distribución normal, 
                utilizamos gráficos QQ (Quantile-Quantile). Estos gráficos comparan los cuantiles de los datos 
                observados con los cuantiles esperados de una distribución normal.

                - **Conjunto 1:** Alojamientos cuyos anfitriones **no son superanfitriones**.
                - **Conjunto 2:** Alojamientos cuyos anfitriones **sí son superanfitriones**.

                Si los datos siguen una distribución normal, los puntos deberían alinearse aproximadamente sobre la línea diagonal.
            """)

            #QQPLOTS
            # Preparar los datos
            superhost_false = df[df['host_is_superhost'] == 'f']['price'].values
            superhost_true = df[df['host_is_superhost'] == 't']['price'].values

            # Crear figura y subplots
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))

            # Generar QQ plots para ambos conjuntos
            generar_qqplot(superhost_false, 'QQ plot para el conjunto 1 (No superanfitriones)', axes[0])
            generar_qqplot(superhost_true, 'QQ plot para el conjunto 2 (Superanfitriones)', axes[1])

            plt.tight_layout()

            # Mostrar el QQ plot en Streamlit
            st.pyplot(fig)
            
            st.write("Se puede observar que los puntos se distribuyen diferente de la diagonal, lo que indica que nuestros datos no siguen una distribución normal.")

            st.write("Ahora visualicemos el promedio de los precios según si el anfitrión es un superanfitrión o no en un gráfico de barras.")


            # Calculamos el precio promedio según si el anfitrión es superanfitrión o no
            avg_price_superhost = df.groupby('host_is_superhost')['price'].mean().reset_index()

            # Creamos un gráfico de barras con Seaborn
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='host_is_superhost', y='price', data=avg_price_superhost, palette=['#FF5A5F'], ax=ax3)
            ax3.set_xlabel('¿Es superanfitrión?', fontsize=14)
            ax3.set_ylabel('Precio Promedio ($)', fontsize=14)
            ax3.set_xticks([0, 1])
            ax3.set_xticklabels(['No', 'Sí'], fontsize=12)
            ax3.tick_params(axis='y', labelsize=12)
            ax3.grid(True, linestyle='--', alpha=0.7)

            # Añadimos etiquetas de datos
            for index, value in enumerate(avg_price_superhost['price']):
                ax3.text(index, value, f'${value:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig3)

            #CONCLUSIÓN 
            st.write("""
                     ### **Conclusión**
                     Los resultados obtenidos muestran que los apartamentos gestionados por superanfitriones tienen precios más altos en promedio que los de anfitriones regulares.""")
        elif tipo_analisis_super_anfitriones == "Relación entre super anfitriones y reseñas":
            st.subheader("Relación entre super anfitriones y reseñas")
            # Introducción
            st.write(""" 
                El objetivo de este análisis es determinar si existe una relación entre las puntuaciones 
                de los alojamientos y el hecho de que el anfitrión sea superanfitrión. En primer lugar, se evaluará 
                si existen diferencias significativas entre los dos grupos: aquellos cuyos anfitriones son superanfitriones 
                y aquellos que no lo son. Para ello, se realizará una prueba de Anderson-Darling para evaluar la normalidad 
                de los datos, seguida de pruebas estadísticas adecuadas para comparar las puntuaciones de reviews.
            """)
            
            # BOXPLOT PARA COMPARAR LAS PUNTUACIONES DE REVIEWS
            st.write("En primer lugar, veamos cómo se distribuyen las puntuaciones de reviews entre los dos grupos.")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='host_is_superhost', y='review_scores_rating', data=df_reviews, showfliers=False, ax=ax, color="#FF5A5F")
            ax.set_xlabel('¿Es superanfitrión?', fontsize=12)
            ax.set_ylabel('Puntuación de Reviews', fontsize=12)
            ax.set_xticklabels(['No', 'Sí'], fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)

            # COMENTARIO DEL BOXPLOT
            st.write("""
                A simple vista, los alojamientos gestionados por superanfitriones parecen tener puntuaciones promedio 
                más altas. Sin embargo, para confirmar estas diferencias realizaremos un análisis estadístico.
            """)
            st.markdown("""
                        **<span style="color: red;">NOTA:</span>** 
                        La función para la prueba de *Anderson-Darling* devuelve un valor estadístico que se compara con su valor crítico para rechazar o no su hipótesis nula, 
                        no devuelve un *p-valor*. Por este motivo se puede ver *NaN* en la columna correspondiente al *p-valor*.
                        """, unsafe_allow_html=True)
            # PRUEBA DE NORMALIDAD DE LAS PUNTUACIONES DE REVIEWS


            # Creamos los dos conjuntos de datos para comparar las puntuaciones de reviews
            superhost_false_reviews = df_reviews[df_reviews['host_is_superhost'] == 'f']['review_scores_rating']
            superhost_true_reviews = df_reviews[df_reviews['host_is_superhost'] == 't']['review_scores_rating']

            # Llamamos a la función para realizar las pruebas estadísticas
            conclusiones = diagrama_flujo_test_hipótesis_Anderson_Darling(
                superhost_false_reviews,
                superhost_true_reviews,
                alpha=0.05
            )

            # Mostramos las conclusiones en Streamlit
            st.write("### Resultados de las Pruebas de Hipótesis")
            for conclusion in conclusiones:
                st.write(f"- {conclusion}")

            # QQPLOTS
            st.write("""
                ### Análisis de la Normalidad con QQ Plots
                Para evaluar cómo se desvía la distribución de los datos de una distribución normal, 
                utilizamos gráficos QQ (Quantile-Quantile). Estos gráficos comparan los cuantiles de los datos 
                observados con los cuantiles esperados de una distribución normal.

                - **Conjunto 1:** Alojamientos cuyos anfitriones **no son superanfitriones**.
                - **Conjunto 2:** Alojamientos cuyos anfitriones **sí son superanfitriones**.

                Si los datos siguen una distribución normal, los puntos deberían alinearse aproximadamente sobre la línea diagonal.
            """)

            # Preparar los datos
            superhost_false_reviews_array = superhost_false_reviews.values
            superhost_true_reviews_array = superhost_true_reviews.values

            # Crear figura y subplots
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))

            # Generar QQ plots para ambos conjuntos
            generar_qqplot(superhost_false_reviews_array, 'QQ plot para el conjunto 1 (No superanfitriones)', axes[0])
            generar_qqplot(superhost_true_reviews_array, 'QQ plot para el conjunto 2 (Superanfitriones)', axes[1])

            plt.tight_layout()

            # Mostrar el QQ plot en Streamlit
            st.pyplot(fig)

            st.write("""
                Se puede observar que los puntos se distribuyen diferente de la diagonal, lo que indica que nuestros datos no siguen una distribución normal.
            """)

            # GRÁFICO DE BARRAS CON EL PROMEDIO DE PUNTUACIONES DE REVIEWS
            st.write("Ahora visualicemos el promedio de las puntuaciones de reviews según si el anfitrión es un superanfitrión o no en un gráfico de barras.")
            avg_reviews_superhost = df_reviews.groupby('host_is_superhost')['review_scores_rating'].mean().reset_index()

            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='host_is_superhost', y='review_scores_rating', data=avg_reviews_superhost, palette=['#FF5A5F'], ax=ax3)
            ax3.set_xlabel('¿Es superanfitrión?', fontsize=14)
            ax3.set_ylabel('Puntuación Promedio de Reviews', fontsize=14)
            ax3.set_xticks([0, 1])
            ax3.set_xticklabels(['No', 'Sí'], fontsize=12)
            ax3.tick_params(axis='y', labelsize=12)
            ax3.grid(True, linestyle='--', alpha=0.7)

            for index, value in enumerate(avg_reviews_superhost['review_scores_rating']):
                ax3.text(index, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

            st.pyplot(fig3)

            # CONCLUSIÓN
            st.write("""
                ### **Conclusión**
                Los resultados obtenidos muestran que los alojamientos gestionados por superanfitriones tienen puntuaciones de reviews más altas en promedio 
                que los alojamientos gestionados por anfitriones regulares. Esto refuerza la percepción de mayor calidad y satisfacción asociada con los superanfitriones.
            """)
elif selected_option == "Panel de control | PowerBI":
    st.header("Panel de control | PowerBI")
    st.write("""
        En esta sección se presenta un panel de control interactivo creado con Power BI.
        En este panel interactivo el usuario podrá visualizar los apartamentos que mejor se adapten a sus necesidades, por ejmplo,
        según el rango de precio, el tipo de habitación, el barrio, etc. 

    """)
    
    # Incrustar el informe en un iframe
    st.components.v1.iframe(src=ruta_powerbi, width=1200, height=800)










        

                     
                     
                     



        


