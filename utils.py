

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)


def mae_porcentual (y_true, y_pred): 
    """
    Vamos a desglosar la función anterior (mean_absolute_percentage_error) para ver lo que hace paso a paso (el resultado es el mismo):
        1. Nos aseguramos que y_true e y_pred son arrrays de numpy
        2. Obtenemos el valor absoluto (ignorando el signo negativo)
        3. Ahora obtendremos el porcentaje en el que se ha equivocado para cada valor de la columna
        4. Finalmente obtenemos la media de todos los errores 
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diferencia_prediccion = np.abs(y_true - y_pred)
    porcentaje = diferencia_prediccion / y_true * 100
    return np.mean(porcentaje)


def update_resultados(model:LinearRegression, 
                      X : DataFrame, y:DataFrame, 
                      index_name: str, 
                      resultados = DataFrame({}) 
                    ) -> DataFrame:
    """
    Devuelve un DataFrame con la mae, la mae_porcentual, el mse, el rmse, y el coeficiente de determinación (R2).
    Si no se le pasa ningún DataFrame de 'resultados' como parámetro, utiliza por defecto uno vacío.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test) 

    mae = mean_absolute_error(y_test, y_pred)
    mae_porc = mae_porcentual (y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = model.score(X_train, y_train)

    nuevo_registro = pd.DataFrame({
        'MAE': [mae],
        'MAE Porcentual': [mae_porc],
        'MSE': [mse],
        'RMSE': [rmse],
        'R2': [r2]
        }, index=[f'{index_name}'])

    if nuevo_registro.index[0] not in resultados.index:
        resultados = pd.concat([resultados, nuevo_registro])
    else:
        print(f"El modelo '{nuevo_registro.index[0]}' ya existe en los resultados.")
    return resultados

def calcular_residuos(y : DataFrame, y_predic: DataFrame, col: str) -> None:
    """
    Dibuja en una gráfica los residuos (eje y) frente a los valores predichos. 
    Intentamos que la distribución sea lo más uniforme posible para evitar patrones o sesgos en los datos
    (al margen de que se aleje más o menos al 0).
    """
    residuos = y - y_predic
    #x[:, 0] -> selecciona todas las filas de la primera columna
    sns.scatterplot(x=y_predic[:,0], y=residuos[f'{col}'])
    plt.ylabel('Residuos')
    plt.axhline(y=0, color='r', linestyle='--')


def drop_outliers(df:DataFrame) -> DataFrame:
    df_copy = df.copy()

    for column in df_copy.columns:
        if df_copy[column].dtype == 'object':
            continue

        # borrar outliers en numéricos
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        filter = (df_copy[column] >= lower_limit) & (df_copy[column] <= upper_limit)
        df_copy = df_copy[filter]

    return df_copy