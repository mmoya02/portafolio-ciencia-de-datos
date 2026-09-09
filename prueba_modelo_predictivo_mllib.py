# =============================================================================
# PRUEBA: MODELO PREDICTIVO EN APACHE SPARK MLLIB
# =============================================================================
#
# Objetivo:
# Preparar datos de ventas, crear una etiqueta binaria de riesgo,
# entrenar un modelo de clasificación y evaluar su desempeño.
#
# Regla para label:
# 1 = transacción riesgosa si:
#     - Monto_Total > 7000
#     - O la venta ocurre entre las 00:00 y 05:59
#
# 0 = transacción normal
# =============================================================================


# =============================================================================
# 1. IMPORTAR LIBRERÍAS
# =============================================================================

import os
import sys
from pathlib import Path

# Indicar a Spark qué Python utilizar
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    hour,
    when,
    expr
)

from pyspark.ml import Pipeline

from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler
)

from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)


# =============================================================================
# 2. INICIAR SPARK
# =============================================================================

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("PruebaMLlibVentas") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# =============================================================================
# 3. CARGAR DATASET
# =============================================================================

carpeta = Path(__file__).resolve().parent

ruta_csv = carpeta / "ventas_simuladas.csv"


print("\nARCHIVO UTILIZADO:")
print(ruta_csv)


df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(str(ruta_csv))


print("\nDATOS ORIGINALES:")

df.show(
    5,
    truncate=False
)


print("\nESQUEMA ORIGINAL:")

df.printSchema()


# =============================================================================
# 4. PREPARAR Y LIMPIAR LOS DATOS
# =============================================================================

# Convertir las columnas numéricas
df = df \
    .withColumn(
        "Cantidad",
        col("Cantidad").cast("double")
    ) \
    .withColumn(
        "Precio_Unitario",
        col("Precio_Unitario").cast("double")
    ) \
    .withColumn(
        "Monto_Total",
        col("Monto_Total").cast("double")
    )


# Convertir Fecha_Hora.
#
# try_to_timestamp evita que Spark se detenga
# cuando encuentra valores incorrectos como "invalid-date".
#
# Si una fecha es inválida, devuelve NULL.

df = df.withColumn(
    "Fecha_Hora_TS",
    expr(
        "try_to_timestamp(Fecha_Hora, 'yyyy-MM-dd H:mm:ss')"
    )
)


# Eliminar registros con datos faltantes o fechas inválidas

df_limpio = df.dropna(
    subset=[
        "Sucursal",
        "Producto",
        "Cantidad",
        "Precio_Unitario",
        "Monto_Total",
        "Fecha_Hora_TS"
    ]
)


# Extraer la hora de cada transacción

df_limpio = df_limpio.withColumn(
    "Hora",
    hour(
        col("Fecha_Hora_TS")
    )
)


print("\nREGISTROS VÁLIDOS:")

print(
    df_limpio.count()
)


print("\nDATOS LIMPIOS:")

df_limpio.show(
    5,
    truncate=False
)


# =============================================================================
# 5. CREAR COLUMNA LABEL
# =============================================================================

# Transacción riesgosa:
#
# 1 = Monto_Total > 7000
#     O venta realizada antes de las 06:00
#
# 0 = Normal


df_limpio = df_limpio.withColumn(
    "label",

    when(
        (col("Monto_Total") > 7000)
        |
        (col("Hora") < 6),

        1.0

    ).otherwise(
        0.0
    )
)


print("\nDISTRIBUCIÓN DE LA VARIABLE LABEL:")

df_limpio.groupBy(
    "label"
).count().orderBy(
    "label"
).show()


# =============================================================================
# 6. DIVIDIR DATOS EN ENTRENAMIENTO Y PRUEBA
# =============================================================================

train_raw, test_raw = df_limpio.randomSplit(
    [0.8, 0.2],
    seed=42
)


print(
    "\nRegistros de entrenamiento:",
    train_raw.count()
)

print(
    "Registros de prueba:",
    test_raw.count()
)


# =============================================================================
# 7. STRINGINDEXER
# =============================================================================

# Convertir Sucursal a valor numérico

index_sucursal = StringIndexer(
    inputCol="Sucursal",
    outputCol="Sucursal_index",
    handleInvalid="keep"
)


# Convertir Producto a valor numérico

index_producto = StringIndexer(
    inputCol="Producto",
    outputCol="Producto_index",
    handleInvalid="keep"
)


# =============================================================================
# 8. VECTORASSEMBLER
# =============================================================================

# Combinar variables predictoras en una sola columna "features"

assembler = VectorAssembler(

    inputCols=[

        "Cantidad",

        "Precio_Unitario",

        "Monto_Total",

        "Hora",

        "Sucursal_index",

        "Producto_index"

    ],

    outputCol="features"
)


# =============================================================================
# 9. PIPELINE DE PREPARACIÓN
# =============================================================================

pipeline_prep = Pipeline(

    stages=[

        index_sucursal,

        index_producto,

        assembler

    ]
)


# Ajustar preparación usando solamente datos de entrenamiento

prep_model = pipeline_prep.fit(
    train_raw
)


# Transformar entrenamiento y prueba

train = prep_model.transform(
    train_raw
)

test = prep_model.transform(
    test_raw
)


# =============================================================================
# 10. MOSTRAR FEATURES + LABEL
# =============================================================================

print(
    "\nDATAFRAME FINAL: FEATURES + LABEL"
)

train.select(
    "features",
    "label"
).show(
    10,
    truncate=False
)


print(
    "\nESQUEMA FINAL:"
)

train.select(
    "features",
    "label"
).printSchema()


# =============================================================================
# 11. ENTRENAR RANDOM FOREST
# =============================================================================

modelo = RandomForestClassifier(

    labelCol="label",

    featuresCol="features",

    numTrees=10,

    seed=42
)


modelo_entrenado = modelo.fit(
    train
)


# =============================================================================
# 12. GENERAR PREDICCIONES
# =============================================================================

predicciones = modelo_entrenado.transform(
    test
)


print(
    "\nPREDICCIONES:"
)

predicciones.select(

    "label",

    "prediction",

    "probability"

).show(
    20,
    truncate=False
)


# =============================================================================
# 13. EVALUAR ACCURACY
# =============================================================================

evaluador_accuracy = MulticlassClassificationEvaluator(

    labelCol="label",

    predictionCol="prediction",

    metricName="accuracy"

)


accuracy = evaluador_accuracy.evaluate(
    predicciones
)


# =============================================================================
# 14. EVALUAR AREA UNDER ROC
# =============================================================================

evaluador_auc = BinaryClassificationEvaluator(

    labelCol="label",

    rawPredictionCol="rawPrediction",

    metricName="areaUnderROC"

)


auc = evaluador_auc.evaluate(
    predicciones
)


# =============================================================================
# 15. MOSTRAR RESULTADOS
# =============================================================================

print("\n==============================")
print("MÉTRICAS DEL MODELO")
print("==============================")


print(
    f"Accuracy: {accuracy:.4f}"
)


print(
    f"Area Under ROC (AUC): {auc:.4f}"
)


# =============================================================================
# 16. CERRAR SPARK
# =============================================================================

spark.stop()


print(
    "\nPROCESAMIENTO FINALIZADO CORRECTAMENTE."
)