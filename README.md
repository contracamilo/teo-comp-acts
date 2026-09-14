# Laboratorios en AWS academy

- **Actividad 4 – Laboratorio en AWS**
- **Estudiante:** Camilo Rivera Quintero
- **Docente:** Rodrigo Fernandez Aranda
- **Curso:** Teoría de la Computación 2 - Virtual

## Sobre esta actividad

Este repositorio reúne la evidencia de los tres laboratorios guiados de Amazon SageMaker que desarrollé para la Actividad 4.

- **Laboratorio 3.1** – Amazon SageMaker: creación e importación de datos.
- **Laboratorio 3.2** – Amazon SageMaker: exploración de datos.
- **Laboratorio 3.3** – Amazon SageMaker: codificación de datos categóricos.

---

## 🔑 Evidencia de acceso a la plataforma de laboratorio

![Acceso a la plataforma](evidencias/lab3-1/acceso-plataforma.png)

---

## 🧪 Laboratorio 3.1 – Creación e importación de datos

### ¿Qué hice y por qué?

En este laboratorio el objetivo era dejar lista una instancia de notebook de SageMaker y entender cómo se mueve uno dentro de JupyterLab antes de meterme con datos de verdad. Hice lo siguiente:

1. **Creé una instancia de cuaderno** llamada `MyNotebook`, de tipo `ml.m5.xlarge`, con el identificador de plataforma `Amazon Linux 2023, Jupyter Lab 4`. Dejé el rol de IAM que me sugirió la consola (con `AmazonSageMakerFullAccess`) y el resto de las opciones por defecto, porque para este laboratorio no necesitaba nada adicional.
2. **Abrí JupyterLab** desde la instancia una vez pasó de estado *Pending* a *InService*, y me encontré con los notebooks de ejemplo ya cargados: `PythonCheatSheet.ipynb` y `linear_learner_mnist.ipynb`. Los recorrí para ubicarme en la interfaz (menú de archivo, kernels, panel de Git) antes de escribir código propio.
3. **Creé mi propio notebook** (`lab31.ipynb`) con el kernel `conda_python3`, le puse un título en una celda de Markdown (`# Importing the data`) y agregué el código para descargar el dataset de la columna vertebral (Vertebral Column Dataset, del repositorio de UC Irvine) directamente desde su URL, extraerlo con `zipfile` y cargarlo en un `DataFrame` de Pandas usando `scipy.io.arff`.
4. Al ejecutar la celda del import me salió `ModuleNotFoundError: No module named 'scipy'`, porque el kernel no lo trae instalado por defecto. Lo resolví agregando `!pip install scipy` en una celda antes del import y volviendo a correr todo. Lo dejo documentado porque me pareció importante mostrar que el error hace parte del proceso, no solo el resultado final.
5. Cargué los datos del archivo de dos clases (`column_2C_weka.arff`) en un `DataFrame` y revisé las primeras filas con `df.head()` para confirmar que la carga había funcionado.

### Pasos y capturas

- **Configuración de la instancia de notebook** (`MyNotebook`, `ml.m5.xlarge`, Amazon Linux 2023 / Jupyter Lab 4):
  ![Configuración de la instancia](evidencias/lab3-1/config.png)
- **Instancia en estado `InService`**, lista para abrir JupyterLab:
  ![Instancia InService](evidencias/lab3-1/mynotebook.png)
- **Notebook `lab31.ipynb`** con el fix de `scipy` aplicado y el código de descarga/extracción ya ejecutado:
  ![Código de descarga y extracción](evidencias/lab3-1/extraction-code.png)

### Resultado final de la tarea

![Datos de la columna vertebral cargados en el DataFrame](evidencias/lab3-1/table.png)

### Commits de esta tarea

- `chore: evidence from lab 3-1` ([`40e19e8`](../../commit/40e19e8))

---

## 🧪 Laboratorio 3.2 – Exploración de datos

### ¿Qué hice y por qué?

Usé el mismo dataset de la columna vertebral, el escenario es que trabajamos para un proveedor de salud que quiere mejorar la detección de anomalías en pacientes ortopédicos a partir de seis características biomecánicas (incidencia pélvica, inclinación pélvica, ángulo de lordosis lumbar, inclinación del sacro, radio pélvico y grado de espondilolistesis). Antes de pensar en entrenar cualquier modelo, segn los videos de entrenamiento debemos hacer un análisis exploratorio con estadística descriptiva, para esto usamos en Kernel de condas y las libs de python, `pandas` para las estadísticas y `matplotlib`/`seaborn` para las visualizaciones.

1. **Inicié el laboratorio** usando el notebook predefinido `3_2-machinelearning.ipynb`, que ya trae el código de configuración listo; solo tuve que agregar un par de líneas para instalar las librerías que hacían falta.
2. **Revisé la forma y las columnas** del `DataFrame` con `df.shape` y `df.columns`, y confirmé los tipos de dato con `df.dtypes`: las seis características son numéricas y la columna `class` es la variable objetivo. Este paso es básico pero necesario según las guias.
3. **Saqué estadísticas descriptivas** primero de una sola columna (`pelvic_incidence.describe()`) y luego de todo el dataset (`df.describe()`), para ver rangos, medias y detectar de una vez posibles valores atípicos.
4. **Grafiqué las distribuciones** con gráficos de densidad (KDE) de las seis características en una cuadrícula 4x2. Ahí noté que casi todas están sesgadas a la derecha (colas largas hacia valores altos), excepto `pelvic_radius`, que es la única con forma de campana simétrica. `degree_spondylolisthesis` fue la más extrema: un pico angosto pegado a 0 con una cola que se estira hasta 400-600, señal clara de valores atípicos.
5. **Profundicé en `degree_spondylolisthesis`** con gráfico de densidad, histograma y boxplot, confirmando ese grupo de valores atípicos alrededor de 400.
6. **Analicé la variable objetivo (`class`)**: con `value_counts()` vi un desbalance moderado (~1/3 *Normal*, ~2/3 *Anormal*). Convertí la clase a numérica con un mapeador (`Abnormal → 1`, `Normal → 0`), porque los modelos de ML no trabajan con texto.
7. **Comparé cada característica contra la clase**, primero con un scatter de `degree_spondylolisthesis` y después completando el reto del notebook: repetí el mismo scatter para las otras cinco variables y calculé `df.groupby('class').mean()` para comparar promedios entre grupos. El hallazgo fue que: `degree_spondylolisthesis` es 17 veces más alta en promedio en pacientes anormales (2.19 vs. 37.78) — por lejos la variable más discriminante. Le siguen `pelvic_tilt`, `lumbar_lordosis_angle` y `pelvic_incidence`, con diferencias de 25% a 54% entre clases. `pelvic_radius` resultó la menos útil: es la única característica más baja en pacientes anormales, y con la diferencia relativa más pequeña (~7%).
8. **Calculé la correlación entre variables** con `df.corr()`, ordenando qué tan correlacionada está cada característica con la clase, y la visualicé con una matriz de dispersión (`scatter_matrix`) y un mapa de calor con `seaborn`. Los números confirmaron el orden que ya había visto con las medias: `degree_spondylolisthesis` tiene la correlación más alta con `class` (**0.44**), seguida de `pelvic_incidence` (0.35), `pelvic_tilt` (0.33), `lumbar_lordosis_angle` (0.31) y `sacral_slope` (0.21); `pelvic_radius` es la única con correlación **negativa** (-0.31), coherente con que sus valores bajan en pacientes anormales en vez de subir. El heatmap además dejó ver algo que no había notado antes: varias características predictoras están bastante correlacionadas *entre ellas* (por ejemplo `pelvic_incidence` con `sacral_slope`, 0.81, y con `lumbar_lordosis_angle`, 0.72) — es decir, parte de esa información está duplicada entre columnas.

**Conclusión de la tarea:** el análisis confirma que un futuro modelo se apoyaría fuerte en `degree_spondylolisthesis` como señal principal lo cual tiene sentido clínicamente, ya que la espondilolistesis es una de las dos condiciones que conforman la categoría "Anormal" del dataset original, complementado por `pelvic_incidence` y `pelvic_tilt`, mientras que `pelvic_radius` aportaría poco por sí sola (y en dirección contraria a las demás). También quedó claro que antes de modelar habría que decidir qué hacer con el desbalance de clases, los valores atípicos de `degree_spondylolisthesis`, y la multicolinealidad entre `pelvic_incidence`, `sacral_slope` y `lumbar_lordosis_angle` (usar las tres juntas en un modelo podría ser redundante).

### Pasos y capturas

> - [Notebook `3_2-machinelearning.ipynb`](evidencias/lab3-2/lab3-2.png)
> - [Salida de `df.describe()`](evidencias/lab3-2/describe.png)
> - [Los gráficos de densidad (KDE) de los features](evidencias/lab3-2/KDE.png) .
> - [El histograma o boxplot de `degree_spondylolisthesis` con valores atípicos.](evidencias/lab3-2/boxplot.png)
> - [La tabla de correlación ordenada y Matriz de dispersión (`scatter_matrix`](evidencias/lab3-2/matrix.png)
> - [Mapa de calor de correlación (`seaborn.heatmap`)](evidencias/lab3-2/heat_map.png)

### Resultado final de la tarea

>
> [Resultado Lab 3.2](evidencias/lab3-2/success.pngg)

### Commits de esta tarea

> - `chore: evidence from lab 3-2` ()
