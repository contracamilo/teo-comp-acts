# Laboratorios en AWS academy

**Actividad 4 – Laboratorio en AWS**
**Estudiante:** Camilo Rivera Quintero
**Docente:** Rodrigo Fernandez Aranda
**Curso:** Teoría de la Computación 2 - Virtual

## Sobre esta actividad

Este repositorio reúne la evidencia de los tres laboratorios guiados de Amazon SageMaker que desarrollé para la Actividad 4.

- **Laboratorio 3.1** – Amazon SageMaker: creación e importación de datos.
- **Laboratorio 3.2** – Amazon SageMaker: exploración de datos.
- **Laboratorio 3.3** – Amazon SageMaker: codificación de datos categóricos.

---

## 🔑 Evidencia de acceso a la plataforma de laboratorio

[Acceso a la plataforma](evidencias/lab3-1/acceso-plataforma.png)

---

## 🧪 Laboratorio 3.1 – Creación e importación de datos

### ¿Qué hice y por qué?

En este laboratorio el objetivo era dejar lista una instancia de notebook de SageMaker y entender cómo se mueve uno dentro de JupyterLab antes de meterme con datos de verdad. Hice lo siguiente:

1. **Creé una instancia de cuaderno** llamada `MyNotebook`, de tipo `ml.m4.xlarge`, usando el identificador de plataforma `notebook-al2-v3` (Amazon Linux 2). En la configuración adicional elegí la configuración de ciclo de vida que incluye `ml-pipeline`, para que los notebooks de ejemplo se cargaran automáticamente. Lo hice así porque es la configuración que pide la guía del laboratorio y porque me aseguraba tener disponibles los notebooks de muestra sin tener que subirlos a mano.
2. **Abrí JupyterLab** desde la instancia una vez pasó de estado *Pending* a *InService*, y recorrí la interfaz: la barra de menú (File, Edit, Run, Kernel, Git...), el explorador de archivos y el notebook de ejemplo `PythonCheatSheet.ipynb`. Esto me sirvió para ubicar dónde está cada cosa antes de empezar a escribir código propio.
3. **Abrí el notebook de muestra** `linear_learner_mnist.ipynb` y saqué una copia con "Create a Copy", solo para explorarlo (no lo ejecuté porque requiere un bucket de S3 que no hace parte de este laboratorio).
4. **Creé un notebook nuevo** con el kernel `conda_python310`, le puse un título en una celda de Markdown (`# Importing the data`) y agregué el código para descargar el dataset de la columna vertebral (Vertebral Column Dataset, del repositorio de UC Irvine) directamente desde su URL, extraerlo con `zipfile` y cargarlo en un `DataFrame` de Pandas usando `scipy.io.arff`. Lo hice en dos celdas separadas (importaciones y descarga/extracción) para poder depurar cada parte por separado si algo fallaba.
5. Cargué los datos del archivo de dos clases (`column_2C_weka.arff`) en un `DataFrame` y revisé las primeras filas con `df.head()` para confirmar que la carga había funcionado.

### Pasos y capturas

> 🔲 **PENDIENTE:** agrega una captura por cada uno de estos momentos (puedes usar la carpeta `evidencias/lab-3.1/`):
>
> - [Configuración de la instancia de notebook](evidencias/lab3-1/mynotebook.png)
> - [Instancia en estado `InService`](evidencias/lab3-1/mynotebook.png)
> - [PythonCheatSheet.ipynb](evidencias/lab3-1/PythonCheatSheet.ipynb)
> - [Celdas de código con la descarga/extracción del dataset](evidencias/lab3-1/extraction-code.png)

### Resultado final de la tarea

> - [Datos de la columna vertebral](evidencias/lab3-1/table.png)
