# Análisis de Predicción de Enfermedades Cardiovasculares

## Descripción del Proyecto

Este proyecto tiene como objetivo predecir la presencia de enfermedades cardiovasculares en pacientes utilizando diferentes algoritmos de Machine Learning: **Regresión Logística, Árboles de Decisión y Random Forest.**

Para ello, se utiliza un dataset público que incluye información médica básica de pacientes, como presión arterial, niveles de colesterol y frecuencia cardíaca máxima, entre otras. 


## Información del Dataset

**Fuente:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets)  
**Tamaño:** 1000 registros con 13 variables predictoras.

### Variables del Dataset

| **Variable**         | **Descripción**                                                | **Valores**                                                                                   |
|-----------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `patientid`          | Identificador único de paciente                                | Número entero                                                                                 |
| `age`                | Edad en años                                                  | 29 - 77 años                                                                                 |
| `gender`             | Sexo del paciente                                             | 0: Mujer, 1: Hombre                                                                           |
| `chestpain`          | Tipo de dolor de pecho                                        | 0: Angina típica, 1: Angina atípica, 2: Dolor no anginoso, 3: Asintomático                    |
| `restingBP`          | Presión arterial en reposo (mm Hg)                            | 94 - 200                                                                                      |
| `serumcholestrol`    | Niveles de colesterol sérico (mg/dl)                          | 126 - 564                                                                                    |
| `fastingbloodsugar`  | Azúcar en sangre en ayunas (> 120 mg/dl)                      | 0: No, 1: Sí                                                                                 |
| `restingelectro`     | Resultados electrocardiográficos en reposo                    | 0: Normal, 1: Anomalía ST-T, 2: Hipertrofia ventricular izquierda probable                   |
| `maxheartrate`       | Frecuencia cardíaca máxima alcanzada                          | 71 - 202                                                                                     |
| `exerciseangia`      | Angina inducida por ejercicio                                 | 0: No, 1: Sí                                                                                 |
| `oldpeak`            | Depresión del segmento ST                                    | 0 - 6.2                                                                                      |
| `slope`              | Pendiente del segmento ST                                     | 1: Ascendente, 2: Plana, 3: Descendente                                                      |
| `noofmajorvessels`   | Número de vasos principales coloreados por fluoroscopia       | 0, 1, 2, 3                                                                                   |
| `target`             | Variable objetivo (presencia de enfermedad)                  | 0: Ausencia, 1: Presencia                                                                     |


## Objetivo del análisis

Construir un modelo predictivo capaz de determinar si un paciente tiene o no una enfermedad cardiovascular (`target`) basándonos en sus características clínicas.


## Resultados principales

- **Random Forest** es el modelo más robusto y ofrece el mejor rendimiento general. Tiene la precisión más alta, el menor número de errores en la matriz de confusión, y utiliza un conjunto equilibrado de predictores. Además, tiene menor riesgo de sobreajuste, ya que la diferencia entre el rendimiento en entrenamiento y prueba es pequeña.

- **Regresión Logística** también es un buen modelo, pero tiene una precisión ligeramente inferior y podría no capturar relaciones no lineales tan bien como el Random Forest.

- **Árboles de Decisión** tiene un buen rendimiento, pero la eliminación de muchos predictores puede limitar su capacidad de generalización.


## Conclusiones

Basándonos en los resultados principales, para abordar este problema de clasificación, el modelo de **Random Forest** es la opción más adecuada, proporcionando el mejor equilibrio entre precisión, generalización y robustez. 
