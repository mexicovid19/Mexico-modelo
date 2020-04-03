# Modelo para las proyecciones del COVID-19 en México

## ESP
En este repositorio se encuentra el modelo utilizado para [la página para el monitoreo del Covid-19 en México](https://mexicovid19.github.io/Mexico/) y su respectivo [repositorio](https://github.com/mexicovid19/Mexico). Este modelo es una simplificación del propuesto por [Alex Arenas, Jesús Gómez-Gardeñes y sus colaboradores](https://covid-19-risk.github.io/map/), cuyo reporte técnico puedes ver [aquí](https://covid-19-risk.github.io/map/model.pdf). Escogimos seguir esta propuesta debido a la precisión con la que describe el caso español.

Para correr el modelo, ejecuta el script `run_SEAIHRD_model.py` que se encuentra en la carpeta `./src`. Puedes cambiar los parámetros del modelo directamente desde el script. Los detalles técnicos de nuestro modelo, así como la descripción de los parámetros utilizados los puedes consultar [aquí](https://github.com/blas-ko/Mexico-modelo/blob/master/descripcion_modelo.ipynb).

>DESCARGO DE RESPONSABILIDAD: El carácter de nuestras predicciones es meramente ilustrativo, nuestro equipo NO está formado por epidemiólogos.

Contacto:
mexicovid19contacto@gmail.com

<hr>

## ENG
This repository contains the model used in the [MexiCovid-19 website](https://mexicovid19.github.io/Mexico/). Our model is a simplification of the one from [Alex Arenas, Jesús Gómez-Gardeñes, et al.](https://covid-19-risk.github.io/map/), which technical report can be found [here](https://covid-19-risk.github.io/map/model.pdf). We chose their model as our basis given the success they had on their predictions for Spain.

To run the model, execute the script `run_SEAIHRD_model.py` in the `./src` directory. You can change the parameters of the model within the script. The technical details of our model, as well as the description of its parameters can be seen [here](https://github.com/blas-ko/Mexico-modelo/blob/master/descripcion_modelo.ipynb) (still in spanish only, sorry.). In the figure below we show the results of our model.

>DISCLAIMER: Our predictions are merely ilustrative, our team is NOT formed by epidemiologists.

Contact:
mexicovid19contacto@gmail.com


![proyecciones](https://github.com/blas-ko/Mexico-modelo/blob/master/media/covid19_mex_proyecciones.png "proyecciones")
Fig. 1: Proyección de los casos totales confirmados de COVID-19 en México
