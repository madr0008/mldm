# Scripts de ejecución
En este directorio se recogen los scripts, en R, necesarios para generar los ficheros de ejecución de la experimentación con MLDM. Todos ellos incluyen rutas propias de la máquina en la que se llevó a cabo la experimentación, por lo que es necesaria la personalización del código para adaptarse a otros equipos.

## Requisitos

Se presupone la existencia de un directorio `/datasets` en el que se se encuentran almacenados los distintos MLD en formato ARFF y XML, para las distintas particiones, con la siguiente estructura:

- `datasets` -- Directorio base
	- `cal500` -- script principal para la llamada a los procesos de entrenamiento y muestreo
		- `1` -- partición 1
			- `cal500.arff` -- Fichero ARFF
			- `cal500.xml` -- Fichero XML
			
Y así con el resto de particiones y MLD. Todos estos ficheros pueden descargarse desde [Cometa](https://cometa.ujaen.es/).

## Scripts de ejecución
Dentro del subdirectorio `Ejecucion` se encuentran aquellos scripts para generar ficheros necesarios para la ejecución.

### MLDM
Los ficheros para la ejecución del algoritmo MLDM son:
- `generar_configs.R` -- generación de ficheros config en un directorio para las diferentes variantes de MLDM  
- `generar_script_ejecucion_mldm.R` -- generación de un script bash de ejecución de las distintas variantes
- `generar_metricas_mldm.R` -- generación de ficheros csv con métricas descriptivas de los MLDM resultantes

### Clasificadores
Los ficheros para la ejecución de clasificadores multietiqueta tras aplicar MLDM son:
- `generar_script_ejecucion_clasificadores.R` -- generación de un script bash de ejecución de los distintos clasificadores con los MLD generados
- `generar_metricas_clasificadores.R` -- generación de ficheros csv con métricas de evaluación de los clasificadores ejecutados

## Scripts para generación de gráficas

Dentro del subdirectorio `Graficas` se encuentran aquellos scripts para generar gráficas usando R, que son los siguientes:
- `generar_graficas_metricas.R` -- generación de gráficas de MeanIR y SCUMBLE tras la ejecución de los algoritmos
- `generar_graficas_evaluacion.R` -- generación de gráficas de métricas de evaluación tras la ejecución de los clasificadores
