# MLDM: Multilabel Diffusion Models
Repositorio del TFG "Diseño de algoritmos de remuestreo multi-etiqueta con modelos generativos profundos".

Se trata de la implementación de un modelo de difusión para oversampling de datos multietiqueta.
Esta implementación es una adaptación del modelo [TabDDPM](https://github.com/rotot0/tab-ddpm).

## Ejecución del modelo
1. Instalar [conda](https://docs.conda.io/en/latest/miniconda.html) para gestionar el entorno virtual.
2. Ejecutar los siguientes comandos, para crear el entorno e instalar las dependencias necesarias.
    ```bash
    export REPO_DIR=/path/to/the/code
    cd $REPO_DIR

    conda create -n mldm python=3.9.7
    conda activate mldm

    pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt

    conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
    conda env config vars set PROJECT_DIR=${REPO_DIR}

    conda deactivate
    conda activate mldm
    ```

### Conjuntos de datos

Los conjuntos de datos multietiqueta (MLD) soportados por el algoritmo son aquellos en formato ARFF, acompañados de un fichero XML que especifique el nombre de las etiquetas. Este formato es el mismo que usa la biblioteca [MULAN](https://mulan.sourceforge.net/).

En el repositorio [Cometa](https://cometa.ujaen.es/) se recopila una gran variedad de MLD, completos o previamente particionados. 

### Llamada al algoritmo

Para ejecutar el algoritmo sobre un conjunto de datos, tan solo hay que ejecutar los siguientes comandos.

``` bash
conda activate mldm
cd $PROJECT_DIR
python scripts/pipeline.py --config_file=config.toml
```

Los parámetros para la ejecución del modelo se encuentran un fichero de configuración con formato `toml`. La estructura y parámetros recogidos en este fichero se explican [aquí](CONFIG_DESCRIPTION.md).

## Estructura de ficheros
`mldm/` -- Directorio con la implementación del método propuesto

- `mldm/gaussian_multinomial_diffusion.py` -- modelo de difusión  
- `mldm/modules.py` -- otros módulos que componen el modelo principal
- `mldm/utils.py` -- funciones matemáticas para el modelo

`scripts/` -- Directorio con los scripts del proyecto

- `scripts/pipeline.py` -- script principal para la llamada a los procesos de entrenamiento y muestreo
- `scripts/sample.py` -- script para el proceso de muestreo
- `scripts/train.py` -- script para el proceso de entrenamiento
- `scripts/utils_train.py` -- script con funciones auxiliares para el entrenamiento

`lib/` -- Directorio con bibliotecas locales del proyecto

- `lib/data.py` -- definición de clases y métodos para trabajo con MLD
- `lib/util.py` -- script con funciones auxiliares para el entrenamiento

## Referencias

Este proyecto ha sido construido a partir de el trabajo previo reflejado en los siguientes artículos:

- Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2022). TabDDPM: Modelling Tabular Data with Diffusion Models. arXiv preprint arXiv:2209.15421.


- Kim, J., Lee, C., Shin, Y., Park, S., Kim, M., Park, N., & Cho, J. (2022, August). Sos: Score-based oversampling for tabular data. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 762-772).