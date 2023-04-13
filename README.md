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

    conda create -n tddpm python=3.9.7
    conda activate tddpm

    pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt

    # if the following commands do not succeed, update conda
    conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
    conda env config vars set PROJECT_DIR=${REPO_DIR}

    conda deactivate
    conda activate tddpm
    ```

#### Conjuntos de datos

Los conjuntos de datos multietiqueta (MLD) soportados por el algoritmo son aquellos en formato ...

En el repositorio [Cometa](https://cometa.ujaen.es/) se recopila una gran variedad de MLD, completos o previamente particionados. 

### Llamada al algoritmo

Para ejecutar el algoritmo sobre un conjunto de datos, tan solo hay que ejecutar los siguientes comandos.

``` bash
conda activate tddpm
cd $PROJECT_DIR
wget "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0" -O data.tar
tar -xvf data.tar
```

## Estructura de ficheros
`mldm/` -- Directorio con la implementación del método propuesto

- `mldm/gaussian_multinomial_diffusion.py` -- modelo de difusión  
- `modules.py` -- otros módulos que componen el modelo principal
- `utils.py` -- funciones matemáticas para el modelo

`scripts/` -- Directorio con los scripts principales

- `scripts/pipeline.py` are used to train, sample and eval TabDDPM using a given config  
- `scripts/tune_ddpm.py` -- tune hyperparameters of TabDDPM
- `scripts/eval_[catboost|mlp|simple].py` -- evaluate synthetic data using a tuned evaluation model or simple models
- `scripts/eval_seeds.py` -- eval using multiple sampling and multuple eval seeds
- `scripts/eval_seeds_simple.py` --  eval using multiple sampling and multuple eval seeds (for simple models)
- `scripts/tune_evaluation_model.py` -- tune hyperparameters of eval model (CatBoost or MLP)
- `scripts/resample_privacy.py` -- privacy calculation  

To understand the structure of `config.toml` file, read `CONFIG_DESCRIPTION.md`.

## Referencias

Este proyecto ha sido construido a partir de el trabajo previo reflejado en los siguientes artículos:

- Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2022). TabDDPM: Modelling Tabular Data with Diffusion Models. arXiv preprint arXiv:2209.15421.


- Kim, J., Lee, C., Shin, Y., Park, S., Kim, M., Park, N., & Cho, J. (2022, August). Sos: Score-based oversampling for tabular data. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 762-772).