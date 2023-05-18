# Descripción del fichero .toml para MLDM
A continuación se especifican los parámetros recogidos en el fichero `toml` que toma como argumento el algoritmo MLDM.

Argumentos principales:
- `seed = 0` -- Semilla para valores aleatorios
- `parent_dir = "datasets/birds"` -- directorio con los conjuntos de datos
- `real_data_path = "datasets/birds/birds"` -- subdirectorio en el que se encuentran los datos
- `model_type = "mlp"` -- Tipo de modelo para el proceso inverso de difusión
- `device = "cuda:0"` -- Dispositivo en el que se llevarán a cabo los cálculos

Parámetros del modelo MLP para difusión inversa:
- `d_layers = [256, 256]` -- Capas ocultas de la red neuronal
- `dropout = 0.0` -- Valor de exclusión de la red neuronal

Parámetros del proceso de difusión
- `num_timesteps = 1000` -- Número de pasos entre datos originales y ruido
- `gaussian_loss_type = "mse"` -- Métrica para el cálculo del error gaussiano
- `scheduler = "cosine"` -- Esquema de adición de ruido a los datos

Parámetros del entrenamiento:
- `steps = 1000` -- Número de pasos del proceso de entrenamiento
- `lr = 0.001` -- Tasa de aprendizaje
- `weight_decay = 1e-05` -- Decaimiento de los pesos
- `batch_size = 4096` -- 

Parámetros de transformación de datos:
- `seed = 0` -- Semilla para las transformaciones de los datos
- `normalization = "quantile"` -- Tipo de normalización en datos numéricos
- `cat_encoding = "__none__"` -- Transformación para datos categóricos. Por ejemplo, `ohe` (one hot encoding)

Parámetros del proceso de muestreo:
- `sample_percentage = 25` -- Porcentaje de instancias a generar respecto al tamaño original del dataset
- `strategy = "general"` -- Estrategia de muestreo
- `max_iterations = 10000` -- Número máximo de iteraciones para la estrategia general
- `batch_size = 1000` -- Número de muestras generadas por tanda

En conjunto, el fichero de configuración tiene este aspecto:

```toml
seed = 0
parent_dir = "../datasets/birds"
real_data_path = "../datasets/birds/birds"
model_type = "mlp"
device = "cuda:0"

[model_params.rtdl_params]
d_layers = [
    256,
    256,
]
dropout = 0.0

[diffusion_params]
num_timesteps = 1000
gaussian_loss_type = "mse"
scheduler = "cosine"

[train.main]
steps = 1000
lr = 0.001
weight_decay = 1e-05
batch_size = 4096

[train.T]
seed = 0
normalization = "quantile"
cat_encoding = "__none__"

[sample]
sample_percentage = 25
strategy = "general"
max_iterations = 1000
batch_size = 10000
seed = 0
```