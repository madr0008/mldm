datasets <- c("cal500", "corel5k", "emotions", "genbase", "medical", "scene", "chess", "yeast")

particiones = c("1", "2", "3", "4", "5")

#estrategias <- c("general", "specific" ,"mixed")
estrategias <- c("specific" ,"mixed")

normalizaciones <- c("standard", "minmax", "quantile")

cat_encodings <- list("one-hot","one-hot",c("none", "one-hot"),"one-hot","one-hot",c("none", "one-hot"),"one-hot",c("none", "one-hot"))

for (i in c(1:length(datasets))) {
  dataset <- datasets[i]
  for (particion in particiones) {
    for (estrategia in estrategias) {
      for (normalizacion in normalizaciones) {
        for (cat_encoding in cat_encodings[[i]]) {
          texto <- paste('seed = 0
parent_dir = "',paste("/datasets",dataset,particion,sep="/"),'"
real_data_path = "',paste("/datasets",dataset,particion,dataset,sep="/"),'"
output_file = "',paste(paste("/resultados",dataset,particion,paste(dataset,particion,estrategia,normalizacion,cat_encoding,sep="_"),sep="/"),"arff",sep="."),'"
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
steps = 50000
lr = 0.001
weight_decay = 1e-05
batch_size = 4096

[train.T]
seed = 0
normalization = "',normalizacion,'"
cat_encoding = "',ifelse(cat_encoding=="none","__none__","one-hot"),'"

[sample]
sample_percentage = 25
strategy = "mixed"
label_percentage = 25
max_iterations = 100000
normalize_last = "',ifelse(normalizacion=="quantile","False","True"),'"
normalize_quantile = "True"
batch_size = 1000
seed = 0',sep="")
        write(texto,file=paste("/home/mdavila/ejecucion_mldm/configs/",paste(dataset,particion,estrategia,normalizacion,cat_encoding,sep="_"),".toml",sep=""))    
        }
      }
    }
  }
}
