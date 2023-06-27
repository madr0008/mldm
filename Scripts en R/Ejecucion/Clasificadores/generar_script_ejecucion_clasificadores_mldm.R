datasets <- c("cal500", "corel5k", "emotions", "genbase", "medical", "scene", "chess", "yeast")
particiones <- c(1:5)
clasificadores <- c("BPMLL", "BR-J48", "HOMER", "LP-J48", "MLkNN")
estrategias <- c("specific", "mixed")
normalizaciones <- c("minmax","standard","quantile")

linea <- "#!/bin/bash"
write(linea,file="ejecucion.sh",append=TRUE)

for (clasificador in clasificadores) {
  for (dataset in datasets) {
    for (particion in particiones) {
      for (estrategia in estrategias) {
        for (normalizacion in normalizaciones) {
          linea <- paste("java -jar RunMLClassifier/RunMLClassifier.jar -path ","datasets/",dataset,"/",as.character(particion)," -dataset ",dataset,"_",as.character(particion),"_",estrategia,"_",normalizacion,"_one-hot"," -algorithm ",clasificador," >> salida.csv",sep="")
          write(linea,file="ejecucion.sh",append=TRUE)
        }
      }
    }
  }
}

