linea <- "#!/bin/bash"
write(linea,file="/home/mdavila/ejecucion_mldm/ejecucion.sh",append=TRUE)

ficheros <- list.files(path="/home/mdavila/ejecucion_mldm/configs")

for (fichero in ficheros) {
  linea <- paste("python /mldm/scripts/pipeline.py --config /ejecucion/configs/",fichero,sep="")
  write(linea,file="/home/mdavila/ejecucion_mldm/ejecucion.sh",append=TRUE)
}
