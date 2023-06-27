library(mldr)

datasets <- c("cal500", "corel5k", "emotions", "genbase", "medical", "scene", "chess", "yeast")
particiones <- c(1:5)
estrategias <- c("specific", "mixed")
normalizaciones <- c("minmax","standard","quantile")

porDataset <- lapply(datasets, function(dataset) {
  porParticion <- lapply(particiones, function(i) {
    directorio <- paste("/home/mdavila/ejecucion_mldm/resultados_mldm",dataset,as.character(i), sep="/")
    salidas <- list()
    for (estrategia in estrategias) {
      for (normalizacion in normalizaciones) {
        nombre <- paste(directorio,paste(dataset,as.character(i),estrategia,normalizacion,"one-hot",sep="_"),sep="/")
        aux <- read.arff(paste(nombre,"arff",sep="."), use_xml=TRUE, auto_extension=FALSE, xml_file=paste("/home/mdavila/ejecucion_mldm/xml/",dataset,".xml",sep=""))
        ds <- mldr_from_dataframe(aux$dataframe,aux$labelIndices,aux$attributes,aux$name)
        salidas[[length(salidas) + 1]] <- summary(ds)
      }
    }
    salidas
  })
})


linea <- "MLD;Particion;Algoritmo;Estrategia;Normalizacion;Numero de Instancias;Numero de Labelsets;Numero de labelsets unicos;Cardinalidad;Densidad;meanIR;SCUMBLE;SCUMBLE.CV;TCS"
write(linea,file="Metricas_mldm.csv",append=TRUE)
for (d in c(1:length(porDataset))) {
  for (p in c(1:5)) {
    for (e in c(1:2)) {
      for (n in c(1:3)) {
        aux <- paste(as.character(porDataset[[d]][[p]][[e+n]]), collapse=";")
        linea <- paste(datasets[d],as.character(p),"MLDM",estrategias[e],normalizaciones[n],aux,sep=";")
        write(linea,file="Metricas_mldm.csv",append=TRUE)
      }
    }
  }
}
