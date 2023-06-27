library(mldr)

datasets <- c("cal500", "corel5k", "emotions", "genbase", "medical", "scene", "chess", "yeast")

algoritmos <- c("LPROS", "MLROS", "MLSMOTE", "REMEDIAL", "MLSOL")

porDataset <- lapply(datasets, function(dataset) {
  porParticion <- lapply(c(1:5), function(i) {
    directorio <- paste("/home/miguelangel/resultados",dataset,as.character(i), sep="/")
    ficheros <- list.files(path=directorio, full.names=TRUE)
    salidas <- list()
    aux <- readRDS(paste("/home/miguelangel/datos/",dataset,"/",as.character(i),"/",dataset,"-stra-2x5-",as.character(i),"-tra.rds",sep=""))
    salidas[[length(salidas) + 1]] <- summary(aux)
    for (fichero in ficheros) {
      for (algoritmo in algoritmos) {
        if (grepl(algoritmo, fichero, fixed=TRUE)) {
          xml <- paste("/home/miguelangel/resultados/xml/",dataset,".xml",sep="")
          aux <- read.arff(fichero, use_xml=TRUE, auto_extension=FALSE, xml_file=xml)
          ds <- mldr_from_dataframe(aux$dataframe,aux$labelIndices,aux$attributes,aux$name)
          salidas[[length(salidas) + 1]] <- summary(ds)
        }
      }
    }
    salidas
  })
})

algoritmos2 <- c("Ninguno", "LPROS", "MLROS", "MLSMOTE", "REMEDIAL", "MLSOL")

linea <- "MLD;Algoritmo;Numero de Instancias;Numero de Labelsets;Numero de labelsets unicos;Cardinalidad;Densidad;meanIR;SCUMBLE;SCUMBLE.CV;TCS"
write(linea,file="Metricas_TFG.csv",append=TRUE)
for (d in c(1:length(porDataset))) {
  for (p in c(1:5)) {
    for (a in c(1:length(algoritmos2))) {
      aux <- paste(as.character(porDataset[[d]][[p]][[a]]), collapse=";")
      linea <- paste(datasets[d],p,algoritmos2[a],aux,sep=";")
      write(linea,file="Metricas_TFG.csv",append=TRUE)
    }
  }
}


#Chess normal
library(mldr.datasets)

for (i in c(1:5)) {
  aux <- readRDS(paste("/home/miguelangel/datos/chess/",as.character(i),"/chess-stra-2x5-",as.character(i),"-tra.rds",sep=""))
  mld <- mldr_from_dataframe(stackex_chess$dataset[aux,],stackex_chess$labels$index,stackex_chess$attributes,stackex_chess$name)
  summary(mld)
}
