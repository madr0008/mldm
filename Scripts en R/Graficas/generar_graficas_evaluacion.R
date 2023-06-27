#RADAR CHART

library(fmsb)
library(RColorBrewer)
library(scales)

maxCol <- function(d) {apply(as.matrix(d), 2, function(x){max(unlist(x))})}
minCol <- function(d) {apply(as.matrix(d), 2, function(x){min(unlist(x))})}

df <- read.csv("/home/mdavila/ejecucion_mldm/graficas/evaluacion/ExampleBasedF-Measure.csv", sep=";", dec=".")

datasets <- c("cal500", "corel5k", "emotions", "genbase", "medical", "scene", "chess", "yeast")
algoritmos <- c("Ninguno","LPROS", "MLROS", "MLSMOTE", "MLSOL", "REMEDIAL", "MLDM")

coul <- brewer.pal(8, "Accent")
colors_border <- coul
colors_in <- alpha(coul,0.3)

# Split the screen in 6 parts
par(mfrow=c(2,3))

titulos <- c("BPMLL", "BR.J48", "HOMER", "LP.J48", "MLkNN")

for(i in 1:5) {
  
  aux <- as.data.frame(do.call(rbind, lapply(algoritmos, function(a) { unlist(lapply(datasets, function(d) { df[df$MLD==d & df$Algoritmo==a,][[titulos[i]]] }))  })))
  rownames(aux) <- algoritmos
  colnames(aux) <- datasets
  #data <- rbind(rep(max(maxCol(aux)),8) , rep(min(minCol(aux)),8) , aux)
  #data <- rbind(rep(1,8), rep(0,8) , aux)
  
  # Custom the radarChart !
  radarchart( data, axistype=1, 
              pcol=colors_border, pfcol=colors_in, plwd=4, plty=1 , 
              cglcol="grey", cglty=3, axislabcol="grey", cglwd=0.8,
              vlcex=0.8,
              title=titulos[i]
  )
  
}

plot.new()
legend(x=0.5, y=0.5, legend = rownames(data[-c(1,2),]), bty = "n", pch=20, col=colors_in, text.col = "black", cex=2, pt.cex=3, xjust=0.5, yjust=0.5, title="Algoritmo")



#GRAFICO DE BARRAS

library(ggplot2)
library(grid)
library(gridExtra)
library(ggpubr)

df <- read.csv("/home/mdavila/ejecucion_mldm/graficas/evaluacion/One-Error.csv", sep=";", dec=".")
colores <- c("#000000", "#E41A1C", "#377EB8", "#FF7F00", "#984EA3", "#4DAF4A", "#804000")
orden_algoritmos <- c("Ninguno", "LPROS", "MLROS", "MLSMOTE", "MLSOL", "REMEDIAL", "MLDM")
df$Algoritmo <- factor(df$Algoritmo, levels = orden_algoritmos)

titulos <- c("BPMLL", "BR.J48", "HOMER", "LP.J48", "MLkNN")

grafica1 <- ggplot(df, aes(x = MLD, y = BPMLL, fill = Algoritmo)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "", y = "One-Error", fill = "Algoritmo de Resampling", title = titulos[1]) +
  scale_fill_manual(values = colores, breaks = orden_algoritmos) +
  theme_minimal() +
  theme(legend.position="none", plot.title=element_text(hjust = 0.5))

grafica2 <- ggplot(df, aes(x = MLD, y = BPMLL, fill = Algoritmo)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "", y = "One-Error", fill = "Algoritmo de Resampling", title = titulos[2]) +
  scale_fill_manual(values = colores, breaks = orden_algoritmos) +
  theme_minimal() +
  theme(legend.position="none", plot.title=element_text(hjust = 0.5))

grafica3 <- ggplot(df, aes(x = MLD, y = BPMLL, fill = Algoritmo)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "", y = "One-Error", fill = "Algoritmo de Resampling", title = titulos[3]) +
  scale_fill_manual(values = colores, breaks = orden_algoritmos) +
  theme_minimal() +
  theme(legend.position="none", plot.title=element_text(hjust = 0.5))

grafica4 <- ggplot(df, aes(x = MLD, y = BPMLL, fill = Algoritmo)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "", y = "One-Error", fill = "Algoritmo de Resampling", title = titulos[4]) +
  scale_fill_manual(values = colores, breaks = orden_algoritmos) +
  theme_minimal() +
  theme(legend.position="none", plot.title=element_text(hjust = 0.5))

grafica5 <- ggplot(df, aes(x = MLD, y = BPMLL, fill = Algoritmo)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "", y = "One-Error", fill = "Algoritmo de Resampling", title = titulos[5]) +
  scale_fill_manual(values = colores, breaks = orden_algoritmos) +
  theme_minimal() +
  theme(legend.position="none", plot.title=element_text(hjust = 0.5))

graficaAux <- ggplot(df, aes(x = MLD, y = BPMLL, fill = Algoritmo)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "", y = "One-Error", fill = "Algoritmo de Resampling", title = titulos[1]) +
  scale_fill_manual(values = colores, breaks = orden_algoritmos) +
  theme_minimal()

legend <- get_legend(graficaAux)
grafica6 <- as_ggplot(legend)

grid.arrange(grafica1, grafica2, grafica3, grafica4, grafica5, grafica6, ncol=2, nrow=3)
