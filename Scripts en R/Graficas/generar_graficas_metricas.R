library(ggplot2)

df <- read.csv("/home/mdavila/ejecucion_mldm/graficas/metricas/meanIR.csv", sep=";", dec=".")
colores <- c("#000000", "#E41A1C", "#377EB8", "#FF7F00", "#984EA3", "#4DAF4A", "#804000")
orden_algoritmos <- c("Ninguno", "LPROS", "MLROS", "MLSMOTE", "MLSOL", "REMEDIAL", "MLDM")
df$Algoritmo <- factor(df$Algoritmo, levels = orden_algoritmos)

ggplot(df, aes(x = MLD, y = MeanIR, fill = Algoritmo)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "MLD", y = "MeanIR", fill = "Algoritmo de Resampling") +
  scale_fill_manual(values = colores, breaks = orden_algoritmos) +
  scale_y_log10() +
  theme_minimal()


df2 <- read.csv("/home/mdavila/ejecucion_mldm/graficas/metricas/SCUMBLE.csv", sep=";", dec=".")
df2$Algoritmo <- factor(df2$Algoritmo, levels = orden_algoritmos)
df2$SCUMBLE = df2$SCUMBLE*10000

ggplot(df2, aes(x = MLD, y = SCUMBLE, fill = Algoritmo)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "MLD", y = "SCUMBLE", fill = "Algoritmo de Resampling") +
  scale_fill_manual(values = colores, breaks = orden_algoritmos) +
  scale_y_log10() +
  theme_minimal()
