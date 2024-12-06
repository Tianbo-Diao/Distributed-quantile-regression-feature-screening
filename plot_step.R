library(tidyverse)
library(patchwork)


df1 <- readxl::read_excel("simulation_data.xlsx",sheet = "plot-iid")
df2 <- readxl::read_excel("simulation_data.xlsx",sheet = "plot-hete")
df3 <- readxl::read_excel("simulation_data.xlsx",sheet = "plot-wang")
df4 <- readxl::read_excel("simulation_data.xlsx",sheet = "plot-diverge")


# Reshape the data into a long format
data_long1 <- df1 %>%
  pivot_longer(cols = SC:FDR, names_to = "Metric", values_to = "Value") %>%
  mutate( Metric = factor(Metric, levels = c("SC", "CF", "PSR", "FDR")), m=factor(m, levels = c("m=5", "m=10", "m=20") ), methods = factor(methods, levels = c("Pearson", "Kendall", "SIRS", "DC", "DFS-QR (0.25)", "DFS-QR (0.50)", "DFS-QR (0.75)"))  ) 

p1 <- ggplot(  data_long1, aes(x = Metric, y = Value, group = methods, colour = methods)   ) +
  geom_line(  ) +               
  geom_point(  ) +   
  facet_grid(m ~ setting) +  
  # theme_minimal() +            # Use a minimal theme
  labs(
    title = "Performance of indicators for example 1",
    x = "Metric",
    y = "Value",
    color = "Methods"
  )  +
  theme(
    plot.title = element_text(size = 10)  
  )



data_long2 <- df2 %>%
  pivot_longer(cols = SC:FDR, names_to = "Metric", values_to = "Value") %>%
  mutate( Metric = factor(Metric, levels = c("SC", "CF", "PSR", "FDR")), m=factor(m, levels = c("m=5", "m=10", "m=20") ), methods = factor(methods, levels = c("Pearson", "Kendall", "SIRS", "DC", "DFS-QR (0.25)", "DFS-QR (0.50)", "DFS-QR (0.75)"))  )  

p2 <- ggplot(  data_long2, aes(x = Metric, y = Value, group = methods, colour = methods)   ) +
  geom_line(  ) +               
  geom_point(  ) +   
  facet_grid(m ~ setting) +  
  # theme_minimal() +            # Use a minimal theme
  labs(
    title = "Performance of indicators for example 2",
    x = "Metric",
    y = "Value",
    color = "Methods"
  ) +
  theme(
    plot.title = element_text(size = 10)  
  )



data_long3 <- df3 %>%
  pivot_longer(cols = SC:FDR, names_to = "Metric", values_to = "Value") %>%
  mutate( Metric = factor(Metric, levels = c("SC", "CF", "PSR", "FDR")), m=factor(m, levels = c("m=5", "m=10", "m=20") ), methods = factor(methods, levels = c("Pearson", "Kendall", "SIRS", "DC", "DFS-QR (0.25)", "DFS-QR (0.50)", "DFS-QR (0.75)"))  )  

p3 <- ggplot(  data_long3, aes(x = Metric, y = Value, group = methods, colour = methods)   ) +
  geom_line(  ) +               
  geom_point(  ) +   
  facet_grid(m ~ setting) +  
  # theme_minimal() +            # Use a minimal theme
  labs(
    title = "Performance of indicators for example 3",
    x = "Metric",
    y = "Value",
    color = "Methods"
  ) +
  theme(
    plot.title = element_text(size = 10)  
  )



data_long4 <- df4 %>%
  pivot_longer(cols = SC:FDR, names_to = "Metric", values_to = "Value") %>%
  mutate( Metric = factor(Metric, levels = c("SC", "CF", "PSR", "FDR")), m=factor(m, levels = c("m=5", "m=10", "m=20") ), methods = factor(methods, levels = c("Pearson", "Kendall", "SIRS", "DC", "DFS-QR (0.25)", "DFS-QR (0.50)", "DFS-QR (0.75)"))  )  

p4 <- ggplot(  data_long4, aes(x = Metric, y = Value, group = methods, colour = methods)   ) +
  geom_line(  ) +               
  geom_point(  ) +   
  facet_grid(m ~ setting) +  
  # theme_minimal() +            # Use a minimal theme
  labs(
    title = "Performance of indicators for example 4",
    x = "Metric",
    y = "Value",
    color = "Methods"
  ) +
  theme(
    plot.title = element_text(size = 10)  
  )

p <- p1 + p2 + plot_layout(guides = "collect")

q <- p3 + p4 + plot_layout(guides = "collect")



ggsave(
  plot = q,
  filename = "example-3-4.png",
  path = "picture",
  width = 8,
  height = 5,
  dpi = 1000
)




# p1 <-  ggplot(data_long, aes(x = Metric, y = Value ) ) +
#   geom_line( aes(color = factor(m)) ) +
#   geom_point( aes(color = factor(m), shape = factor(m)), size=0.8 ) +
#   labs(
#     x = "Metric",
#     y = "Value",
#   )   +
#   scale_y_continuous( limits = c(0, 2.50), breaks=c(seq(0,2.50,0.25)),  labels = scales::number_format(accuracy = 0.01) ) +
#   scale_x_continuous( breaks= pretty_breaks() ) +
#   theme_bw() +
#   facet_grid(vars(class), vars(setting), scales = "free_x") +
#   theme( legend.position = "bottom" ) +
#   guides(color = guide_legend(title = "number of machines"), shape = guide_legend(title = "number of machines"))
# p1










