# Load necessary packages
library(tidyverse)
library(knitr)
library(kableExtra)
library(ggplot2)
library(scales)  # For pretty_breaks()

# Load the saved results
output_dir <- "./output"
analyzed_results <- readRDS(file.path(output_dir, "analyzed_results.rds"))
summary_data <- read.csv(file.path(output_dir, "clustering_summary.csv"))

# Filter out solutions with 0 clusters
summary_data_filtered <- summary_data %>%
  filter(n_clusters > 0)

# Prepare data for the bar chart
chart_data <- summary_data %>%
  filter(n_clusters > 0) %>%
  group_by(model, year, method, use_umap) %>%
  summarise(
    mean_quality = mean(quality_score, na.rm = TRUE),
    se_quality = sd(quality_score, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(config = paste(method, ifelse(use_umap, "UMAP", "No UMAP")))

# Create the grouped bar chart
quality_chart <- ggplot(chart_data, aes(x = model, y = mean_quality, fill = config)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.8) +
  geom_errorbar(aes(ymin = mean_quality - se_quality, ymax = mean_quality + se_quality),
                position = position_dodge(width = 0.9), width = 0.25) +
  facet_wrap(~ year, scales = "free_x", nrow = 1) +
  labs(title = "Mean Quality Scores by Model, Year, Method, and UMAP Usage",
       x = "Model", y = "Mean Quality Score",
       fill = "Configuration") +
  scale_fill_brewer(palette = "Set2", name = "Configuration") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom",
        legend.box = "vertical",
        legend.margin = margin()) +
  scale_y_continuous(breaks = pretty_breaks(n = 10))

# Print the chart
print(quality_chart)

# Save the chart
ggsave(file.path(output_dir, "quality_scores_comparison.png"), quality_chart, width = 12, height = 8, dpi = 300)

# Print summary of the chart data
cat("\nSummary of Mean Quality Scores:\n")
print(chart_data)