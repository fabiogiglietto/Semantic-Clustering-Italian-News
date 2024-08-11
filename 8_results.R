# Load necessary packages
library(tidyverse)
library(knitr)
library(kableExtra)
library(broom)
library(emmeans)

# Load the saved results (assuming these files exist in the specified directory)
output_dir <- "./output"
analyzed_results <- readRDS(file.path(output_dir, "analyzed_results.rds"))
summary_data <- read.csv(file.path(output_dir, "clustering_summary.csv"))

# Filter out invalid results
summary_data_filtered <- summary_data %>%
  filter(
    n_clusters > 0,  # Remove solutions with 0 clusters
    !is.na(quality_score),  # Remove solutions with NA quality scores
    quality_score >= -3,  # Remove solutions with unreasonably low quality scores
    quality_score <= 3    # Remove solutions with unreasonably high quality scores
  )

# Check for any remaining invalid or extreme values
summary_stats <- summary_data_filtered %>%
  summarise(
    min_clusters = min(n_clusters, na.rm = TRUE),
    max_clusters = max(n_clusters, na.rm = TRUE),
    min_quality = min(quality_score, na.rm = TRUE),
    max_quality = max(quality_score, na.rm = TRUE)
  )

print(summary_stats)

# Print the number of rows before and after filtering
cat("Rows before filtering:", nrow(summary_data), "\n")
cat("Rows after filtering:", nrow(summary_data_filtered), "\n")

# If you want to see which rows were removed:
removed_rows <- anti_join(summary_data, summary_data_filtered)
if(nrow(removed_rows) > 0) {
  cat("Sample of removed rows:\n")
  print(head(removed_rows))
}

# Updated function to extract metrics for each configuration
get_metrics <- function(data, method, year) {
  data %>%
    filter(method == !!method, year == !!year) %>%
    group_by(model, use_umap) %>%
    reframe(
      n_solutions = n(),
      avg_quality = mean(quality_score, na.rm = TRUE),
      median_quality = median(quality_score, na.rm = TRUE),
      sd_quality = sd(quality_score, na.rm = TRUE),
      se_quality = sd(quality_score, na.rm = TRUE) / sqrt(n()),
      min_quality = min(quality_score, na.rm = TRUE),
      max_quality = max(quality_score, na.rm = TRUE),
      iqr_quality = IQR(quality_score, na.rm = TRUE),
      cv_quality = sd(quality_score, na.rm = TRUE) / mean(quality_score, na.rm = TRUE),
      min_clusters = min(n_clusters, na.rm = TRUE),
      max_clusters = max(n_clusters, na.rm = TRUE),
      avg_noise = ifelse(method == "hdbscan", mean(num_noise_points, na.rm = TRUE), NA)
    )
}

# Get metrics for each method and year
hdbscan_metrics_2018 <- get_metrics(summary_data_filtered, "hdbscan", 2018)
hdbscan_metrics_2022 <- get_metrics(summary_data_filtered, "hdbscan", 2022)
kmeans_metrics_2018 <- get_metrics(summary_data_filtered, "kmeans", 2018)
kmeans_metrics_2022 <- get_metrics(summary_data_filtered, "kmeans", 2022)

# Combine results
combined_metrics <- bind_rows(
  mutate(hdbscan_metrics_2018, method = "HDBSCAN", year = 2018),
  mutate(hdbscan_metrics_2022, method = "HDBSCAN", year = 2022),
  mutate(kmeans_metrics_2018, method = "K-means", year = 2018),
  mutate(kmeans_metrics_2022, method = "K-means", year = 2022)
)

# Create comparison table
comparison_table <- combined_metrics %>%
  select(method, year, model, use_umap, n_solutions, avg_quality, se_quality) %>%
  mutate(
    use_umap = ifelse(use_umap, "Yes", "No"),
    avg_quality = sprintf("%.3f", avg_quality),
    se_quality = sprintf("%.3f", se_quality)
  ) %>%
  arrange(year, method, model, use_umap)

# Print the table
print(kable(comparison_table, format = "pipe", caption = "Comparison of Clustering Quality Scores"))

# Aggregate the data
aggregated_table <- comparison_table %>%
  group_by(method, year, model, use_umap) %>%
  summarise(
    n_solutions = first(n_solutions),
    avg_quality = first(avg_quality),
    se_quality = first(se_quality),
    .groups = "drop"
  ) %>%
  arrange(year, desc(avg_quality), model, method, use_umap)  # Sort avg_quality from min to max

# Function to create table for a specific year
create_year_table <- function(data, year) {
  year_data <- data %>%
    filter(year == !!year) %>%
    select(-year) %>%
    mutate(
      avg_quality = sprintf("%.3f", as.numeric(avg_quality)),
      se_quality = sprintf("%.3f", as.numeric(se_quality))
    )
  
  total_solutions <- sum(year_data$n_solutions)
  
  year_data %>%
    bind_rows(tibble(
      method = "Total",
      model = "",
      use_umap = "",
      n_solutions = total_solutions,
      avg_quality = "",
      se_quality = ""
    ))
}

# Create tables for each year
table_2018 <- create_year_table(aggregated_table, 2018)
table_2022 <- create_year_table(aggregated_table, 2022)

# Print the tables
print(kable(table_2018, format = "pipe", caption = "Aggregated Comparison of Clustering Quality Scores (2018)"))
print(kable(table_2022, format = "pipe", caption = "Aggregated Comparison of Clustering Quality Scores (2022)"))


# Statistical tests

# 1. ANOVA for 2018
anova_model_2018 <- aov(quality_score ~ model * method, data = summary_data %>% filter(year == 2018))
print(summary(anova_model_2018))

# 2. ANOVA for 2022
anova_model_2022 <- aov(quality_score ~ model * method, data = summary_data %>% filter(year == 2022))
print(summary(anova_model_2022))

# 3. T-test to compare clustering methods for each year
t_test_result_2018 <- t.test(quality_score ~ method, data = summary_data %>% filter(year == 2018))
print(t_test_result_2018)

t_test_result_2022 <- t.test(quality_score ~ method, data = summary_data %>% filter(year == 2022))
print(t_test_result_2022)

# 4. Linear model to measure the effect of clustering method and use of UMAP for each year
lm_model_2018 <- lm(quality_score ~ method + use_umap + model, data = summary_data %>% filter(year == 2018))
print(summary(lm_model_2018))

lm_model_2022 <- lm(quality_score ~ method + use_umap + model, data = summary_data %>% filter(year == 2022))
print(summary(lm_model_2022))