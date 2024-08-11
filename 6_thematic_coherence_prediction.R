library(tidyverse)
library(progress)
library(openai)

# Read and preprocess the data
coherence_rating_sample <- read_csv("./data/final_sample.csv",
                                         col_types = cols(year = col_character()))

coherence_rating_sample <- coherence_rating_sample %>%
  mutate(input = paste("Text 1:", text_a, "\nText 2:", text_b))

# Read the JSONL file and convert to a dataframe
data <- jsonlite::stream_in(file("rawdata/validation_data.jsonl"))
df <- as_tibble(data)

# Flatten the dataframe
flat_df <- df %>%
  unnest(messages) %>%
  group_by(group = ceiling(row_number() / 3)) %>%
  pivot_wider(names_from = role, values_from = content) %>%
  ungroup() %>%
  select(-group)

# Extract the system message
system_message <- flat_df$system[1]

# Function to get prediction
get_prediction <- function(user_input, system_message) {
  tryCatch({
    response <- openai::create_chat_completion(
      model = "ft:gpt-4o-mini-2024-07-18:uniurb::9ovTt9Dp",
      messages = list(
        list(role = "system", content = system_message),
        list(role = "user", content = user_input)
      ),
      temperature = 0
    )
    as.numeric(str_extract(response$choices$message.content, "(?<=<result>)\\d+(?=</result>)"))
  }, error = function(e) {
    warning(paste("Error in prediction:", e$message))
    return(NA)
  })
}

# Get predictions for all inputs with a progress bar
total_inputs <- nrow(coherence_rating_sample)
pb <- progress_bar$new(
  format = "[:bar] :percent ETA: :eta",
  total = total_inputs,
  clear = FALSE,
  width = 60
)

predictions <- map_dbl(coherence_rating_sample$input, function(input) {
  result <- get_prediction(input, system_message)
  pb$tick()
  return(result)
})

# Add predictions to the dataframe
coherence_rating_sample <- coherence_rating_sample %>%
  mutate(model_rating = predictions)

saveRDS(coherence_rating_sample, "./data/coherence_rating_sample_rated.rds")

# Calculate statistics, excluding non-codable pairs (99) and disregarding the year
statistics_overall <- coherence_rating_sample %>%
  filter(model_rating != 99) %>%  # Exclude non-codable pairs
  group_by(category) %>%
  summarise(
    mean_prediction = mean(model_rating, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  arrange(category)

# Print the overall statistics
print("Overall statistics (disregarding year):")
print(statistics_overall)

# Calculate statistics by year (as before)
statistics_by_year <- coherence_rating_sample %>%
  filter(model_rating != 99) %>%  # Exclude non-codable pairs
  group_by(year, category) %>%
  summarise(
    mean_prediction = mean(model_rating, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  arrange(year, category)

# Print the statistics by year
print("Statistics by year:")
print(statistics_by_year)

# Optionally, you can save both statistics to CSV files
write_csv(statistics_overall, "./data/prediction_statistics_overall.csv")
write_csv(statistics_by_year, "./data/prediction_statistics_by_year.csv")

# Prepare data for ANOVA and post-hoc tests by year
anova_data <- coherence_rating_sample %>%
  filter(model_rating != 99) %>%  # Exclude non-codable pairs
  select(year, category, model_rating)

# Perform one-way ANOVA and Tukey's HSD post-hoc test by year
anova_results_by_year <- anova_data %>%
  group_by(year) %>%
  nest() %>%
  mutate(
    anova = map(data, ~ aov(model_rating ~ category, data = .x)),
    summary = map(anova, summary),
    tukey = map(anova, TukeyHSD)
  )

# Print ANOVA and Tukey's HSD results by year
anova_results_by_year %>%
  rowwise() %>%
  mutate(print_results = list({
    cat("Year:", year, "\n")
    cat("One-way ANOVA results:\n")
    print(summary)
    cat("Tukey's HSD post-hoc test results:\n")
    print(tukey)
  }))

# Visualize the results
ggplot(anova_data, aes(x = category, y = model_rating)) +
  geom_boxplot() +
  facet_wrap(~ year, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Predictions by Category and Year",
       x = "Category",
       y = "Prediction")

ggsave("prediction_distribution_by_year.png", width = 12, height = 8)