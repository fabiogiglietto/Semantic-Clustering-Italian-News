# Install and load required packages
# install.packages("jsonlite")
# install.packages("readr")
library(jsonlite)
library(readr)
library(httr)
library(tidyverse)
library(openai)

# Step 2: Read the JSONL file
data <- jsonlite::stream_in(file("data/validation_data.jsonl"))

# Step 3: Convert to a dataframe
df <- as.data.frame(data)

flat_df <- df %>%
  unnest(messages) %>%
  group_by(group = ceiling(row_number() / 3)) %>%
  pivot_wider(names_from = role, values_from = content) %>%
  ungroup() %>%
  select(-group)

# Extract the actual labels from the 'assistant' column
flat_df$actual_label <- as.numeric(str_extract(flat_df$assistant, "(?<=<result>)\\d+(?=</result>)"))

# Extract the system message (assuming it's the same for all rows)
system_message <- flat_df$system[1]

# Prepare the user inputs
user_inputs <- flat_df$user

get_prediction <- function(user_input, system_message) {
  response <- openai::create_chat_completion(
    model = "ft:gpt-4o-mini-2024-07-18:uniurb::9ovTt9Dp",
    messages = list(
      list(role = "system", content = system_message),
      list(role = "user", content = user_input)
    ),
    temperature = 0  # Set to 0 for most deterministic output
  )

  # Extract the prediction from the response
  response$choices$message.content
}

# Get predictions for all inputs
raw_predictions <- map(user_inputs, safely(~get_prediction(., system_message)))

# Extract successful predictions and their indices
successful_preds <- keep(raw_predictions, ~!is.null(.$result))
successful_indices <- which(map_lgl(raw_predictions, ~!is.null(.$result)))

# Extract numeric labels from the successful predictions
predicted_labels <- map_chr(successful_preds, ~.$result) %>%
  str_extract("(?<=<result>)\\d+(?=</result>)") %>%
  as.numeric()

# Add predictions to the dataframe
flat_df$predicted_label <- NA_real_
flat_df$predicted_label[successful_indices] <- predicted_labels

# Calculate metrics
library(caret)

# Convert labels to factors with all possible levels
all_levels <- sort(unique(c(flat_df_clean$actual_label, flat_df_clean$predicted_label)))
flat_df_clean$actual_label <- factor(flat_df_clean$actual_label, levels = all_levels)
flat_df_clean$predicted_label <- factor(flat_df_clean$predicted_label, levels = all_levels)

# Create confusion matrix
cm <- confusionMatrix(flat_df_clean$predicted_label, 
                      flat_df_clean$actual_label,
                      mode = "everything")

# Print overall accuracy
print(paste("Accuracy:", cm$overall['Accuracy']))

# Print class-wise metrics
print("Recall by class:")
print(cm$byClass[,'Recall'])
print("Precision by class:")
print(cm$byClass[,'Precision'])
print("F1 Score by class:")
print(cm$byClass[,'F1'])

# Calculate and print macro-averaged metrics
print(paste("Macro Avg Recall:", mean(cm$byClass[,'Recall'], na.rm = TRUE)))
print(paste("Macro Avg Precision:", mean(cm$byClass[,'Precision'], na.rm = TRUE)))
print(paste("Macro Avg F1 Score:", mean(cm$byClass[,'F1'], na.rm = TRUE)))