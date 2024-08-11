# Install and load required packages
# install.packages("jsonlite")
# install.packages("readr")
library(jsonlite)
library(readr)
library(httr)
library(dplyr)

# Read the JSONL file line by line
lines <- read_lines("rawdata/thematic_coherence_data_adjusted.jsonl")

# Parse each line as a separate JSON object
data <- lapply(lines, fromJSON)

# Set seed for reproducibility
set.seed(42)

# Split the data into training and validation sets (80% training, 20% validation)
split_index <- sample(seq_len(length(data)), size = 0.8 * length(data))
training_data <- data[split_index]
validation_data <- data[-split_index]

# Save the training and validation sets as JSONL files
write_lines(sapply(training_data, toJSON, auto_unbox = TRUE), "data/training_data.jsonl")
write_lines(sapply(validation_data, toJSON, auto_unbox = TRUE), "data/validation_data.jsonl")

# Verify training data file
training_lines <- read_lines("data/training_data.jsonl")
cat("Training data first line:", training_lines[1], "\n")

# Verify validation data file
validation_lines <- read_lines("data/validation_data.jsonl")
cat("Validation data first line:", validation_lines[1], "\n")

write_lines('{"messages":[{"role":"system","content":"You are an expert in thematic coherence."},{"role":"user","content":"Text 1: Example text 1\nText 2: Example text 2"},{"role":"assistant","content":"<result>1</result>"}]}', "data/example_data.jsonl")

# Function to upload a file to OpenAI
upload_file <- function(file_path, api_key) {
  response <- httr::POST(
    url = "https://api.openai.com/v1/files",
    httr::add_headers(Authorization = paste("Bearer", api_key)),
    body = list(
      file = httr::upload_file(file_path),
      purpose = "fine-tune"
    )
  )

  # Parse the response to get the file ID
  response_content <- httr::content(response, as = "parsed")
  if (!is.null(response_content$error)) {
    stop(response_content$error$message)
  }
  return(response_content$id)
}

# Get the API key from the environment variable
api_key <- Sys.getenv("OPENAI_API_KEY")

# Check if the API key is available
if (api_key == "") {
  stop("API key not found in the environment variable 'OPENAI_VERA_PROJ_ID_API_KEY'")
}

# Upload example file
example_file_id <- upload_file("data/example_data.jsonl", api_key)
cat("Uploaded example file ID:", example_file_id, "\n")

# Upload training file
training_file_id <- upload_file("data/training_data.jsonl", api_key)
cat("Uploaded training file ID:", training_file_id, "\n")

# Upload validation file
validation_file_id <- upload_file("data/validation_data.jsonl", api_key)
cat("Uploaded validation file ID:", validation_file_id, "\n")

# Create a fine-tuning job with validation data
response <- httr::POST(
  url = "https://api.openai.com/v1/fine_tuning/jobs",
  httr::add_headers(Authorization = paste("Bearer", api_key)),
  body = jsonlite::toJSON(list(
    training_file = training_file_id,
    validation_file = validation_file_id,
    model = "gpt-4o-mini-2024-07-18"
  ), auto_unbox = TRUE),
  encode = "json"
)

# Parse the response to get the job ID
response_content <- httr::content(response, as = "parsed")
job_id <- response_content$id
cat("Created fine-tuning job ID:", job_id, "\n")

# Check the status of the fine-tuning job
response <- GET(
  url = paste0("https://api.openai.com/v1/fine_tuning/jobs/", job_id),
  add_headers(Authorization = paste("Bearer", api_key))
)

# Parse and print the response
response_content <- content(response, as = "parsed")
print(response_content)

# Function to retrieve fine-tuning job metrics
get_fine_tuning_job_metrics <- function(job_id, api_key) {
  response <- GET(
    url = paste0("https://api.openai.com/v1/fine_tuning/jobs/", job_id),
    add_headers(Authorization = paste("Bearer", api_key))
  )

  # Parse the response
  response_content <- content(response, as = "parsed")
  return(response_content)
}
