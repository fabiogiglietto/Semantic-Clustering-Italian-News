library(tidyverse)
library(stringr)
library(reticulate)
library(pbapply)
library(futile.logger)
library(openai)

Sys.setenv(TOKENIZERS_PARALLELISM = "false")

# Set up logging
flog.appender(appender.file("embedding_process.log"))

# Use reticulate to interface with Python
use_virtualenv("r-reticulate")  # Replace with your virtual environment name if different
py_config()  # This will print Python configuration information

# Import required Python libraries
transformers <- import("transformers")
torch <- import("torch")

# Check if CUDA is available
cuda_available <- torch$cuda$is_available()
flog.info(paste("CUDA available:", cuda_available))

# Set the device
device <- if(cuda_available) torch$device("cuda") else torch$device("cpu")
flog.info(paste("Using device:", device$type))

# Load the data
data <- readRDS("./rawdata/classified_urls_clean.rds")
flog.info(paste("Initial number of rows:", nrow(data)))

# Remove duplicates
data <- unique(data)
flog.info(paste("Number of rows after removing duplicates:", nrow(data)))

# Process the data
data <- data %>%
  mutate(alltext_clean = case_when(
    is.na(share_title) & is.na(share_main_blurb) ~ NA_character_,
    is.na(share_title) ~ share_main_blurb,
    is.na(share_main_blurb) ~ share_title,
    TRUE ~ paste(share_title, share_main_blurb, sep = ". ")
  )) %>%
  mutate(alltext_clean = alltext_clean %>%
           str_remove_all("<.*?>") %>%
           str_remove_all("(http|https)://[[:alnum:]/]+") %>%
           str_trim()
  ) %>%
  select(url_rid, election, alltext_clean)

# Load BERT model and tokenizer
model_name <- "Musixmatch/umberto-commoncrawl-cased-v1"
tokenizer <- transformers$AutoTokenizer$from_pretrained(model_name)
model <- transformers$AutoModel$from_pretrained(model_name)$to(device)

# Function to get BERT embedding
get_bert_embedding <- function(text) {
  tryCatch({
    inputs <- tokenizer(text, return_tensors = "pt", padding = TRUE, truncation = TRUE, max_length = 512L)
    input_list <- list(
      input_ids = inputs$input_ids$to(device),
      attention_mask = inputs$attention_mask$to(device)
    )
    with(torch$no_grad(), {
      outputs <- do.call(model, input_list)
    })
    last_hidden_state <- outputs$last_hidden_state
    attention_mask <- input_list$attention_mask
    input_mask_expanded <- attention_mask$unsqueeze(-1L)$expand_as(last_hidden_state)$float()
    sum_embeddings <- torch$sum(last_hidden_state * input_mask_expanded, 1L)
    sum_mask <- torch$clamp(input_mask_expanded$sum(1L), min = 1e-9)
    mean_pooled <- sum_embeddings / sum_mask
    embeddings <- mean_pooled$squeeze()$cpu()$numpy()
    return(embeddings)
  }, error = function(e) {
    flog.error(paste("Error in get_bert_embedding:", e$message))
    return(NULL)
  })
}

# Function to get OpenAI embedding with retry logic
get_openai_embedding <- function(text) {
  retry_attempts <- 3
  for (attempt in 1:retry_attempts) {
    result <- tryCatch({
      openai::create_embedding(model = "text-embedding-3-large", input = text)
    }, error = function(e) {
      flog.warn(paste("Error occurred:", e$message, "Attempt", attempt, "of", retry_attempts))
      NULL
    })
    if (!is.null(result)) {
      return(as.numeric(result$data$embedding[[1]]))
    }
    Sys.sleep(0.1)  # Wait before retrying
  }
  return(rep(NA_real_, 3072))
}

# Function to process embeddings with retry
process_embeddings <- function(texts, embedding_function, max_retries = 3) {
  embeddings <- pblapply(texts, function(text) {
    for (i in 1:max_retries) {
      result <- embedding_function(text)
      if (!is.null(result)) return(result)
      flog.warn(paste("Retry", i, "for text:", substr(text, 1, 50), "..."))
    }
    flog.error(paste("Failed to generate embedding after", max_retries, "attempts for text:", substr(text, 1, 50), "..."))
    return(NULL)
  })
  return(embeddings)
}

# Test with 100 items
set.seed(123)  # for reproducibility
test_indices <- sample(nrow(data), 100)
test_data <- data[test_indices, ]

flog.info("Starting test with 100 items for BERT embeddings")
test_bert_embeddings <- process_embeddings(test_data$alltext_clean, get_bert_embedding)

flog.info("Starting test with 100 items for OpenAI embeddings")
test_openai_embeddings <- process_embeddings(test_data$alltext_clean, get_openai_embedding)

# Check test results
null_bert_embeddings <- which(sapply(test_bert_embeddings, is.null))
null_openai_embeddings <- which(sapply(test_openai_embeddings, is.null))

if (length(null_bert_embeddings) > 0) {
  flog.warn(paste("Test: Failed to generate BERT embeddings for", length(null_bert_embeddings), "out of 100 items"))
} else {
  flog.info("Test: Successfully generated BERT embeddings for all 100 test items")
}

if (length(null_openai_embeddings) > 0) {
  flog.warn(paste("Test: Failed to generate OpenAI embeddings for", length(null_openai_embeddings), "out of 100 items"))
} else {
  flog.info("Test: Successfully generated OpenAI embeddings for all 100 test items")
}

# If test is successful, process the entire dataset
if (length(null_bert_embeddings) == 0 && length(null_openai_embeddings) == 0) {
  flog.info("Starting to process the entire dataset")
  data$umberto_embedding <- process_embeddings(data$alltext_clean, get_bert_embedding)
  data$text_embedding_3_large <- process_embeddings(data$alltext_clean, get_openai_embedding)
  
  # Final check for NULL embeddings
  null_bert_indices <- which(sapply(data$umberto_embedding, is.null))
  null_openai_indices <- which(sapply(data$text_embedding_3_large, is.null))
  
  if (length(null_bert_indices) > 0) {
    flog.warn(paste("There are still", length(null_bert_indices), "NULL BERT embeddings after processing"))
  } else {
    flog.info("Successfully generated BERT embeddings for all items in the dataset")
  }
  
  if (length(null_openai_indices) > 0) {
    flog.warn(paste("There are still", length(null_openai_indices), "NULL OpenAI embeddings after processing"))
  } else {
    flog.info("Successfully generated OpenAI embeddings for all items in the dataset")
  }
  
  # Save the modified dataset
  saveRDS(data, "./data/df_pol_combined_embeddings.rds")
  flog.info("Saved the dataset with embeddings")
} else {
  flog.error("Test failed. Please review the logs and address any issues before processing the entire dataset.")
}