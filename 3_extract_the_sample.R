library(tidyverse)

# Load the main dataset
df <- readRDS("./data/df_pol_combined_embeddings.rds")

# Load the validation pairs
validation_pairs_2018 <- readRDS("./data/validation_pairs_by_year_2018.rds")
validation_pairs_2022 <- readRDS("./data/validation_pairs_by_year_2022.rds")

# Function to sample pairs
sample_pairs <- function(pairs, n) {
  pairs %>% slice_sample(n = min(n, nrow(.)))
}

# Function to select intra pairs based on consistency score
select_intra_pairs <- function(pairs, n) {
  pairs %>% 
    arrange(desc(consistency)) %>% 
    slice_head(n = min(n, nrow(.)))
}

# Function to extract text for pairs
extract_text <- function(pairs, df) {
  pairs %>%
    left_join(df %>% select(url_rid, text_a = alltext_clean), by = c("url_rid1" = "url_rid")) %>%
    left_join(df %>% select(url_rid, text_b = alltext_clean), by = c("url_rid2" = "url_rid"))
}

# Function to create sample
create_sample <- function(year, validation_pairs, df, total_pairs, intra_ratio = 0.5, inter_ratio = 0.3, noise_ratio = 0.2) {
  cat(paste("Creating sample for year", year, "\n"))
  
  intra_pairs <- select_intra_pairs(validation_pairs$intra, ceiling(total_pairs * intra_ratio))
  inter_pairs <- sample_pairs(validation_pairs$inter, ceiling(total_pairs * inter_ratio))
  noise_pairs <- sample_pairs(validation_pairs$noise, ceiling(total_pairs * noise_ratio))
  
  sample <- bind_rows(
    extract_text(intra_pairs, df) %>% mutate(category = "intra"),
    extract_text(inter_pairs, df) %>% mutate(category = "inter"),
    extract_text(noise_pairs, df) %>% mutate(category = "noise")
  ) %>% 
    mutate(year = year)
  
  # Remove pairs with identical text
  sample <- sample %>% filter(text_a != text_b)
  
  # Limit each url_rid to appear at most 100 times
  url_rid_counts <- table(c(sample$url_rid1, sample$url_rid2))
  url_rid_to_keep <- names(url_rid_counts[url_rid_counts <= 10])
  
  sample %>% filter(url_rid1 %in% url_rid_to_keep & url_rid2 %in% url_rid_to_keep)
}

# Create samples for each year
total_pairs <- 6000
year_2018_ratio <- 0.5  # 3000 pairs for 2018
year_2022_ratio <- 0.5  # 3000 pairs for 2022

sample_2018 <- create_sample("2018", validation_pairs_2018, df, total_pairs * year_2018_ratio)
sample_2022 <- create_sample("2022", validation_pairs_2022, df, total_pairs * year_2022_ratio)

# Combine samples
combined_sample <- bind_rows(sample_2018, sample_2022)

# Print available pairs in each category
print("Available pairs after initial filtering:")
print(table(combined_sample$year, combined_sample$category))

# Function to sample the required number of pairs for each category and year
sample_category <- function(data, n) {
  data %>% slice_sample(n = min(n, nrow(.)))
}

# Calculate the required number of pairs for each category and year
n_intra_2018 <- floor(total_pairs * year_2018_ratio * 0.5)
n_inter_2018 <- floor(total_pairs * year_2018_ratio * 0.3)
n_noise_2018 <- floor(total_pairs * year_2018_ratio * 0.2)

n_intra_2022 <- floor(total_pairs * year_2022_ratio * 0.5)
n_inter_2022 <- floor(total_pairs * year_2022_ratio * 0.3)
n_noise_2022 <- floor(total_pairs * year_2022_ratio * 0.2)

# Sample the required number of pairs for each category and year
final_sample <- bind_rows(
  sample_category(combined_sample %>% filter(year == "2018" & category == "intra"), n_intra_2018),
  sample_category(combined_sample %>% filter(year == "2018" & category == "inter"), n_inter_2018),
  sample_category(combined_sample %>% filter(year == "2018" & category == "noise"), n_noise_2018),
  sample_category(combined_sample %>% filter(year == "2022" & category == "intra"), n_intra_2022),
  sample_category(combined_sample %>% filter(year == "2022" & category == "inter"), n_inter_2022),
  sample_category(combined_sample %>% filter(year == "2022" & category == "noise"), n_noise_2022)
)

# Print final sample distribution
print("Final sample distribution:")
print(table(final_sample$year, final_sample$category))

# Reshuffle the sample
final_sample <- final_sample %>% slice_sample(n = nrow(final_sample))

# Save the final sample as a CSV file
write_csv(final_sample, "./data/final_sample.csv")
