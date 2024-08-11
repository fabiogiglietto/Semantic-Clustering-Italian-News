library(tidyverse)
library(progress)
library(reticulate)
library(data.table)

# Set up PyTorch
use_virtualenv("r-reticulate")  # Replace with your virtual environment name if different
py_config()  # This will print Python configuration information

compute_pairwise_consistency <- function(clustering_results, election_year) {
  torch <- reticulate::import("torch")
  py_gc <- reticulate::import("gc")
  
  # Extract solution indices for the given election year
  solutions <- which(sapply(clustering_results, function(x) x$summary$year == election_year))
  n_solutions <- length(solutions)
  
  if (n_solutions == 0) {
    stop(sprintf("No solutions found for election year %s", election_year))
  }
  
  # Get all unique items (url_rids) across all solutions for this year
  items <- unique(unlist(lapply(solutions, function(sol) clustering_results[[sol]]$detailed$url_rid)))
  n_items <- length(items)
  
  cat(sprintf("Computing pairwise consistency for %d items across %d solutions in election %s\n", 
              n_items, n_solutions, election_year))
  
  # Create a mapping of url_rid to index
  item_to_index <- setNames(seq_along(items) - 1, items)  # 0-based indexing for Python
  
  # Check if CUDA is available
  if (torch$cuda$is_available()) {
    cat("CUDA is available. Using GPU for computations.\n")
    device <- torch$device("cuda:0")
    torch$cuda$set_per_process_memory_fraction(0.5, device)
  } else {
    cat("CUDA is not available. Using CPU for computations.\n")
    device <- torch$device("cpu")
  }
  
  # Initialize a CPU tensor to store pairwise consistency
  consistency_matrix <- torch$zeros(c(n_items, n_items), device="cpu")
  
  # Create progress bar
  pb <- progress_bar$new(
    format = "  Processing solutions [:bar] :percent eta: :eta",
    total = n_solutions,
    clear = FALSE,
    width = 60
  )
  
  # For each solution in the specific year, update the consistency matrix
  for (sol in solutions) {
    sol_data <- clustering_results[[sol]]$detailed
    
    # Create a vector of cluster assignments for this solution
    cluster_vector <- rep(NA_integer_, n_items)
    cluster_vector[item_to_index[sol_data$url_rid] + 1] <- sol_data$cluster
    
    # Convert to torch tensor on GPU
    cluster_tensor <- torch$tensor(cluster_vector, dtype=torch$long, device=device)
    
    # Create a boolean matrix of same-cluster assignments
    same_cluster <- (cluster_tensor$unsqueeze(0L) == cluster_tensor$unsqueeze(1L))
    
    # Add to consistency matrix (move to CPU immediately)
    consistency_matrix <- consistency_matrix + same_cluster$float()$cpu()
    
    # Clear GPU tensors and free up memory
    rm(cluster_tensor, same_cluster)
    py_gc$collect()  # Python garbage collection
    torch$cuda$empty_cache()  # Clear PyTorch cache
    
    gc()
    
    # Update progress bar
    pb$tick()
  }
  
  # Normalize the consistency matrix
  consistency_matrix <- consistency_matrix / n_solutions
  
  cat("Creating pairwise consistency dataframe...\n")
  
  # Convert to numpy array
  consistency_cpu <- consistency_matrix$numpy()
  
  # Clear tensor
  rm(consistency_matrix)
  gc()
  py_gc$collect()  # Python garbage collection
  torch$cuda$empty_cache()  # Clear PyTorch cache
  
  nonzero_indices <- which(consistency_cpu != 0, arr.ind = TRUE)
  pairwise_consistency <- data.table(
    election = as.integer(election_year),
    url_rid1 = items[nonzero_indices[,1]],
    url_rid2 = items[nonzero_indices[,2]],
    consistency = round(consistency_cpu[nonzero_indices], 4)  # Reduce precision to 4 decimal places
  )[url_rid1 < url_rid2]  # Keep only unique pairs
  
  # Clear large CPU objects
  rm(consistency_cpu, nonzero_indices)
  gc()  # R garbage collection
  
  setDF(pairwise_consistency)  # Convert back to data.frame
  
  cat("Pairwise consistency computation completed.\n")
  
  pairwise_consistency
}

create_validation_pairs <- function(item_summaries, pairwise_consistency, year, n_pairs = 1000) {
  tryCatch({
    cat("Starting function for year:", year, "\n")
    
    # Ensure input are data.tables
    setDT(item_summaries)
    setDT(pairwise_consistency)
    
    cat("Data converted to data.tables\n")
    
    # Filter summaries for the specific year and ensure freq_noise is numeric
    summaries <- item_summaries[election == year]
    summaries[, freq_noise := as.numeric(freq_noise)]
    
    cat("Summaries filtered for year:", year, "\n")
    cat("Number of rows in summaries:", nrow(summaries), "\n")
    
    # Ensure consistency column is numeric
    pairwise_consistency[, consistency := as.numeric(consistency)]
    
    cat("Consistency column converted to numeric\n")
    cat("First few values of consistency:", head(pairwise_consistency$consistency), "\n")
    
    # Calculate thresholds
    consistency_75 <- quantile(pairwise_consistency$consistency, 0.75, na.rm = TRUE)
    consistency_25 <- quantile(pairwise_consistency$consistency, 0.25, na.rm = TRUE)
    consistency_50 <- quantile(pairwise_consistency$consistency, 0.50, na.rm = TRUE)
    noise_median <- median(summaries$freq_noise, na.rm = TRUE)
    noise_75 <- quantile(summaries$freq_noise, 0.75, na.rm = TRUE)
    
    cat("Thresholds calculated\n")
    cat("consistency_75:", consistency_75, "\n")
    cat("consistency_25:", consistency_25, "\n")
    cat("consistency_50:", consistency_50, "\n")
    cat("noise_median:", noise_median, "\n")
    cat("noise_75:", noise_75, "\n")
    
    # Intra-cluster pairs
    intra_pairs <- pairwise_consistency[consistency > consistency_75]
    cat("Intra-cluster pairs filtered by consistency\n")
    intra_pairs <- intra_pairs[summaries, on = .(url_rid1 = url_rid), nomatch = 0]
    cat("Intra-cluster pairs joined with summaries (url_rid1)\n")
    intra_pairs <- intra_pairs[summaries, on = .(url_rid2 = url_rid), nomatch = 0]
    cat("Intra-cluster pairs joined with summaries (url_rid2)\n")
    
    intra_pairs <- intra_pairs[freq_noise < noise_median & i.freq_noise < noise_median]
    cat("Intra-cluster pairs filtered by noise\n")
    intra_pairs <- intra_pairs[order(-consistency)][1:min(n_pairs, .N), .(url_rid1, url_rid2, consistency)]
    
    cat("Intra-cluster pairs calculated\n")
    cat("Number of intra-cluster pairs:", nrow(intra_pairs), "\n")
    
    # Inter-cluster pairs
    inter_pairs <- pairwise_consistency[consistency > consistency_25 & consistency < consistency_50]
    cat("Inter-cluster pairs filtered by consistency\n")
    inter_pairs <- inter_pairs[summaries, on = .(url_rid1 = url_rid), nomatch = 0]
    cat("Inter-cluster pairs joined with summaries (url_rid1)\n")
    inter_pairs <- inter_pairs[summaries, on = .(url_rid2 = url_rid), nomatch = 0]
    cat("Inter-cluster pairs joined with summaries (url_rid2)\n")
    inter_pairs <- inter_pairs[freq_noise < noise_median & i.freq_noise < noise_median, .(url_rid1, url_rid2, consistency)]
    
    cat("Inter-cluster pairs calculated\n")
    cat("Number of inter-cluster pairs:", nrow(inter_pairs), "\n")
    
    # Noise pairs
    noise_pairs <- pairwise_consistency[summaries, on = .(url_rid1 = url_rid), nomatch = 0]
    cat("Noise pairs joined with summaries (url_rid1)\n")
    noise_pairs <- noise_pairs[summaries, on = .(url_rid2 = url_rid), nomatch = 0]
    cat("Noise pairs joined with summaries (url_rid2)\n")
    noise_pairs <- noise_pairs[((freq_noise > noise_75 & i.freq_noise < noise_median) | 
                                  (i.freq_noise > noise_75 & freq_noise < noise_median)), 
                               .(url_rid1, url_rid2, consistency)]
    
    cat("Noise pairs calculated\n")
    cat("Number of noise pairs:", nrow(noise_pairs), "\n")
    
    # Sample inter and noise pairs if necessary
    if (nrow(inter_pairs) > n_pairs) {
      inter_pairs <- inter_pairs[sample(.N, n_pairs)]
    }
    if (nrow(noise_pairs) > n_pairs) {
      noise_pairs <- noise_pairs[sample(.N, n_pairs)]
    }
    
    cat("Sampling completed\n")
    
    list(intra = intra_pairs, inter = inter_pairs, noise = noise_pairs)
  }, error = function(e) {
    cat("Error occurred:", conditionMessage(e), "\n")
    cat("Error occurred in:", conditionCall(e), "\n")
    print(traceback())
    stop(e)
  })
}
# Main execution

# Load the clustering results
clustering_results <- readRDS("./data/clustering_results.rds")

cat("Processing clustering results...\n")

# Create progress bar for processing results
pb <- progress_bar$new(
  format = "  Processing clustering results [:bar] :percent eta: :eta",
  total = length(clustering_results),
  clear = FALSE,
  width = 60
)

# Process all results
all_results <- rbindlist(lapply(seq_along(clustering_results), function(i) {
  result <- clustering_results[[i]]$detailed
  key <- clustering_results[[i]]$key
  setDT(result)
  result[, solution := key]
  # The year is already in the 'year' column, no need to extract it from the key
  pb$tick()
  result
}), use.names = TRUE, fill = TRUE)

cat("\nCalculating summary statistics...\n")

# Now calculate summary statistics across all solutions for each url_rid
item_summaries <- all_results[, .(
  freq_noise = mean(cluster == -1, na.rm = TRUE),
  n_solutions = .N,
  most_common_cluster = if(all(cluster == -1, na.rm = TRUE)) NA_integer_ else as.integer(names(which.max(table(cluster[cluster != -1])))),
  freq_most_common = if(all(cluster == -1, na.rm = TRUE)) 0 else max(table(cluster[cluster != -1])) / .N,
  election = first(year)
), by = .(url_rid)]

# Set key for faster operations later
setkey(item_summaries, url_rid, election)

# Print summary statistics
cat("\nSummary of freq_noise:\n")
print(summary(item_summaries$freq_noise))

cat("\nDistribution of n_solutions:\n")
print(table(item_summaries$n_solutions))

cat("\nSummary of freq_most_common:\n")
print(summary(item_summaries$freq_most_common))

# Save the results
cat("\nSaving item summaries...\n")
saveRDS(item_summaries, "./data/item_cluster_summaries.rds")

cat("Item summaries saved. Process complete.\n")

# Check for pairwise consistency files
pairwise_consistency_2018_file <- "./data/pairwise_consistency_2018.rds"
pairwise_consistency_2022_file <- "./data/pairwise_consistency_2022.rds"

if (file.exists(pairwise_consistency_2018_file)) {
  cat("Loading pairwise consistency for 2018 from existing file...\n")
  pairwise_consistency_2018 <- readRDS(pairwise_consistency_2018_file)
} else {
  cat("Computing pairwise consistency for 2018...\n")
  pairwise_consistency_2018 <- compute_pairwise_consistency(clustering_results, "2018")
  saveRDS(pairwise_consistency_2018, pairwise_consistency_2018_file)
}

if (file.exists(pairwise_consistency_2022_file)) {
  cat("Loading pairwise consistency for 2022 from existing file...\n")
  pairwise_consistency_2022 <- readRDS(pairwise_consistency_2022_file)
} else {
  cat("Computing pairwise consistency for 2022...\n")
  pairwise_consistency_2022 <- compute_pairwise_consistency(clustering_results, "2022")
  saveRDS(pairwise_consistency_2022, pairwise_consistency_2022_file)
}

cat("Creating validation pairs for each election year...\n")
validation_pairs_2018 <- create_validation_pairs(item_summaries, pairwise_consistency_2018, "2018")
validation_pairs_2022 <- create_validation_pairs(item_summaries, pairwise_consistency_2022, "2022")

saveRDS(validation_pairs_2018, "./data/validation_pairs_by_year_2018.rds")
saveRDS(validation_pairs_2022, "./data/validation_pairs_by_year_2022.rds")

cat("Process completed.\n")