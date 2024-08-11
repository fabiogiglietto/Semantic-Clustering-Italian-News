library(parallel)
library(dplyr)
library(reticulate)
library(progress)

# Set up Python environment
reticulate::use_virtualenv("/home/fabio.giglietto/.virtualenvs/r-reticulate", required = TRUE)

# Import Python modules
umap_learn <- import("umap")
hdbscan <- import("hdbscan")
sklearn_decomposition <- import("sklearn.decomposition")
np <- import("numpy")

# Load the dataset
data <- readRDS("~/Semantic-Clustering-Italian-News/data/df_pol_combined_embeddings.rds")

perform_dim_reduction <- function(data, method, n_components, n_neighbors = NULL) {
  if (method == "UMAP") {
    reducer <- umap_learn$UMAP(n_components = as.integer(n_components),
                               n_neighbors = as.integer(n_neighbors),
                               random_state = 42L,
                               n_jobs = 1L)  # Set n_jobs to 1 to avoid warning
  } else if (method == "PCA") {
    if (is.null(n_components)) {
      reducer <- sklearn_decomposition$PCA(n_components = 0.8, random_state = 42L)
    } else {
      reducer <- sklearn_decomposition$PCA(n_components = as.integer(n_components), random_state = 42L)
    }
  }
  reduced_data <- reducer$fit_transform(data)
  return(reduced_data)
}

perform_clustering <- function(data, min_cluster_size) {
  cat("Starting HDBSCAN clustering with min_cluster_size:", min_cluster_size, "\n")
  
  # Set seed for NumPy
  np$random$seed(42L)
  
  clusterer <- hdbscan$HDBSCAN(
    min_cluster_size = as.integer(min_cluster_size),
    metric = 'euclidean'
  )
  result <- clusterer$fit(data)
  cat("HDBSCAN clustering completed\n")
  cat("Labels shape:", length(result$labels_), "\n")
  cat("Probabilities shape:", length(result$probabilities_), "\n")
  return(list(labels = result$labels_, probabilities = result$probabilities_))
}

create_clustering_summary <- function(data, embedding_col, method, n_components, n_neighbors, min_cluster_size, year) {
  cat("Starting clustering summary for:", year, embedding_col, method, "\n")
  cat("Data dimensions:", nrow(data), "rows,", ncol(data), "columns\n")
  
  embeddings <- data[[embedding_col]]
  
  if (is.null(embeddings) || length(embeddings) == 0) {
    cat("Error: Embedding column is empty or NULL\n")
    return(NULL)
  }
  
  # Convert list column to matrix
  embedding_matrix <- tryCatch({
    do.call(rbind, lapply(embeddings, unlist))
  }, error = function(e) {
    cat("Error creating embedding matrix:", conditionMessage(e), "\n")
    return(NULL)
  })
  
  if (is.null(embedding_matrix)) return(NULL)
  
  cat("Embedding matrix dimensions:", nrow(embedding_matrix), "rows,", ncol(embedding_matrix), "columns\n")
  
  tryCatch({
    # Convert R matrix to NumPy array
    py_embedding_matrix <- r_to_py(embedding_matrix)
    
    reduced_data <- perform_dim_reduction(py_embedding_matrix, method, n_components, n_neighbors)
    
    if (nrow(reduced_data) == 0) {
      cat("Error: Dimensionality reduction returned an empty matrix\n")
      return(NULL)
    }
    
    cat("Reduced data dimensions:", nrow(reduced_data), "rows,", ncol(reduced_data), "columns\n")
    
    clustering_result <- perform_clustering(reduced_data, min_cluster_size)
    
    cat("Clustering completed. Labels:", length(clustering_result$labels), "Probabilities:", length(clustering_result$probabilities), "\n")
    
    # Debug: Print clustering results
    cat("Unique labels:", paste(unique(clustering_result$labels), collapse=", "), "\n")
    cat("Range of probabilities:", range(clustering_result$probabilities), "\n")
    
    cat("Creating summary dataframe...\n")
    summary <- data.frame(
      year = year,
      method = method,
      embedding = embedding_col,  # Add embedding column
      n_components = if(method == "PCA" && is.null(n_components)) "80%" else n_components,
      min_cluster_size = min_cluster_size,
      n_clusters = length(unique(clustering_result$labels[clustering_result$labels != -1])),
      n_noise = sum(clustering_result$labels == -1),
      total_points = length(clustering_result$labels)
    )
    
    if (!is.null(n_neighbors)) {
      summary$n_neighbors <- n_neighbors
    }
    
    cat("Summary created:", nrow(summary), "rows,", ncol(summary), "columns\n")
    print(summary)  # Print the summary dataframe
    
    cat("Creating detailed results dataframe...\n")
    detailed_results <- data.frame(
      year = year,
      url_rid = data$url_rid,
      cluster = clustering_result$labels,
      probability = clustering_result$probabilities
    )
    
    cat("Detailed results created:", nrow(detailed_results), "rows,", ncol(detailed_results), "columns\n")
    print(head(detailed_results))  # Print the first few rows of detailed results
    
    cat("Returning results...\n")
    return(list(
      summary = summary,
      detailed_results = detailed_results
    ))
  }, error = function(e) {
    cat("Error in clustering process:", conditionMessage(e), "\n")
    return(NULL)
  })
}

# Create a function to process a single combination of parameters
process_combination <- function(params, id) {
  year <- params$year
  embedding <- params$embedding
  method <- params$method
  comp <- params$comp
  neigh <- params$neigh
  mcs <- params$mcs
  
  key <- paste(id, year, embedding, method, comp, ifelse(is.na(neigh), "NA", neigh), mcs, sep="_")
  message("Starting processing for: ", key)
  
  tryCatch({
    filtered_data <- data %>% filter(election == year)
    message("Data dimensions after filtering: ", nrow(filtered_data), " rows, ", ncol(filtered_data), " columns")
    
    if (nrow(filtered_data) == 0) {
      message("Warning: No data after filtering for year ", year)
      return(NULL)
    }
    
    result <- create_clustering_summary(
      filtered_data, embedding, method, 
      if(comp == "80pct" && method == "PCA") NULL else comp, 
      neigh, mcs, year
    )
    if (!is.null(result)) {
      message("Clustering completed successfully for: ", key)
      message("Summary dimensions: ", nrow(result$summary), " rows, ", ncol(result$summary), " columns")
      message("Detailed results dimensions: ", nrow(result$detailed_results), " rows, ", ncol(result$detailed_results), " columns")
      return(list(key = key, summary = result$summary, detailed = result$detailed_results))
    } else {
      message("Clustering returned NULL for: ", key)
      return(NULL)
    }
  }, error = function(e) {
    message("Error processing ", key, ": ", conditionMessage(e))
    return(NULL)
  })
}

# Parameters to test
umap_components <- c(5, 10)
umap_neighbors <- c(5, 15, 30)
pca_components <- c(5, 10, 20, "80pct")
min_cluster_sizes <- c(50, 100, 200, 300, 400, 500, 1000)

# Create the parameter combinations
param_combinations <- expand.grid(
  year = c("2018", "2022"),
  embedding = c("umberto_embedding", "text_embedding_3_large"),
  method = c("UMAP", "PCA"),
  comp = c(umap_components, pca_components),
  neigh = c(umap_neighbors, NA),
  mcs = min_cluster_sizes,
  stringsAsFactors = FALSE
) %>%
  filter(
    (method == "UMAP" & comp %in% umap_components & !is.na(neigh)) |
      (method == "PCA" & comp %in% pca_components & is.na(neigh))
  )

# Set up parallel processing
num_cores <- 48
cl <- makeCluster(num_cores)

# Export necessary functions and libraries to the cluster
clusterEvalQ(cl, {
  library(dplyr)
  library(reticulate)
  reticulate::use_virtualenv("/home/fabio.giglietto/.virtualenvs/r-reticulate", required = TRUE)
})

# Verify Python configuration and import modules on each node
clusterEvalQ(cl, {
  print(reticulate::py_config())
  
  tryCatch({
    umap_learn <- reticulate::import("umap")
    print("UMAP imported successfully")
  }, error = function(e) print(paste("Error importing UMAP:", e$message)))
  
  tryCatch({
    hdbscan <- reticulate::import("hdbscan")
    print("HDBSCAN imported successfully")
  }, error = function(e) print(paste("Error importing HDBSCAN:", e$message)))
  
  tryCatch({
    sklearn_decomposition <- reticulate::import("sklearn.decomposition")
    print("sklearn.decomposition imported successfully")
  }, error = function(e) print(paste("Error importing sklearn.decomposition:", e$message)))
  
  tryCatch({
    np <- reticulate::import("numpy")
    print("numpy imported successfully")
  }, error = function(e) print(paste("Error importing numpy:", e$message)))
  
  list(umap = !is.null(umap_learn),
       hdbscan = !is.null(hdbscan),
       sklearn = !is.null(sklearn_decomposition),
       numpy = !is.null(np))
})

clusterExport(cl, c("perform_dim_reduction", "perform_clustering", "create_clustering_summary", 
                    "data",
                    "umap_components", "umap_neighbors", "pca_components", "min_cluster_sizes",
                    "param_combinations", "process_combination"))

# Set up progress bar
total_combinations <- nrow(param_combinations)
pb <- progress_bar$new(
  format = "[:bar] :percent eta: :eta",
  total = total_combinations,
  clear = FALSE,
  width = 60
)

# Run the processing in parallel
results <- parLapplyLB(cl, seq_len(nrow(param_combinations)), function(i) {
  tryCatch({
    result <- process_combination(param_combinations[i,], i)
    result
  }, error = function(e) {
    message("Error in combination ", i, ": ", conditionMessage(e))
    NULL
  })
})

# Update progress bar after parallel processing
pb$update(1)

# Process the results
cat("Total results:", length(results), "\n")
cat("Non-null results:", sum(!sapply(results, is.null)), "\n")

all_summaries <- do.call(rbind, lapply(results, function(x) if(!is.null(x)) x$summary else NULL))
detailed_clustering_results <- list()
for (result in results) {
  if (!is.null(result) && !is.null(result$key) && !is.null(result$detailed)) {
    detailed_clustering_results[[result$key]] <- result$detailed
  }
}

cat("Rows in all_summaries:", nrow(all_summaries), "\n")
cat("Number of detailed results:", length(detailed_clustering_results), "\n")

# Print a sample of the results
if (!is.null(all_summaries) && nrow(all_summaries) > 0) {
  cat("Sample of all_summaries:\n")
  print(head(all_summaries))
} else {
  cat("No summaries available.\n")
}

if (length(detailed_clustering_results) > 0) {
  cat("Sample of first detailed result:\n")
  first_key <- names(detailed_clustering_results)[1]
  print(head(detailed_clustering_results[[first_key]]))
} else {
  cat("No detailed results available.\n")
}

# Print the first few non-null results
cat("First few non-null results:\n")
non_null_results <- Filter(Negate(is.null), results)
for (i in 1:min(5, length(non_null_results))) {
  cat("Result", i, "key:", non_null_results[[i]]$key, "\n")
  print(str(non_null_results[[i]]))
}

# Explicit check for matching number of solutions
if (nrow(all_summaries) == length(detailed_clustering_results)) {
  cat("The number of solutions in summaries and detailed results match.\n")
} else {
  cat("WARNING: Mismatch in number of solutions!\n")
  cat("Summaries:", nrow(all_summaries), "\n")
  cat("Detailed results:", length(detailed_clustering_results), "\n")
}

# Save the clustering summaries and detailed results
saveRDS(results, "./data/clustering_results.rds")
