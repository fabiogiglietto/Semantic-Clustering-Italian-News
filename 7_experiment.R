Sys.setenv(CUDA_VISIBLE_DEVICES="0,2,1")
Sys.setenv(TF_GPU_ALLOCATOR = "cuda_malloc_asyn")

# Add this at the beginning of your script
library(futile.logger)
flog.appender(appender.tee("clustering_experiment.log"))

flog.info("Starting clustering experiment script")

library(tidyverse)
library(data.table)
library(reticulate)
library(fpc)
library(stats)
library(parallel)

flog.info("Libraries loaded successfully")

# Set up Python environment
use_virtualenv("r-reticulate", required = TRUE)
flog.info("Python environment set up")

# Function to check and install Python packages
install_if_missing <- function(packages) {
  lapply(packages, function(package) {
    if (!py_module_available(package)) {
      flog.info(paste("Installing Python package:", package))
      system2("pip", args = c("install", package))
    }
  })
}

# Install necessary Python packages
flog.info("Checking and installing necessary Python packages")
install_if_missing(c("umap-learn", "hdbscan", "numpy", "scikit-learn", "torch", "kmeans_pytorch", "gc"))

# Create R interfaces to Python functions
flog.info("Importing Python modules")
umap <- reticulate::import("umap")
hdbscan <- reticulate::import("hdbscan")
np <- reticulate::import("numpy")
sklearn_pairwise <- reticulate::import("sklearn.metrics.pairwise")
kmeans_pytorch <- reticulate::import("kmeans_pytorch")
torch <- reticulate::import("torch")
py_gc <- reticulate::import("gc")

# Function for comprehensive cleanup
cleanup_memory <- function() {
  py_gc$collect()
  gc()
  if (torch$cuda$is_available()) torch$cuda$empty_cache()
}

cleanup_memory()

# Global variable for distance matrices
global_distance_matrices <- list()
reduced_matrices <- list()

# R function to compute cosine distance
compute_cosine_distance_r <- function(data, embedding_col, year, data_name) {
  key <- paste(data_name, embedding_col, year, sep = "_")
  
  flog.info(paste("Computing cosine distance for", key))
  flog.info(paste("Data dimensions:", nrow(data), "x", ncol(data)))
  
  if (!embedding_col %in% names(data)) {
    flog.error(paste("Embedding column", embedding_col, "not found in data"))
    return(NULL)
  }
  
  if (!is.null(global_distance_matrices[[key]])) {
    flog.info(paste("Using CACHED cosine distance matrix for", key))
    return(global_distance_matrices[[key]])
  }
  
  flog.info(paste("Computing NEW cosine distance matrix for", key))
  embeddings <- data[[embedding_col]]
  
  if (is.null(embeddings) || length(embeddings) == 0) {
    flog.error("Embeddings are NULL or empty")
    return(NULL)
  }
  
  flog.info(paste("Embeddings length:", length(embeddings)))
  
  embedding_matrix <- tryCatch({
    do.call(rbind, lapply(embeddings, unlist))
  }, error = function(e) {
    flog.error(paste("Error creating embedding matrix:", e$message))
    return(NULL)
  })
  
  if (is.null(embedding_matrix) || nrow(embedding_matrix) == 0) {
    flog.error("Embedding matrix is NULL or empty")
    return(NULL)
  }
  
  flog.info(paste("Embedding matrix dimensions:", nrow(embedding_matrix), "x", ncol(embedding_matrix)))
  
  distance_matrix <- tryCatch({
    sklearn_pairwise$cosine_distances(embedding_matrix)
  }, error = function(e) {
    flog.error(paste("Error computing cosine distances:", e$message))
    return(NULL)
  })
  
  if (is.null(distance_matrix)) {
    flog.error("Failed to compute cosine distances")
    return(NULL)
  }
  
  # Store the computed matrix for future use
  global_distance_matrices[[key]] <<- distance_matrix
  
  flog.info(paste("Cosine distance computation completed for", key))
  flog.info(paste("Distance matrix dimensions:", nrow(distance_matrix), "x", ncol(distance_matrix)))
  
  return(distance_matrix)
}

# Load data
flog.info("Loading data from RDS files")
data <- readRDS("~/Semantic-Clustering-Italian-News/data/df_pol_combined_embeddings.rds")
coherence_rating <- readRDS("./data/coherence_rating_sample_rated.rds")
flog.info("Data loaded successfully")

# After loading your data
if (!"url_rid" %in% names(data)) {
  flog.warn("'url_rid' column not found in data. Adding row numbers as identifiers.")
  data$url_rid <- row_number(data)
}

# Prepare data for parallel processing
flog.info("Preparing data for parallel processing")
data_list <- list(
  text_2018 = list(data = data %>% filter(election == "2018"), col = "text_embedding_3_large"),
  text_2022 = list(data = data %>% filter(election == "2022"), col = "text_embedding_3_large"),
  umberto_2018 = list(data = data %>% filter(election == "2018"), col = "umberto_embedding"),
  umberto_2022 = list(data = data %>% filter(election == "2022"), col = "umberto_embedding")
)

# Compute distance matrices in parallel
flog.info("Computing distance matrices in parallel")
n_cores <- detectCores()
flog.info(paste("Using", n_cores, "cores for parallel processing"))

# Use mclapply but assign its results to a temporary variable
temp_results <- mclapply(names(data_list), function(name) {
  x <- data_list[[name]]
  year <- unique(x$data$election)
  result <- compute_cosine_distance_r(x$data, x$col, year, name)
  list(name = name, key = paste(name, x$col, year, sep = "_"), result = result)
}, mc.cores = n_cores)

# Update global_distance_matrices with the results
for (item in temp_results) {
  global_distance_matrices[[item$key]] <- item$result
}

flog.info("Distance matrices computed and stored in global_distance_matrices")
flog.info(paste("Keys in global_distance_matrices:", 
                paste(names(global_distance_matrices), collapse = ", ")))

# Optionally, remove the temporary variable to free up memory
rm(temp_results)
gc()

# Update save_progress function to handle multiple iterations
save_progress <- function(results, data_name, method, params, filename = "clustering_progress.rds") {
  current_progress <- if (file.exists(filename)) readRDS(filename) else list()
  if (is.null(current_progress[[data_name]])) current_progress[[data_name]] <- list()
  if (is.null(current_progress[[data_name]][[method]])) current_progress[[data_name]][[method]] <- list()
  
  # Create a unique key for this experiment
  key <- paste(names(params), params, sep = "_", collapse = "_")
  
  current_progress[[data_name]][[method]][[key]] <- results
  saveRDS(current_progress, filename)
  flog.info(paste("Progress saved for", data_name, "-", method, "-", key))
}

# UMAP Dimension Reduction Function
perform_umap <- function(data, embedding_col, n_components = 10, n_neighbors = 5, year) {
  flog.info(paste("Performing UMAP for year", year, "with", n_components, "components and", n_neighbors, "neighbors"))
  key <- paste(deparse(substitute(data)), embedding_col, n_components, n_neighbors, year, sep = "_")
  
  if (!is.null(reduced_matrices[[key]])) {
    flog.info("Using cached UMAP result")
    return(reduced_matrices[[key]])
  }
  
  flog.info("Computing new UMAP result")
  embeddings <- data[[embedding_col]]
  embedding_matrix <- do.call(rbind, lapply(embeddings, unlist))
  embeddings_np <- r_to_py(embedding_matrix)
  
  umap_model <- umap$UMAP(n_components = as.integer(n_components),
                          n_neighbors = as.integer(n_neighbors),
                          random_state = 42L,
                          verbose = TRUE)
  umap_result <- umap_model$fit_transform(embeddings_np)
  
  flog.info(paste("UMAP completed. Output dimensionality:", ncol(umap_result)))
  
  reduced_matrices[[key]] <<- umap_result
  return(umap_result)
}

# Modified HDBSCAN function with enhanced logging
perform_hdbscan_clustering <- function(data, embedding_col, use_umap = TRUE, n_components = 10, n_neighbors = 5, min_cluster_size = 50, year, data_name) {
  tryCatch({
    flog.info(paste("Starting HDBSCAN clustering for", data_name, "year", year))
    flog.info(paste("Parameters: use_umap =", use_umap, ", n_components =", n_components, 
                    ", n_neighbors =", n_neighbors, ", min_cluster_size =", min_cluster_size))
    
    if (!embedding_col %in% names(data)) {
      flog.error(paste("Embedding column", embedding_col, "not found in data for", data_name))
      return(NULL)
    }
    
    if (use_umap) {
      flog.info("Performing UMAP")
      embedding_matrix <- perform_umap(data, embedding_col, n_components, n_neighbors, year)
      metric <- 'euclidean'
    } else {
      flog.info("Computing cosine distance")
      embedding_matrix <- compute_cosine_distance_r(data, embedding_col, year, data_name)
      metric <- 'precomputed'
    }
    
    if (is.null(embedding_matrix) || nrow(embedding_matrix) == 0) {
      flog.error("Embedding matrix is NULL or empty")
      return(NULL)
    }
    
    flog.info(paste("Embedding matrix dimensions:", nrow(embedding_matrix), "x", ncol(embedding_matrix)))
    
    embeddings_np <- r_to_py(embedding_matrix)
    
    flog.info("Creating HDBSCAN model")
    hdbscan_model <- hdbscan$HDBSCAN(min_cluster_size = as.integer(min_cluster_size),
                                     metric = metric,
                                     core_dist_n_jobs = -1)
    
    flog.info("Fitting HDBSCAN model")
    hdbscan_result <- hdbscan_model$fit(embeddings_np)
    
    num_clusters <- length(unique(hdbscan_result$labels_[hdbscan_result$labels_ != -1]))
    num_noise_points <- sum(hdbscan_result$labels_ == -1)
    
    flog.info(paste("HDBSCAN completed -", "Number of clusters:", num_clusters, "Number of noise points:", num_noise_points))
    
    data$cluster <- as.vector(hdbscan_result$labels_)
    
    return(list(
      data = data,
      stats = data.frame(
        n_clusters = num_clusters,
        num_noise_points = num_noise_points,
        use_umap = use_umap,
        n_components = n_components,
        n_neighbors = n_neighbors,
        min_cluster_size = min_cluster_size
      )
    ))
  }, error = function(e) {
    flog.error(paste("Error in perform_hdbscan_clustering:", e$message))
    flog.error(paste("Traceback:", paste(capture.output(traceback()), collapse = "\n")))
    return(NULL)
  })
}

perform_kmeans_clustering <- function(data, embedding_col, n_clusters, use_umap = FALSE, n_components = 10, n_neighbors = 5, tol = 1e-4, max_iter = 300, year, data_name) {
  tryCatch({
    flog.info(paste("Starting K-means clustering for", data_name, "year", year))
    
    # Input validation
    if (!embedding_col %in% names(data)) {
      flog.error(paste("Embedding column", embedding_col, "not found in data for", data_name))
      return(NULL)
    }
    
    if (n_clusters < 1) {
      flog.error(paste("Invalid number of clusters:", n_clusters, "for", data_name, "year", year))
      return(NULL)
    }
    
    # Prepare embedding matrix
    if (use_umap) {
      flog.info(paste("Performing UMAP for", data_name, "year", year))
      embedding_matrix <- tryCatch({
        perform_umap(data, embedding_col, n_components, n_neighbors, year)
      }, error = function(e) {
        flog.error(paste("UMAP failed:", e$message))
        return(NULL)
      })
      if (is.null(embedding_matrix)) {
        return(NULL)
      }
    } else {
      flog.info(paste("Preparing embedding matrix for", data_name, "year", year))
      embeddings <- data[[embedding_col]]
      embedding_matrix <- tryCatch({
        do.call(rbind, lapply(embeddings, unlist))
      }, error = function(e) {
        flog.error(paste("Failed to prepare embedding matrix:", e$message))
        return(NULL)
      })
      if (is.null(embedding_matrix)) {
        return(NULL)
      }
    }
    
    cleanup_memory()
    
    # Perform K-means clustering
    clustering_successful <- FALSE
    tryCatch({
      device <- if (torch$cuda$is_available()) "cuda" else "cpu"
      flog.info(paste("Running K-means on device:", device))
      
      matrix_torch <- torch$from_numpy(embedding_matrix)$to(device)
      
      time_fit_predict <- system.time({
        res <- kmeans_pytorch$kmeans(
          X = matrix_torch,
          num_clusters = as.integer(n_clusters),
          tol = tol,
          distance = "cosine",
          device = device,
          iter_limit = as.integer(max_iter)
        )
        cluster_assignments <- as.vector(res[[1]]$cpu()$numpy())
      })
      
      clustering_successful <- TRUE
      
      if (length(unique(cluster_assignments)) == 1) {
        flog.warn(paste("All points assigned to the same cluster for", data_name, "year", year))
      }
      
      rm(matrix_torch, res)
      cleanup_memory()
      
    }, error = function(e) {
      flog.error(paste("K-means clustering failed:", e$message))
      clustering_successful <- FALSE
    })
    
    if (clustering_successful) {
      data$cluster <- cluster_assignments + 1
      actual_n_clusters <- length(unique(data$cluster))
      flog.info(paste("K-means clustering completed for", data_name, "year", year, 
                      "- Number of clusters:", actual_n_clusters, 
                      "Time taken:", time_fit_predict["elapsed"], "seconds"))
      
      return(list(
        data = data,
        stats = data.frame(
          n_clusters = actual_n_clusters,
          num_noise_points = 0,  # K-means doesn't have noise points
          use_umap = use_umap,
          n_components = n_components,
          n_neighbors = n_neighbors
        )
      ))
    } else {
      flog.error("K-means clustering failed. Returning NULL.")
      return(NULL)
    }
  }, error = function(e) {
    flog.error(paste("Error in perform_kmeans_clustering:", e$message))
    flog.error(paste("Traceback:", paste(capture.output(traceback()), collapse = "\n")))
    return(NULL)
  })
}

# Cluster Statistics Function
calculate_cluster_stats <- function(clustered_data, cluster_col) {
  if (!cluster_col %in% names(clustered_data)) {
    flog.warn(paste("Column", cluster_col, "not found in data. Returning NA for all statistics."))
    return(data.frame(
      num_clusters = NA,
      max_cluster_size = NA,
      min_cluster_size = NA,
      avg_cluster_size = NA,
      median_cluster_size = NA,
      num_outliers = NA
    ))
  }
  
  clustered_data %>%
    group_by(!!sym(cluster_col)) %>%
    summarise(count = n()) %>%
    summarise(
      num_clusters = n(),
      max_cluster_size = max(count),
      min_cluster_size = min(count),
      avg_cluster_size = mean(count),
      median_cluster_size = median(count),
      num_outliers = sum(clustered_data[[cluster_col]] == -1)
    )
}

# Cluster Quality Function
calculate_cluster_quality <- function(clusters_df, pair_data, cluster_col = "cluster") {
  flog.info(paste("Starting cluster quality calculation for column:", cluster_col))
  
  if (!cluster_col %in% names(clusters_df)) {
    flog.warn(paste("Column", cluster_col, "not found in data. Returning NA for all quality metrics."))
    return(list(
      quality_score = NA,
      avg_same_cluster = NA,
      avg_diff_cluster = NA,
      n_same = NA,
      n_diff = NA,
      n_total = NA
    ))
  }
  
  # Filter out pairs with model_rating of 99
  pair_data_filtered <- pair_data %>%
    filter(model_rating != 99)
  
  flog.info(paste("Number of pairs after filtering:", nrow(pair_data_filtered)))
  
  # Join cluster assignments to pair data
  pair_data_filtered <- pair_data_filtered %>%
    left_join(clusters_df %>% select(url_rid, !!sym(cluster_col)), by = c("url_rid1" = "url_rid")) %>%
    left_join(clusters_df %>% select(url_rid, !!sym(cluster_col)), by = c("url_rid2" = "url_rid"), suffix = c("_a", "_b"))
  
  flog.info(paste("Number of pairs after joining:", nrow(pair_data_filtered)))
  
  # Determine if pairs are in the same cluster
  pair_data_filtered <- pair_data_filtered %>%
    mutate(same_cluster = !!sym(paste0(cluster_col, "_a")) == !!sym(paste0(cluster_col, "_b")) &
             !!sym(paste0(cluster_col, "_a")) != -1 & !!sym(paste0(cluster_col, "_b")) != -1)
  
  # Calculate average similarity for same-cluster and different-cluster pairs
  avg_same_cluster <- mean(pair_data_filtered$model_rating[pair_data_filtered$same_cluster], na.rm = TRUE)
  avg_diff_cluster <- mean(pair_data_filtered$model_rating[!pair_data_filtered$same_cluster], na.rm = TRUE)
  
  flog.info(paste("Average similarity for same-cluster pairs:", avg_same_cluster))
  flog.info(paste("Average similarity for different-cluster pairs:", avg_diff_cluster))
  
  # Calculate cluster quality score
  quality_score <- avg_same_cluster - avg_diff_cluster
  
  flog.info(paste("Calculated quality score:", quality_score))
  
  # Count number of same-cluster and different-cluster pairs
  n_same <- sum(pair_data_filtered$same_cluster, na.rm = TRUE)
  n_diff <- sum(!pair_data_filtered$same_cluster, na.rm = TRUE)
  
  flog.info(paste("Number of same-cluster pairs:", n_same))
  flog.info(paste("Number of different-cluster pairs:", n_diff))
  
  result <- list(
    quality_score = quality_score,
    avg_same_cluster = avg_same_cluster,
    avg_diff_cluster = avg_diff_cluster,
    n_same = n_same,
    n_diff = n_diff,
    n_total = nrow(pair_data_filtered)
  )
  
  flog.info("Cluster quality calculation completed. Results:")
  flog.info(paste(capture.output(print(result)), collapse = "\n"))
  
  return(result)
}

# New function for parameter grid creation
create_parameter_grid <- function(method) {
  if (method == "hdbscan") {
    grid <- expand.grid(
      min_cluster_size = c(10, 50, 100, 200),
      use_umap = c(TRUE, FALSE),
      n_components = c(5, 10),
      n_neighbors = c(5, 15, 30)
    )
    # For non-UMAP runs, we only need one combination of n_components and n_neighbors
    grid <- grid[!(grid$use_umap == FALSE & (grid$n_components != 5 | grid$n_neighbors != 5)), ]
  } else if (method == "kmeans") {
    fixed_clusters <- c(25L, 50L, 75L, 100L, 125L, 150L, 175L, 200L)
    grid <- expand.grid(
      n_clusters = fixed_clusters,
      use_umap = c(TRUE, FALSE),
      n_components = c(5, 10),
      n_neighbors = c(5, 15, 30)
    )
    # For non-UMAP runs, we only need one combination of n_components and n_neighbors
    grid <- grid[!(grid$use_umap == FALSE & (grid$n_components != 5 | grid$n_neighbors != 5)), ]
  }
  return(grid)
}

# Function to determine optimal number of clusters for KMeans
determine_optimal_clusters <- function(data, embedding_col, max_clusters = 200) {
  embeddings <- data[[embedding_col]]
  embedding_matrix <- do.call(rbind, lapply(embeddings, unlist))
  
  # Convert R matrix to a PyTorch tensor
  embedding_tensor <- torch$from_numpy(np$array(embedding_matrix, dtype = torch$float32))
  device <- ifelse(torch$cuda$is_available(), "cuda", "cpu")
  embedding_tensor <- embedding_tensor$to(device)
  
  # Prepare to track within-cluster sum of squares (WCSS)
  wcss <- numeric(max_clusters)
  
  for (k in 1:max_clusters) {
    flog.info(paste("Calculating WCSS for k =", k))
    
    # Fit KMeans model
    cluster_result <- kmeans_pytorch$kmeans(
      X = embedding_tensor,
      num_clusters = as.integer(k),
      distance = 'euclidean',
      device = device,
      iter_limit = 300,
      tol = 1e-4,
      tqdm_flag = FALSE  # Disable tqdm progress bar
    )
    
    # Calculate WCSS manually
    cluster_assignments <- cluster_result[[1]]
    cluster_centers <- cluster_result[[2]]
    
    wcss[k] <- calculate_wcss(embedding_tensor, cluster_assignments, cluster_centers)
    
    # Clean up
    rm(cluster_result, cluster_assignments, cluster_centers)
    cleanup_memory()
  }
  
  # Applying the Elbow Method to find the optimal k
  diffs <- diff(wcss) / wcss[-length(wcss)]
  optimal_k <- which.min(diffs[diffs < 0.1])
  
  flog.info(paste("Optimal number of clusters determined:", optimal_k))
  
  return(optimal_k)
}

# Helper function to calculate WCSS
calculate_wcss <- function(data, assignments, centers) {
  n_clusters <- centers$shape[0]
  wcss <- 0
  for (i in 1:n_clusters) {
    cluster_points <- data[assignments == (i-1), ]  # Adjust for 0-based indexing
    center <- centers[i, ]
    wcss <- wcss + torch$sum((cluster_points - center)^2)$item()
  }
  return(wcss)
}

# run_experiments function
run_experiments <- function(data, embedding_col, year, method, param_grid, data_name, n_iterations = 5, filename = "clustering_progress.rds") {
  results <- list()
  
  for (i in 1:nrow(param_grid)) {
    params <- param_grid[i, ]
    
    # Check if this experiment has already been run
    current_progress <- if (file.exists(filename)) readRDS(filename) else list()
    key <- paste(names(params), params, sep = "_", collapse = "_")
    
    existing_iterations <- length(current_progress[[data_name]][[method]][[key]])
    
    if (existing_iterations >= n_iterations) {
      flog.info(paste("Skipping already completed experiment for", data_name, "-", method, "-", key))
      results[[length(results) + 1]] <- current_progress[[data_name]][[method]][[key]]
      next
    }
    
    flog.info(paste("Running experiment for", data_name, "-", method, "-", key))
    
    iteration_results <- current_progress[[data_name]][[method]][[key]] # Start with existing results
    
    for (iteration in (existing_iterations + 1):n_iterations) {
      flog.info(paste("Iteration", iteration, "of", n_iterations))
      
      if (method == "kmeans") {
        result <- perform_kmeans_clustering(
          data, 
          embedding_col, 
          n_clusters = params$n_clusters,
          use_umap = params$use_umap, 
          n_components = params$n_components,
          n_neighbors = params$n_neighbors,
          tol = 1e-4,
          max_iter = 300,
          year = year,
          data_name = data_name
        )
        cluster_col <- "cluster_kmeans"
      } else if (method == "hdbscan") {
        result <- perform_hdbscan_clustering(
          data, 
          embedding_col, 
          use_umap = params$use_umap, 
          n_components = params$n_components, 
          min_cluster_size = params$min_cluster_size,
          n_neighbors = params$n_neighbors,
          year = year,
          data_name = data_name
        )
        cluster_col <- "cluster"
      }
      
      quality <- calculate_cluster_quality(
        result$data,
        coherence_rating %>% filter(year == year),
        "cluster"
      )
      
      result$quality <- quality
      result$iteration <- iteration
      
      iteration_results[[iteration]] <- result
    }
    
    results[[length(results) + 1]] <- iteration_results
    save_progress(iteration_results, data_name, method, params, filename)
  }
  
  # Process results
  if (method == "kmeans" || method == "hdbscan") {
    valid_results <- Filter(function(x) !is.null(x), results)
    if (length(valid_results) > 0) {
      results_df <- do.call(rbind, lapply(valid_results, function(iteration_results) {
        do.call(rbind, lapply(iteration_results, function(x) {
          data.frame(
            min_cluster_size = if(method == "hdbscan" && !is.null(x$params$min_cluster_size)) x$params$min_cluster_size else NA,
            n_clusters = x$stats$n_clusters,
            num_noise_points = if(method == "hdbscan") x$stats$num_noise_points else 0,
            use_umap = x$stats$use_umap,
            n_components = x$stats$n_components,
            n_neighbors = x$stats$n_neighbors,
            quality_score = x$quality$quality_score,
            avg_same_cluster = x$quality$avg_same_cluster,
            avg_diff_cluster = x$quality$avg_diff_cluster,
            n_same = x$quality$n_same,
            n_diff = x$quality$n_diff,
            n_total = x$quality$n_total,
            iteration = x$iteration
          )
        }))
      }))
    } else {
      flog.warn(paste("No valid results for", data_name, "-", method))
      results_df <- NULL
    }
  } else {
    flog.warn(paste("Unknown method:", method))
    results_df <- NULL
  }
  
  return(results_df)
}

analyze_results <- function(results) {
  analyzed_data <- list()
  
  for (data_name in names(results)) {
    flog.info(paste("Analyzing results for", data_name))
    hdbscan_results <- results[[data_name]]$hdbscan
    kmeans_results <- results[[data_name]]$kmeans
    
    # Analyze HDBSCAN results
    if (!is.null(hdbscan_results) && length(hdbscan_results) > 0) {
      hdbscan_summary <- rbindlist(hdbscan_results, fill = TRUE)
      
      if (nrow(hdbscan_summary) > 0) {
        hdbscan_summary <- hdbscan_summary[, .(
          n_clusters = mean(n_clusters, na.rm = TRUE),
          num_noise_points = mean(num_noise_points, na.rm = TRUE),
          quality_score = mean(quality_score, na.rm = TRUE),
          avg_same_cluster = mean(avg_same_cluster, na.rm = TRUE),
          avg_diff_cluster = mean(avg_diff_cluster, na.rm = TRUE),
          n_same = sum(n_same, na.rm = TRUE),
          n_diff = sum(n_diff, na.rm = TRUE),
          n_total = sum(n_total, na.rm = TRUE),
          n_iterations = .N
        ), by = .(min_cluster_size, use_umap, n_components, n_neighbors)]
      } else {
        flog.warn("HDBSCAN summary is empty")
      }
    } else {
      hdbscan_summary <- NULL
      flog.info("No HDBSCAN results to analyze")
    }
    
    # Analyze K-means results
    if (!is.null(kmeans_results) && length(kmeans_results) > 0) {
      kmeans_summary <- rbindlist(kmeans_results, fill = TRUE)
      
      if (nrow(kmeans_summary) > 0) {
        kmeans_summary <- kmeans_summary[, .(
          num_noise_points = 0,
          quality_score = mean(quality_score, na.rm = TRUE),
          avg_same_cluster = mean(avg_same_cluster, na.rm = TRUE),
          avg_diff_cluster = mean(avg_diff_cluster, na.rm = TRUE),
          n_same = sum(n_same, na.rm = TRUE),
          n_diff = sum(n_diff, na.rm = TRUE),
          n_total = sum(n_total, na.rm = TRUE),
          n_iterations = .N
        ), by = .(n_clusters, use_umap, n_components, n_neighbors)]
      } else {
        flog.warn("K-means summary is empty")
      }
    } else {
      kmeans_summary <- NULL
      flog.info("No K-means results to analyze")
    }
    
    analyzed_data[[data_name]] <- list(
      hdbscan = hdbscan_summary,
      kmeans = kmeans_summary
    )
  }
  
  return(analyzed_data)
}

# Perform clustering and calculate quality for all methods
perform_analysis <- function(data, embedding_col, year, params, method, seed = 42, data_name) {
  set.seed(seed)
  filtered_data <- data %>% filter(election == year)
  
  flog.info(paste("Starting analysis for", data_name, "year", year, "method", method))
  
  if (method == "hdbscan") {
    result <- perform_hdbscan_clustering(
      filtered_data, 
      embedding_col, 
      use_umap = params$use_umap, 
      n_components = params$n_components, 
      min_cluster_size = params$min_cluster_size,
      n_neighbors = params$n_neighbors,
      year = year,
      data_name = data_name
    )
    cluster_col <- "cluster"
    num_clusters <- attr(result, "num_clusters")
    num_noise_points <- attr(result, "num_noise_points")
  } else if (method == "kmeans") {
    kmeans_result <- perform_kmeans_clustering(
      filtered_data, 
      embedding_col, 
      n_clusters = params$n_clusters,
      num_noise_points <- NA_integer_,
      use_umap = params$use_umap, 
      n_components = params$n_components,
      n_neighbors = params$n_neighbors,
      tol = 1e-4,
      max_iter = 300,
      year = year,
      data_name = data_name
    )
    
    result <- kmeans_result$data
    cluster_col <- "cluster_kmeans"
    num_clusters <- kmeans_result$stats$n_clusters
  } else {
    flog.error(paste("Unknown clustering method:", method))
    return(list(stats = NULL, quality = NULL, num_clusters = NA, params = params))
  }
  
  if (!cluster_col %in% names(result)) {
    flog.warn(paste("Clustering failed for", data_name, "year", year))
    return(list(stats = NULL, quality = NULL, num_clusters = NA, params = params))
  }
  
  stats <- calculate_cluster_stats(result, cluster_col)
  
  # Ensure 'url_rid' column exists
  if (!"url_rid" %in% names(result)) {
    flog.warn("'url_rid' column not found. Using row numbers as identifiers.")
    result$url_rid <- row_number(result)
  }
  
  quality <- calculate_cluster_quality(
    result,
    coherence_rating %>% filter(year == year),
    cluster_col
  )
  
  list(stats = stats, quality = quality, num_clusters = num_clusters, params = params)
}

# Main experimental pipeline
run_experiments <- function(data, embedding_col, year, method, param_grid, data_name, n_iterations = 5, filename = "clustering_progress.rds") {
  results <- list()
  
  for (i in 1:nrow(param_grid)) {
    params <- param_grid[i, ]
    
    # Check if this experiment has already been run
    current_progress <- if (file.exists(filename)) readRDS(filename) else list()
    key <- paste(names(params), params, sep = "_", collapse = "_")
    
    existing_iterations <- length(current_progress[[data_name]][[method]][[key]])
    
    if (existing_iterations >= n_iterations) {
      flog.info(paste("Skipping already completed experiment for", data_name, "-", method, "-", key))
      results[[length(results) + 1]] <- current_progress[[data_name]][[method]][[key]]
      next
    }
    
    flog.info(paste("Running experiment for", data_name, "-", method, "-", key))
    
    iteration_results <- vector("list", n_iterations)
    
    for (iteration in (existing_iterations + 1):n_iterations) {
      flog.info(paste("Iteration", iteration, "of", n_iterations))
      
      if (method == "kmeans") {
        result <- perform_kmeans_clustering(
          data, 
          embedding_col, 
          n_clusters = params$n_clusters,
          use_umap = params$use_umap, 
          n_components = params$n_components,
          n_neighbors = params$n_neighbors,
          tol = 1e-4,
          max_iter = 300,
          year = year,
          data_name = data_name
        )
        cluster_col <- "cluster_kmeans"
      } else if (method == "hdbscan") {
        result <- perform_hdbscan_clustering(
          data, 
          embedding_col, 
          use_umap = params$use_umap, 
          n_components = params$n_components, 
          min_cluster_size = params$min_cluster_size,
          n_neighbors = params$n_neighbors,
          year = year,
          data_name = data_name
        )
        cluster_col <- "cluster"
      }
      
      quality <- calculate_cluster_quality(
        result$data,
        coherence_rating %>% filter(year == year),
        "cluster"
      )
      
      iteration_results[[iteration]] <- data.table(
        n_clusters = as.integer(if(method == "kmeans") result$stats$n_clusters else result$stats$n_clusters),
        num_noise_points = as.integer(if(method == "hdbscan") result$stats$num_noise_points else 0),
        use_umap = as.logical(params$use_umap),
        n_components = as.integer(params$n_components),
        n_neighbors = as.integer(params$n_neighbors),
        quality_score = as.numeric(quality$quality_score),
        avg_same_cluster = as.numeric(quality$avg_same_cluster),
        avg_diff_cluster = as.numeric(quality$avg_diff_cluster),
        n_same = as.integer(quality$n_same),
        n_diff = as.integer(quality$n_diff),
        n_total = as.integer(quality$n_total),
        iteration = as.integer(iteration)
      )
      if (method == "hdbscan") {
        iteration_results[[iteration]][, min_cluster_size := as.integer(params$min_cluster_size)]
      }
    }
    
    results[[length(results) + 1]] <- rbindlist(iteration_results)
    save_progress(results[[length(results)]], data_name, method, params, filename)
  }
  
  return(rbindlist(results, fill = TRUE))
}

# Main experimental pipeline
run_pipeline <- function(data_list, n_iterations = 5, filename = "clustering_progress.rds") {
  all_results <- list()
  
  for (data_name in names(data_list)) {
    flog.info(paste("Processing dataset:", data_name))
    
    data <- data_list[[data_name]]$data
    embedding_col <- data_list[[data_name]]$col
    years <- unique(data$election)
    
    if (length(years) != 1) {
      flog.error(paste("Error: Data for", data_name, "contains multiple years. Skipping this dataset."))
      next
    }
    year <- years
    
    # KMeans experiments
    kmeans_grid <- create_parameter_grid("kmeans")
    kmeans_results <- run_experiments(data, embedding_col, year, "kmeans", kmeans_grid, data_name, n_iterations, filename)
    
    # HDBSCAN experiments
    hdbscan_grid <- create_parameter_grid("hdbscan")
    hdbscan_results <- run_experiments(data, embedding_col, year, "hdbscan", hdbscan_grid, data_name, n_iterations, filename)
    
    all_results[[data_name]] <- list(
      kmeans = kmeans_results,
      hdbscan = hdbscan_results
    )
  }
  
  return(all_results)
}

check_invalid_results <- function(results) {
  if (is.data.table(results)) {
    return(any(
      is.na(results$n_clusters) | results$n_clusters < 0 |
        is.na(results$quality_score) | results$quality_score < -3 | results$quality_score > 3
    ))
  } else {
    # For backward compatibility, keep the old check
    any(sapply(results, function(iteration) {
      if (!is.null(iteration$stats) && !is.null(iteration$quality)) {
        n_clusters <- iteration$stats$n_clusters
        quality_score <- iteration$quality$quality_score
        
        is.na(n_clusters) || is.nan(n_clusters) || n_clusters < 0 ||
          is.na(quality_score) || is.nan(quality_score) || 
          quality_score < -3 || quality_score > 3
      } else {
        TRUE  # Consider it invalid if stats or quality is missing
      }
    }))
  }
}

resume_experiment <- function(data_list, n_iterations = 5, filename = "clustering_progress.rds") {
  if (!file.exists(filename)) {
    flog.info("No previous progress found. Starting from the beginning.")
    return(run_pipeline(data_list, n_iterations, filename))
  }
  
  flog.info("Resuming experiment from saved progress.")
  previous_progress <- readRDS(filename)
  
  modified_run_pipeline <- function(data_list, n_iterations, filename) {
    all_results <- previous_progress  # Start with previous results
    
    for (data_name in names(data_list)) {
      flog.info(paste("Processing dataset:", data_name))
      
      data <- data_list[[data_name]]$data
      embedding_col <- data_list[[data_name]]$col
      year <- unique(data$election)
      
      if (length(year) != 1) {
        flog.error(paste("Error: Data for", data_name, "contains multiple years. Skipping this dataset."))
        next
      }
      
      for (method in c("kmeans", "hdbscan")) {
        param_grid <- create_parameter_grid(method)
        
        for (i in 1:nrow(param_grid)) {
          params <- param_grid[i,]
          key <- paste(names(params), params, sep = "_", collapse = "_")
          
          results <- all_results[[data_name]][[method]][[key]]
          
          should_recalculate <- is.null(results) || nrow(results) < n_iterations || 
            (nrow(results) > 0 && check_invalid_results(results))
          
          if (should_recalculate) {
            flog.info(paste("Recalculating:", data_name, method, key))
            flog.info(paste("Parameters:", 
                            paste(names(params), params, sep = "=", collapse = ", ")))
            
            new_result <- tryCatch({
              run_experiments(data, embedding_col, year, method, params, data_name, n_iterations, filename)
            }, error = function(e) {
              flog.error(paste("Error in run_experiments:", e$message))
              flog.error(paste("Traceback:", paste(capture.output(traceback()), collapse = "\n")))
              return(NULL)
            })
            
            if (!is.null(new_result)) {
              all_results[[data_name]][[method]][[key]] <- new_result
              flog.info(paste("Recalculation completed for:", data_name, method, key))
            } else {
              flog.warn(paste("Failed to recalculate:", data_name, method, key))
            }
          } else {
            flog.info(paste("Skipping recalculation for:", data_name, method, key))
          }
        }
      }
    }
    
    return(all_results)
  }
  
  # Run the modified pipeline
  all_results <- modified_run_pipeline(data_list, n_iterations, filename)
  
  # Save the updated results
  saveRDS(all_results, filename)
  
  flog.info("Experiment completed and results saved.")
  return(all_results)
}

# Helper function to perform recalculation
force_recalculate <- function(data, embedding_col, year, method, params, data_name, n_iterations) {
  results <- data.table()
  for (i in 1:n_iterations) {
    flog.info(paste("Forced recalculation - Iteration", i, "of", n_iterations))
    if (method == "kmeans") {
      result <- perform_kmeans_clustering(
        data, 
        embedding_col, 
        n_clusters = params$n_clusters,
        use_umap = params$use_umap, 
        n_components = params$n_components,
        n_neighbors = params$n_neighbors,
        tol = 1e-4,
        max_iter = 300,
        year = year,
        data_name = data_name
      )
    } else if (method == "hdbscan") {
      result <- perform_hdbscan_clustering(
        data, 
        embedding_col, 
        use_umap = params$use_umap, 
        n_components = params$n_components, 
        min_cluster_size = params$min_cluster_size,
        n_neighbors = params$n_neighbors,
        year = year,
        data_name = data_name
      )
    }
    
    if (is.null(result)) {
      flog.error("Clustering returned NULL result")
      next
    }
    
    quality <- calculate_cluster_quality(
      result$data,
      coherence_rating %>% filter(year == year),
      "cluster"
    )
    
    iteration_result <- data.table(
      n_clusters = as.integer(result$stats$n_clusters),
      num_noise_points = as.integer(if(method == "hdbscan") result$stats$num_noise_points else 0),
      use_umap = params$use_umap,
      n_components = params$n_components,
      n_neighbors = params$n_neighbors,
      quality_score = quality$quality_score,
      avg_same_cluster = quality$avg_same_cluster,
      avg_diff_cluster = quality$avg_diff_cluster,
      n_same = quality$n_same,
      n_diff = quality$n_diff,
      n_total = quality$n_total,
      iteration = i
    )
    if (method == "hdbscan") {
      iteration_result[, min_cluster_size := params$min_cluster_size]
    }
    results <- rbindlist(list(results, iteration_result), fill = TRUE)
  }
  return(results)
}

flog.info("Starting/Resuming experimental runs")
experimental_results <- resume_experiment(data_list, n_iterations = 5)
flog.info("Experimental runs completed")

flog.info("Double chek")
experimental_results <- resume_experiment(data_list, n_iterations = 5)
flog.info("Double chek completed")

flog.info("Starting result analysis")
analyzed_results <- analyze_results(experimental_results)
flog.info("Result analysis completed")

save_final_results <- function(experimental_results, analyzed_results, output_dir = "./output") {
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Move and rename clustering_progress.rds instead of deleting and resaving
  if (file.exists("clustering_progress.rds")) {
    file.rename(from = "clustering_progress.rds", 
                to = file.path(output_dir, "experimental_results.rds"))
    flog.info("Moved and renamed clustering_progress.rds to experimental_results.rds in the output directory.")
  } else {
    flog.warn("clustering_progress.rds not found. Saving experimental_results directly.")
    saveRDS(experimental_results, file.path(output_dir, "experimental_results.rds"))
  }
  
  saveRDS(analyzed_results, file.path(output_dir, "analyzed_results.rds"))
  
  # Create summary_data more efficiently without aggregating iterations
  summary_data <- rbindlist(lapply(names(experimental_results), function(data_name) {
    rbindlist(lapply(c("hdbscan", "kmeans"), function(method) {
      if (!is.null(experimental_results[[data_name]][[method]])) {
        data <- rbindlist(lapply(names(experimental_results[[data_name]][[method]]), function(key) {
          result <- experimental_results[[data_name]][[method]][[key]]
          result[, `:=`(
            model = ifelse(grepl("text", data_name), "text-embedding-3-large", "UmBERTo"),
            year = substr(data_name, nchar(data_name) - 3, nchar(data_name)),
            method = method,
            data_name = data_name,
            params = key
          )]
          return(result)
        }), use.names = TRUE, fill = TRUE)
        return(data)
      }
    }), use.names = TRUE, fill = TRUE)
  }), use.names = TRUE, fill = TRUE)
  
  fwrite(summary_data, file.path(output_dir, "clustering_summary.csv"))
  
  flog.info("All results saved successfully in the output directory.")
}

flog.info("Saving final results")
save_final_results(experimental_results, analyzed_results)
flog.info("Final results saved successfully")

flog.info("Clustering experiment script completed successfully")