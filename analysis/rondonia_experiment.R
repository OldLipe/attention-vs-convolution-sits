#  ---- Import packages ----
setwd("~/pixel-vs-object-sits")
#
# Import SITS package to produce classifications
#
library(sits)

#
# Import terra package to work with raster data
#
library(terra)

#
# Import sf package to work with vector data
#
library(sf)

#
# Set seed to ensure reproducibility
#
set.seed(123)

print(torch::cuda_is_available())

#  ---- Define general functions ----

#
# Function for visualization tuning history
#
optimization_history <- function(tune) {
  acc <- sort(tune$accuracy, decreasing = FALSE)
  acc <- tibble::tibble(
    acc = acc,
    trial = seq_len(length(acc))
  )
  
  p <- ggplot2::ggplot(acc, ggplot2::aes(
    x = .data[["trial"]],
    y = .data[["acc"]]
  ))
  
  p <- p + 
    ggplot2::geom_smooth(
      formula = y ~ x,
      se      = FALSE,
      method  = "loess",
      na.rm   = TRUE
    )
  
  p <- p +
    ggplot2::theme(
      strip.placement = "outside",
      strip.text = ggplot2::element_text(
        colour = "black",
        size   = 11
      ),
      strip.background = ggplot2::element_rect(
        fill  = NA,
        color = NA
      )
    )
  
  p <- p + ggplot2::labs(x = "Trial", y = "Objective Value")
  p
}

#
# Define color palette
#
class_color <- tibble::tibble(name = character(), color = character())
class_color <- class_color |>
  tibble::add_row(name = "Bare_Soil", color = "#D7C49C") |>
  tibble::add_row(name = "Burned_Areas", color = "#EC7063") |>
  tibble::add_row(name = "Forest2", color = "#00B29E") |>
  tibble::add_row(name = "Forests", color = "#1E8449") |>
  tibble::add_row(name = "forest", color = "#1E8449") |>
  tibble::add_row(name = "forests", color = "#1E8449") |>
  tibble::add_row(name = "Forests4", color = "#229C59") |>
  tibble::add_row(name = "Highly_Degraded", color = "#BFD9BD") |>
  tibble::add_row(name = "Water", color = "#2980B9") |>
  tibble::add_row(name = "water", color = "#2980B9") |>
  tibble::add_row(name = "Wetlands", color = "#A0B9C8") |>
  tibble::add_row(name = "wetlands2", color = "#7CF4FA")


#
# Load the color table into `sits`
#
sits_colors_set(colors = class_color, legend = "class_color")

# ---- Create data cube ----

#
# Cube bands
#
cube_bands <- c(
  "B02", "B03", "B04", "B05", "B06", "B07", 
  "B08", "B8A", "B09", "B11", "B12", "CLOUD"
)

#
# Total of workers available
#
multicores <- 24

#
# Define tiles
#
tiles <- sf::st_read("./data/raw/RO/tiles/RO_TILES_SM2.gpkg", quiet = TRUE)

#
# Define dates
#
start_date <- "2022-01-01"
end_date <- "2022-12-31"


lapply(tiles[["tile"]], function(tile) {
  print(tile)
  #
  # Create Sentinel-2 data cube
  #
  cube <- sits_cube(
    source     = "BDC",
    collection = "SENTINEL-2-16D",
    tiles      = tile,
    start_date = start_date,
    end_date   = end_date,
    bands      = cube_bands
  )

  cube <- sits_cube_copy(
    cube = cube,
    multicores = 24,
    output_dir = "./data/raw/RO/cube/"
  )
  
  #
  # Read trained model
  #
  tcnn_model <- readRDS("./data/output/RO/model/tcnn_model.rds")
  
  #
  # Read trained model
  #
  lighttae_model <- readRDS("./data/output/RO/model/ltae_model.rds")
  
  # ---- Pixel-based classification - TCNN ----
  #
  # Define output directory
  #
  output_dir <- "./data/output/RO/classifications/tccn"
  
  #
  # Define version name
  #
  results_version <- "tcnn-8cls-sentinel-2"
  
  roi <- sf::st_read("./data/raw/RO/roi/rondonia.gpkg")
  
  #
  # Classify data cube
  #
  probs_cube <- sits_classify(
    data       = cube,
    ml_model   = tcnn_model,
    roi = roi,
    memsize    = 80,
    gpu_memory = 10,
    multicores = 24,
    output_dir = output_dir,
    version    = results_version,
    progress   = TRUE
  )
  
  output_dir <- "./data/output/RO/classifications/lighttae"
  
  #
  # Define version name
  #
  results_version <- "lighttaeb-8cls-sentinel-2"
  
  #
  # Classify data cube
  #
  probs_cube <- sits_classify(
    data       = cube,
    ml_model   = lighttae_model,
    roi = roi,
    memsize    = 80,
    gpu_memory = 10,
    multicores = 24,
    output_dir = output_dir,
    version    = results_version
  )
  
  #
  # Define output directory
  #
  segment_dir <- "./data/output/RO/segment/"
  
  #
  # Apply spatio-temporal segmentation in Sentinel-2 cube 
  #
  segments <- sits_segment(
    cube = cube,
    seg_fn = sits_slic(
      step = 20,
      compactness = 1,
      dist_fun = "euclidean",
      iter = 20,
      minarea = 20
    ),
    roi = roi,
    output_dir = segment_dir,
    memsize    = 30,
    multicores = 12
  )
  
  # ---- Object-based classification - TCNN ----
  #
  # Define output directory
  #
  output_dir <- "./data/output/RO/segment/tcnn"
  
  #
  # Define version name
  #
  results_version <- "tcnn-8cls-segments-sentinel-2"
  
  #
  # Classify object-based data cube
  #
  probs_cube <- sits_classify(
    data       = segments,
    ml_model   = tcnn_model,
    n_sam_pol  = 40,
    memsize    = 8,
    gpu_memory = 10,
    multicores = 24,
    output_dir = output_dir,
    version    = results_version,
    progress   = FALSE,
    verbose    = FALSE
  )
  #
  # Generate map
  #
  class_cube <- sits_label_classification(
    cube       = probs_cube,
    memsize    = 30,
    multicores = 12,
    output_dir = output_dir,
    version    = results_version
  )
  
  # ---- Object-based classification - LightTAE ----
  #
  # Define output directory
  #
  output_dir <- "./data/output/RO/segment/ltae"
  
  #
  # Define version name
  #
  results_version <- "ltae-8cls-segments-sentinel-2"
  
  #
  # Classify object-based data cube
  #
  probs_cube <- sits_classify(
    data       = segments,
    ml_model   = lighttae_model,
    n_sam_pol  = 40,
    memsize    = 8,
    gpu_memory = 10,
    multicores = 24,
    output_dir = output_dir,
    version    = results_version
  )
  
  #
  # Generate map
  #
  class_cube <- sits_label_classification(
    cube       = probs_cube,
    memsize    = 30,
    multicores = 12,
    output_dir = output_dir,
    version    = results_version
  )
})


# 
# roi <- 
# 
# cube <- sits_cube_copy(
#   cube  = cube, 
#   multicores = 24,
#   output_dir = "~/pixel-vs-object-sits/data/raw/RO/cube/",
#   roi = 
# )

# ---- Read samples ----
#
# Define samples path
#
samples_file <- "./data/raw/RO/samples/samples_sentinel2_ro.rds"

#
# Load samples
#
samples <- readRDS(samples_file)

#
# View samples patterns
#
# options(repr.plot.width = 10, repr.plot.height = 7)
# plot(sits_patterns(sits_select(samples, bands = c("B08", "B12"))))


# ---- Tune TempCNN model  ----
#
# Tune tempCNN model hiperparameters
#
# tuned_tempcnn <- sits_tuning(
#   samples   = samples,
#   ml_method = sits_tempcnn(),
#   params        = sits_tuning_hparams(
#     optimizer   = torch::optim_adamw,
#     cnn_kernels = choice(c(3, 3, 3), c(5, 5, 5), c(7, 7, 7)),
#     cnn_layers  = choice(c(2^5, 2^5, 2^5), c(2^6, 2^6, 2^6), c(2^7, 2^7, 2^7)),
#     opt_hparams = list(
#       lr = loguniform(10^-2, 10^-4)
#     )
#   ),
#   trials     = 50,
#   multicores = 20,
#   progress   = TRUE
# )

# 
# Define tuned path
#
# tuning_dir <- "./data/output/RO/tune/tempcnn/"
# dir.create(tuning_dir, recursive = TRUE, showWarnings = FALSE)

# 
# Save tuned results
#
# saveRDS(tuned_tempcnn, paste0(tuning_dir, "tempcnn_ro.rds"))

# 
# Load tuned results
#
#tuned_tempcnn <- readRDS("./data/output/RO/tune/tempcnn/tempcnn_ro.rds")

# 
# View best tuned params
#
# print(tuned_tempcnn)

#
# Optimization history plot
#
# options(repr.plot.width = 8, repr.plot.height = 7)
# optimization_history(tuned_tempcnn)

# ---- Train TempCNN model ----
#
# Train tempCNN model with best hiperparameters found
#
# tcnn_model <- sits_train(
#   samples, sits_tempcnn(
#     cnn_layers = c(2^6, 2^6, 2^6),
#     cnn_kernels = c(3, 3, 3),
#     cnn_dropout_rates = c(0.2, 0.2, 0.2),
#     dense_layer_nodes = 256,
#     dense_layer_dropout_rate = 0.5,
#     epochs = 150,
#     batch_size = 64,
#     optimizer = torch::optim_adamw,
#     opt_hparams = list(lr = 0.000924),
#     patience = 20,
#     min_delta = 0.01,
#     verbose = FALSE
#   )
# )

#
# Define output directory
#
# base_model_dir <- "./data/output/RO/model/"
# tcnn_dir <- paste0(base_model_dir, "tcnn_model.rds")  

#
# Create directory
#
# dir.create(base_model_dir, recursive = TRUE, showWarnings = FALSE)

#
# Save best model
#
# saveRDS(tcnn_model, tcnn_dir)

#
# Read trained model
#
tcnn_model <- readRDS("./data/output/RO/model/tcnn_model.rds")

#
# Accuracy and validation curves
#
# options(repr.plot.width = 8, repr.plot.height = 7)
# plot(tcnn_model)

# ---- Tune LightTAE model ----
#
# Tune lightttae model 
#
# tuned_lighttae <- sits_tuning(
#   samples   = samples,
#   ml_method = sits_lighttae(),
#   params        = sits_tuning_hparams(
#     optimizer   = torch::optim_adamw,
#     opt_hparams = list(
#       lr           = loguniform(10^-2, 10^-4),
#       weight_decay = loguniform(10^-2, 10^-8)
#     )
#   ),
#   trials     = 50,
#   multicores = 20,
#   progress   = TRUE
# )

# 
# Define output directory
#
tuning_dir <- "./data/output/RO/tune/lighttae/"

# 
# Create directory
#
# dir.create(tuning_dir, recursive = TRUE, showWarnings = FALSE)

# 
# Save model
#
# saveRDS(tuned_lighttae, paste0(tuning_dir, "lighttae_ro.rds"))

#
# Read tuned parameters
#
# tuned_lighttae <- readRDS("./data/output/RO/tune/lighttae/lighttae_ro.rds")

# 
# View best tuned params
#
#  print(tuned_lighttae)

#
# Optimization history plot
#
# options(repr.plot.width = 8, repr.plot.height = 7)
# optimization_history(tuned_lighttae)


# ---- Train LightTAE model ----

#
# Train LightTAE model with best hiperparameters found
#
# lighttae_model <- sits_train(
#   samples, sits_lighttae(
#     epochs = 150,
#     batch_size = 128,
#     optimizer = torch::optim_adamw,
#     opt_hparams = list(lr = 0.00131, weight_decay = 0.0000000634),
#     lr_decay_epochs = 50L,
#     patience = 20L,
#     min_delta = 0.01,
#     verbose = FALSE
#   )
# )

#
# Define output directory
# #
# base_model_dir <- "./data/output/RO/model/"
# ltae_dir <- paste0(base_model_dir, "ltae_model.rds")  
# 
# #
# # Save best model
# #
# saveRDS(lighttae_model, ltae_dir)

#
# Read trained model
#
lighttae_model <- readRDS("./data/output/RO/model/ltae_model.rds")

#
# Accuracy and validation curves
#
# options(repr.plot.width = 8, repr.plot.height = 7)
# plot(lighttae_model)

# ---- Pixel-based classification - TCNN ----
#
# Define output directory
#
output_dir <- "./data/output/RO/classifications/tccn"

#
# Define version name
#
results_version <- "tcnn-8cls-sentinel-2"

#
# Create directory
#
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

roi <- sf::st_read("./data/raw/RO/roi/rondonia.gpkg")

#
# Classify data cube
#
probs_cube <- sits_classify(
  data       = cube,
  ml_model   = tcnn_model,
  roi = roi,
  memsize    = 80,
  gpu_memory = 10,
  multicores = 24,
  output_dir = output_dir,
  version    = results_version,
  progress   = TRUE
)

#
# Apply spatial smooth
#
probs_bayes <- sits_smooth(
  cube           = probs_cube,
  window_size    = 9,
  neigh_fraction = 0.5,
  smoothness     = c(10, 10, 10, 10, 10, 10, 10, 10, 10),
  memsize        = 60,
  multicores     = 24,
  output_dir     = output_dir,
  version        = results_version
)

#
# Generate map
#
class_cube <- sits_label_classification(
  cube       = probs_bayes,
  memsize    = 60,
  multicores = 24,
  output_dir = output_dir,
  version    = results_version
)

#
# View the classified tile
#
plot(class_cube)

# ---- Pixel-based classification - LightTAE ----
#
# Define output directory
#
output_dir <- "./data/output/RO/classifications/lighttae"

#
# Define version name
#
results_version <- "lighttaeb-8cls-sentinel-2"

#
# Create output directory
#
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

#
# Classify data cube
#
probs_cube <- sits_classify(
  data       = cube,
  ml_model   = lighttae_model,
  roi = roi,
  memsize    = 54,
  gpu_memory = 10,
  multicores = 24,
  output_dir = output_dir,
  version    = results_version
)

#
# Apply spatial smooth
#
probs_bayes <- sits_smooth(
  cube           = probs_cube,
  window_size    = 9,
  neigh_fraction = 0.5,
  smoothness     = c(10, 10, 10, 10, 10, 10, 10, 10, 10),
  memsize        = 60,
  multicores     = 24,
  output_dir     = output_dir,
  version        = results_version
)

#
# Generate map
#
class_cube <- sits_label_classification(
  cube       = probs_bayes,
  memsize    = 60,
  multicores = 24,
  output_dir = output_dir,
  version    = results_version
)

#
# View the classified tile
#
plot(class_cube)

# ---- Apply Spatial-temporal segmentation ----

#
# Define output directory
#
segment_dir <- "./data/output/RO/segment/"

#
# Apply spatio-temporal segmentation in Sentinel-2 cube 
#
segments <- sits_segment(
  cube = cube,
  seg_fn = sits_slic(
    step = 20,
    compactness = 1,
    dist_fun = "euclidean",
    iter = 20,
    minarea = 20
  ),
  output_dir = segment_dir,
  memsize    = 30,
  multicores = 12
)

# ---- Object-based classification - TCNN ----
#
# Define output directory
#
output_dir <- "./data/output/RO/segment/tcnn"

#
# Define version name
#
results_version <- "tcnn-8cls-segments"

#
# Create directory
#
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

#
# Classify object-based data cube
#
probs_cube <- sits_classify(
  data       = segments,
  ml_model   = tcnn_model,
  n_sam_pol  = 40,
  memsize    = 8,
  gpu_memory = 10,
  multicores = 24,
  output_dir = output_dir,
  version    = results_version,
  progress   = FALSE,
  verbose    = FALSE
)

#
# Generate map
#
class_cube <- sits_label_classification(
  cube       = probs_cube,
  memsize    = 30,
  multicores = 12,
  output_dir = output_dir,
  version    = results_version
)

# ---- Object-based classification - LightTAE ----
#
# Define output directory
#
output_dir <- "../data/output/RO/segment/ltae"

#
# Define version name
#
results_version <- "ltae-8cls-segments"

#
# Create directory
#
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

#
# Classify object-based data cube
#
probs_cube <- sits_classify(
  data       = segments,
  ml_model   = lighttae_model,
  n_sam_pol  = 40,
  memsize    = 8,
  gpu_memory = 10,
  multicores = 24,
  output_dir = output_dir,
  version    = results_version
)

#
# Generate map
#
class_cube <- sits_label_classification(
  cube       = probs_cube,
  memsize    = 30,
  multicores = 12,
  output_dir = output_dir,
  version    = results_version
)

# ---- Read validation samples ----
#
# Read validation samples
#
samples_val <- sf::st_read("../data/raw/RO/validation/validation_pts.gpkg", quiet = TRUE)

#
# Adjust labels in validation samples
#
samples_val <- samples_val |>
  dplyr::mutate(
    label = dplyr::case_when(
      class %in% c("CR_QM", "CR_SE", "CR_VG")  ~ "deforestation",
      class %in% c("For", "MSFor")  ~ "forest",
      class %in% c("water") ~ "water"
    )
  )

#
# View the first 10 samples
#
print(samples_val)


# ---- Validation pixel-based - TCNN ----

# ---- Train TempCNN model ----

# ---- Train TempCNN model ----

# ---- Train TempCNN model ----

# ---- Train TempCNN model ----

# ---- Train TempCNN model ----

# ---- Train TempCNN model ----

