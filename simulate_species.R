# ---- 0. Dependencies ----
required_pkgs <- c("raster", "virtualspecies", "dplyr", "geodata", "ggplot2", "sf")
to_install <- required_pkgs[!sapply(required_pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install) > 0) install.packages(to_install, repos = "https://cloud.r-project.org")

library(raster)
library(virtualspecies)
library(dplyr)
library(geodata)
library(terra) 
library(ggplot2)
library(sf)


set.seed(42)  


message("Downloading / loading WorldClim data...")
worldclim_terra <- geodata::worldclim_global(var = "bio", res = 10, path = tempdir())

# Subset des 6 variables
bio_vars_terra <- c("wc2.1_10m_bio_1",  "wc2.1_10m_bio_4",
                    "wc2.1_10m_bio_7",  "wc2.1_10m_bio_12",
                    "wc2.1_10m_bio_15", "wc2.1_10m_bio_17")
env_terra  <- worldclim_terra[[bio_vars_terra]]

env_stack <- raster::stack(lapply(seq_len(terra::nlyr(env_terra)),
                                   function(i) as(env_terra[[i]], "Raster")))
bio_vars <- c("bio1", "bio4", "bio7", "bio12", "bio15", "bio17")
names(env_stack) <- bio_vars

# bio1  = Annual Mean Temperature
# bio4  = Temperature Seasonality
# bio7  = Temperature Annual Range
# bio12 = Annual Precipitation
# bio15 = Precipitation Seasonality
# bio17 = Precipitation of Driest Quarter

message(sprintf("Environmental stack: %d layers, %d x %d cells",
                nlayers(env_stack), nrow(env_stack), ncol(env_stack)))



# ---- 2. Simulation parameters ----

N_SPECIES    <- 100
N_SITES      <- 2000 
ALPHA_PA      <- -0.05        # slope for logistic PA conversion 


# Spatial split train/test colors
COLOR_TRAIN  <- "#2166AC"   # blue
COLOR_TEST   <- "#B2182B"   # red
TRAIN_RATIO  <- 0.83
BORN_SUP <- 0.5


PATH <- sprintf("data_%d_%0.2f",N_SPECIES, BORN_SUP)
dir.create(PATH, showWarnings = FALSE)
set.seed(42)

species_prevalences <- exp(runif(N_SPECIES, log(0.01), log(0.80)))

valid_cells <- which(!is.na(values(env_stack[[1]])))
nsites <- min(N_SITES, length(valid_cells))
site_cells <- sample(valid_cells, nsites)
site_coords <- xyFromCell(env_stack, site_cells)

# Extract raster values at the sampled cells and save
env_values <- raster::extract(env_stack, site_cells)

# Coerce to data.frame and ensure layer names
env_df <- as.data.frame(env_values)
names(env_df) <- names(env_stack)
survey_env_df <- data.frame(surveyId = seq_len(nsites), env_df, stringsAsFactors = FALSE)
write.csv(survey_env_df, file = paste(PATH, "/survey_env.csv", sep=""), row.names = FALSE)
message("Saved environmental table to survey_env.csv (surveyId + env variables)")


# ---- Spatial train/test split using  patches (5:1 ratio) ----
message("\nPerforming spatial block-out split")


x_range <- range(site_coords[, 1])
y_range <- range(site_coords[, 2])

N_PATCHES_X <- 12
N_PATCHES_Y <- 12
N_PATCHES_TOTAL <- N_PATCHES_X * N_PATCHES_Y 

# Create patch boundaries
x_breaks <- seq(x_range[1], x_range[2], length.out = N_PATCHES_X + 1)
y_breaks <- seq(y_range[1], y_range[2], length.out = N_PATCHES_Y + 1)

# Assign each site to a patch
x_patch <- cut(site_coords[, 1], breaks = x_breaks, labels = FALSE, include.lowest = TRUE)
y_patch <- cut(site_coords[, 2], breaks = y_breaks, labels = FALSE, include.lowest = TRUE)
patch_id <- (y_patch - 1) * N_PATCHES_X + x_patch  # row-major ordering


N_TEST_PATCHES <- N_PATCHES_TOTAL / 6
N_TRAIN_PATCHES <- N_PATCHES_TOTAL - N_TEST_PATCHES

# Randomly select patches for test
test_patch_ids <- sample(seq_len(N_PATCHES_TOTAL), N_TEST_PATCHES, replace = FALSE)
train_patch_ids <- setdiff(seq_len(N_PATCHES_TOTAL), test_patch_ids)

# Extract site indices for each split
test_indices <- which(patch_id %in% test_patch_ids)
train_indices <- which(patch_id %in% train_patch_ids)


# ---- Create and save spatial split map ----
message("Creating spatial patch-based split map...")

# Create a data frame for plotting with split assignments
plot_df <- data.frame(
  x = site_coords[, 1],
  y = site_coords[, 2],
  split = ifelse(seq_len(nsites) %in% train_indices, "Train", "Test")
)

# Get world coastlines from rnaturalearth
world_sf <- rnaturalearth::ne_coastline(scale = 10, returnclass = "sf")

# Create the map
p <- ggplot() +
  # White background
  theme(panel.background = element_rect(fill = "white", colour = NA),
        plot.background  = element_rect(fill = "white", colour = NA)) +
  # Coastlines in black
  geom_sf(data = world_sf, color = "black", fill = NA, size = 0.3) +
  # Plot sample sites
  geom_point(data = plot_df[plot_df$split == "Train", ],
             aes(x = x, y = y, color = "Train"),
             size = 2.5, alpha = 0.7) +
  geom_point(data = plot_df[plot_df$split == "Test", ],
             aes(x = x, y = y, color = "Test"),
             size = 2.5, alpha = 0.7) +
  # Manual color scale
  scale_color_manual(
    name = "Split",
    values = c("Train" = COLOR_TRAIN, "Test" = COLOR_TEST)
  ) +
  coord_sf() +
  labs(
    title = "Spatial Patch-Based Split",
    x = "Longitude",
    y = "Latitude"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottomright",
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
  )

# Save map as PNG
ggsave(paste(PATH, "/split_map.png", sep=""), plot = p, width = 12, height = 8, dpi = 300)


# ---- Compute the PCA on data) ----
message("Computing PCA for all species...")

env_df_pca <- terra::spatSample(env_terra[[bio_vars_terra]], size = N_SITES, na.rm = TRUE)
names(env_df_pca) <- bio_vars   # match names(env_stack), required by generateSpFromPCA

pca_object <- ade4::dudi.pca(env_df_pca, scannf = FALSE, nf = 2)  # nf = max(axes), axes defaults to c(1,2)
# ---- Simulate virtual species ----

species_list   <- vector("list", N_SPECIES)

message(sprintf("Simulating %d virtual species...", N_SPECIES))

for (i in seq_len(N_SPECIES)) {

  if (i %% 10 == 0) message(sprintf("  Species %d / %d", i, N_SPECIES))

  # --- Générer la suitabilité via PCA ---
  sp <- generateSpFromPCA(
    raster.stack  = env_stack,
    pca           = pca_object,
    niche.breadth = "any",
    plot          = FALSE
  )
  species_list[[i]] <- sp
}
message("All species simulated.")


# # ---- Save outputs ----
message("Saving outputs: producing survey CSV...")


# ---- Extract probability for all species ----
suitability_mat <- do.call(cbind, lapply(species_list, function(sp) {
  values <- raster::extract(sp$suitab.raster, site_cells)
  if (is.list(values)) {
    values <- unlist(values, use.names = FALSE)
  }
  if (is.matrix(values)) {
    values <- values[, 1]
  }
  as.numeric(values)
}))

# Match each species' average probability to the chosen prevalence target.
# We shift probabilities in logit space so the result stays in (0, 1) and
# the column mean matches species_prevalences as closely as numerically possible.
normalize_probabilities_to_prevalence <- function(probabilities, target_prevalence) {
  probabilities <- pmin(pmax(probabilities, 1e-6), 1 - 1e-6)

  objective <- function(offset) {
    mean(plogis(qlogis(probabilities) + offset)) - target_prevalence
  }

  lower <- -50
  upper <- 50
  while (objective(lower) > 0) lower <- lower - 10
  while (objective(upper) < 0) upper <- upper + 10

  offset <- uniroot(objective, lower = lower, upper = upper, tol = 1e-8)$root
  plogis(qlogis(probabilities) + offset)
}

for (j in seq_len(N_SPECIES)) {
  suitability_mat[, j] <- normalize_probabilities_to_prevalence(
    suitability_mat[, j],
    species_prevalences[j]
  )
}

# Save probabilities as CSV, split into train/test like simu_model.py
proba_cols <- paste0("sp_", seq_len(N_SPECIES) - 1)
proba_df <- data.frame(
  surveyId = seq_len(nsites)
)
for (j in seq_len(N_SPECIES)) {
  proba_df[[proba_cols[j]]] <- suitability_mat[, j]
}

proba_train <- proba_df[train_indices, ]
proba_test  <- proba_df[test_indices, ]




write.csv(proba_train, file = paste(PATH ,"/CS4_train_probas.csv", sep=""), row.names = FALSE)
write.csv(proba_test,  file = paste(PATH ,"/CS4_test_probas.csv", sep=""),  row.names = FALSE)
message("Probabilities saved")

# ---- Generate presence-absence by Bernoulli sampling ----

presence_mat <- matrix(0, nrow = nsites, ncol = N_SPECIES)
for (i in seq_len(nsites)) {
  for (j in seq_len(N_SPECIES)) {
    presence_mat[i, j] <- rbinom(1, 1, suitability_mat[i, j])
  }
}

# For each site, collapse observed species ids into a single space-separated string
species_strings <- apply(presence_mat, 1, function(row) {
  ids <- which(row == 1)
  if (length(ids) == 0) return("")
  paste(ids-1, collapse = " ")
})

survey_df <- data.frame(
  surveyId = seq_len(nsites),
  speciesId   = species_strings,
  stringsAsFactors = FALSE
)


# Create train/test dataframes
survey_train <- survey_df[train_indices, ]
survey_test  <- survey_df[test_indices, ]

# Write split CSVs
write.csv(survey_train, file = paste(PATH ,"/CS4_train_species.csv", sep=""), row.names = FALSE)
write.csv(survey_test,  file = paste(PATH ,"/CS4_test_species.csv", sep=""),  row.names = FALSE)


message("DONE")

