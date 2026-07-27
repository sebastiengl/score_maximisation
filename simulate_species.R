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

bio_vars_terra <- c("wc2.1_10m_bio_1",  "wc2.1_10m_bio_4",
                    "wc2.1_10m_bio_7",  "wc2.1_10m_bio_12",
                    "wc2.1_10m_bio_15", "wc2.1_10m_bio_17")
env_terra  <- worldclim_terra[[bio_vars_terra]]

env_stack <- raster::stack(lapply(seq_len(terra::nlyr(env_terra)),
                                   function(i) as(env_terra[[i]], "Raster")))
bio_vars <- c("bio1", "bio4", "bio7", "bio12", "bio15", "bio17")
names(env_stack) <- bio_vars

message(sprintf("Environmental stack: %d layers, %d x %d cells",
                nlayers(env_stack), nrow(env_stack), ncol(env_stack)))

N_SPECIES <- 10
N_SITES   <- 2000
ALPHA_PA  <- -0.05

BORN_SUP <- 0.2

PATH <- sprintf("data_%d_%0.2f", N_SPECIES, BORN_SUP)
dir.create(PATH, showWarnings = FALSE)
set.seed(42)

species_prevalences <- exp(runif(N_SPECIES, log(0.01), log(BORN_SUP)))

valid_cells <- which(!is.na(values(env_stack[[1]])))
nsites <- min(N_SITES, length(valid_cells))
site_cells <- sample(valid_cells, nsites)
site_coords <- xyFromCell(env_stack, site_cells)

env_values <- raster::extract(env_stack, site_cells)

env_df <- as.data.frame(env_values)
names(env_df) <- names(env_stack)
survey_env_df <- data.frame(surveyId = seq_len(nsites), env_df, stringsAsFactors = FALSE)
write.csv(survey_env_df, file = paste(PATH, "/survey_env.csv", sep=""), row.names = FALSE)
message("Saved environmental table to survey_env.csv (surveyId + env variables)")

message("Computing PCA for all species...")

env_df_pca <- terra::spatSample(env_terra[[bio_vars_terra]], size = N_SITES, na.rm = TRUE)
names(env_df_pca) <- bio_vars

pca_object <- ade4::dudi.pca(env_df_pca, scannf = FALSE, nf = 2)

species_list <- vector("list", N_SPECIES)

message(sprintf("Simulating %d virtual species...", N_SPECIES))

for (i in seq_len(N_SPECIES)) {

  if (i %% 10 == 0) message(sprintf("  Species %d / %d", i, N_SPECIES))

  sp <- generateSpFromPCA(
    raster.stack  = env_stack,
    pca           = pca_object,
    niche.breadth = "any",
    plot          = FALSE
  )
  species_list[[i]] <- sp
}
message("All species simulated.")

message("Saving outputs: producing survey CSV...")

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

proba_cols <- paste0("sp_", seq_len(N_SPECIES) - 1)
proba_df <- data.frame(
  surveyId = seq_len(nsites)
)
for (j in seq_len(N_SPECIES)) {
  proba_df[[proba_cols[j]]] <- suitability_mat[, j]
}

write.csv(proba_df, file = paste(PATH, "/all_probas.csv", sep=""), row.names = FALSE)
message("Probabilities saved")

presence_mat <- matrix(0, nrow = nsites, ncol = N_SPECIES)
for (i in seq_len(nsites)) {
  for (j in seq_len(N_SPECIES)) {
    presence_mat[i, j] <- rbinom(1, 1, suitability_mat[i, j])
  }
}

species_strings <- apply(presence_mat, 1, function(row) {
  ids <- which(row == 1)
  if (length(ids) == 0) return("")
  paste(ids - 1, collapse = " ")
})

survey_df <- data.frame(
  surveyId  = seq_len(nsites),
  speciesId = species_strings,
  stringsAsFactors = FALSE
)

write.csv(survey_df, file = paste(PATH, "/all_species.csv", sep=""), row.names = FALSE)

message("DONE")