#!/usr/bin/env Rscript
# ============================================================
# Real-world datasets -- test residual density plots.
#
# Reads:   results/housing_residuals.npz  (via CSV export)
#          results/abalone_residuals.npz  (via CSV export)
#          OR directly from results/housing_residuals.csv
#                          results/abalone_residuals.csv
#
# NOTE: Since R cannot read .npz directly, the Python script
#       scripts/export_residuals_csv.py must be run first to
#       convert the NPZ files to CSV.
#
# Writes:  figures/housing_density_1.pdf
#          figures/abalone_density.pdf
#
# Usage:   python scripts/export_residuals_csv.py  # first
#          Rscript R/plot_realworld.R
# ============================================================

source("R/theme.R")
set_jasa_theme()

# ------------------------------------------------------------------
# Helper: density plot for one dataset
# ------------------------------------------------------------------
plot_residual_density <- function(csv_path, dataset_label, save_path) {

  df <- read_csv(csv_path, show_col_types = FALSE)

  df_long <- df %>%
    pivot_longer(cols = everything(), names_to = "Method", values_to = "Residual") %>%
    mutate(Method = factor(Method,
                           levels = c("OLS", "Huber", "Rho"),
                           labels = c("OLS", "Huber", TeX("$\\tilde{\\rho}$-posterior"))))

  p <- ggplot(df_long, aes(x = Residual, fill = Method, colour = Method)) +
    geom_density(alpha = 0.3, linewidth = 0.8) +
    scale_fill_grey(start = 0.8, end = 0.3) +
    scale_colour_grey(start = 0.5, end = 0.1) +
    coord_cartesian(xlim = quantile(df_long$Residual, c(0.01, 0.99))) +
    labs(x = "Test residual", y = "Density", fill = "", colour = "") +
    theme(legend.position = "bottom")

  ggsave(save_path, p, width = 6, height = 4.5)
  cat("Saved", save_path, "\n")
}

# ------------------------------------------------------------------
# Generate plots
# ------------------------------------------------------------------

if (file.exists("results/housing_residuals.csv")) {
  plot_residual_density(
    "results/housing_residuals.csv",
    "Ames Housing",
    "figures/housing_density_1.pdf"
  )
} else {
  cat("Skipping housing: results/housing_residuals.csv not found.\n")
  cat("Run: python scripts/export_residuals_csv.py\n")
}

if (file.exists("results/abalone_residuals.csv")) {
  plot_residual_density(
    "results/abalone_residuals.csv",
    "Abalone",
    "figures/abalone_density.pdf"
  )
} else {
  cat("Skipping abalone: results/abalone_residuals.csv not found.\n")
  cat("Run: python scripts/export_residuals_csv.py\n")
}
