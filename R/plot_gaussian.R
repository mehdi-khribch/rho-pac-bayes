#!/usr/bin/env Rscript
# ============================================================
# Gaussian location model -- publication figures.
#
# Reads:   results/gaussian_summary.csv, results/gaussian_trials.csv
# Writes:  figures/posterior_risk_gaussian.pdf
#          figures/rmse_gaussian.pdf
#          figures/density_plot_gaussian.pdf
#
# Usage:   Rscript R/plot_gaussian.R
# ============================================================

source("R/theme.R")
set_jasa_theme()

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
df_summary <- read_csv("results/gaussian_summary.csv", show_col_types = FALSE)
df_trials  <- read_csv("results/gaussian_trials.csv", show_col_types = FALSE)

tau_ref <- 0.5
n_ref   <- 200

df_sub <- df_summary %>%
  filter(tau == tau_ref, n == n_ref) %>%
  arrange(epsilon) %>%
  mutate(epsilon_pct = 100 * epsilon)

# ------------------------------------------------------------------
# 1. Posterior risk vs epsilon
# ------------------------------------------------------------------
df_risk <- df_sub %>%
  select(epsilon_pct, MLE = MLE_BayesRisk, Bayes = Bayes_BayesRisk, Rho = Rho_BayesRisk) %>%
  pivot_longer(cols = c(MLE, Bayes, Rho), names_to = "Method", values_to = "Risk") %>%
  mutate(Method = factor(Method, levels = c("MLE", "Bayes", "Rho")))

p_risk <- ggplot(df_risk, aes(x = epsilon_pct, y = Risk, colour = Method, linetype = Method)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2.5) +
  scale_method_colour() +
  scale_method_linetype() +
  labs(x = TeX("Contamination rate $\\varepsilon$ (%)"),
       y = "Posterior risk",
       colour = "", linetype = "") +
  theme(legend.position = "bottom")

ggsave("figures/posterior_risk_gaussian.pdf", p_risk, width = 5.5, height = 4.5)
cat("Saved figures/posterior_risk_gaussian.pdf\n")

# ------------------------------------------------------------------
# 2. RMSE vs epsilon
# ------------------------------------------------------------------
df_rmse <- df_sub %>%
  select(epsilon_pct, MLE = MLE_RMSE, Bayes = Bayes_RMSE, Rho = Rho_RMSE) %>%
  pivot_longer(cols = c(MLE, Bayes, Rho), names_to = "Method", values_to = "RMSE") %>%
  mutate(Method = factor(Method, levels = c("MLE", "Bayes", "Rho")))

p_rmse <- ggplot(df_rmse, aes(x = epsilon_pct, y = RMSE, colour = Method, linetype = Method)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2.5) +
  scale_method_colour() +
  scale_method_linetype() +
  labs(x = TeX("Contamination rate $\\varepsilon$ (%)"),
       y = "RMSE",
       colour = "", linetype = "") +
  theme(legend.position = "bottom")

ggsave("figures/rmse_gaussian.pdf", p_rmse, width = 5.5, height = 4.5)
cat("Saved figures/rmse_gaussian.pdf\n")

# ------------------------------------------------------------------
# 3. Density plot at epsilon = 10%
# ------------------------------------------------------------------
eps_target <- 0.10
prior_std  <- 2.0

row0 <- df_trials %>%
  filter(epsilon == eps_target, tau == tau_ref, n == n_ref) %>%
  slice(1)

theta_mle   <- row0$theta_mle
theta_bayes <- row0$theta_bayes_mean
theta_rho   <- row0$theta_rho_mean
std_bayes   <- row0$std_bayes
std_rho     <- row0$std_rho

theta_grid <- seq(-3, max(theta_mle, 3) + 1, length.out = 400)

df_dens <- tibble(
  theta   = rep(theta_grid, 3),
  density = c(
    dnorm(theta_grid, mean = 0, sd = prior_std),
    dnorm(theta_grid, mean = theta_bayes, sd = std_bayes),
    dnorm(theta_grid, mean = theta_rho, sd = std_rho)
  ),
  Component = factor(rep(c("Prior", "Bayes", "Rho"), each = length(theta_grid)),
                     levels = c("Prior", "Bayes", "Rho"))
)

p_dens <- ggplot(df_dens, aes(x = theta, y = density, fill = Component)) +
  geom_area(alpha = 0.35, colour = NA) +
  geom_line(aes(colour = Component), linewidth = 1.0) +
  scale_fill_grey(
    start = 0.85, end = 0.4,
    labels = c(TeX("$\\pi(\\theta)$"),
               TeX("$\\pi(\\theta | x)$"),
               TeX("$\\rho_\\psi(\\theta | x)$"))
  ) +
  scale_colour_grey(start = 0.4, end = 0.1, guide = "none") +
  geom_vline(xintercept = 0.0, linetype = "solid", linewidth = 1.0, colour = "red") +
  geom_vline(xintercept = theta_mle, linetype = "dashed", linewidth = 0.9, colour = "black") +
  geom_vline(xintercept = theta_rho, linetype = "dotted", linewidth = 0.9, colour = "black") +
  labs(x = TeX("$\\theta$"), y = "Density", fill = NULL) +
  theme(legend.position = "bottom")

ggsave("figures/density_plot_gaussian.pdf", p_dens, width = 5.5, height = 4.5)
cat("Saved figures/density_plot_gaussian.pdf\n")
