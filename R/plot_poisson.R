#!/usr/bin/env Rscript
# ============================================================
# Poisson intensity model -- publication figures.
#
# Reads:   results/poisson_summary.csv, results/poisson_trials.csv
# Writes:  figures/posterior_risk_pois.pdf
#          figures/rmse_pois.pdf
#          figures/density_plot_pois.pdf
#
# Usage:   Rscript R/plot_poisson.R
# ============================================================

source("R/theme.R")
set_jasa_theme()

df_summary <- read_csv("results/poisson_summary.csv", show_col_types = FALSE)
df_trials  <- read_csv("results/poisson_trials.csv", show_col_types = FALSE)

tau_ref <- 0.5
n_ref   <- 200

df_sub <- df_summary %>%
  filter(tau == tau_ref, n == n_ref) %>%
  arrange(epsilon) %>%
  mutate(epsilon_pct = 100 * epsilon)

# ------------------------------------------------------------------
# 1. Posterior risk
# ------------------------------------------------------------------
df_risk <- df_sub %>%
  select(epsilon_pct, MLE = MLE_BayesRisk, Bayes = Bayes_BayesRisk, Rho = Rho_BayesRisk) %>%
  pivot_longer(cols = c(MLE, Bayes, Rho), names_to = "Method", values_to = "Risk") %>%
  mutate(Method = factor(Method, levels = c("MLE", "Bayes", "Rho")))

p_risk <- ggplot(df_risk, aes(x = epsilon_pct, y = Risk, colour = Method, linetype = Method)) +
  geom_line(linewidth = 1.1) + geom_point(size = 2.5) +
  scale_method_colour() + scale_method_linetype() +
  labs(x = TeX("Contamination rate $\\varepsilon$ (%)"),
       y = "Posterior risk", colour = "", linetype = "") +
  theme(legend.position = "bottom")

ggsave("figures/posterior_risk_pois.pdf", p_risk, width = 5.5, height = 4.5)
cat("Saved figures/posterior_risk_pois.pdf\n")

# ------------------------------------------------------------------
# 2. RMSE
# ------------------------------------------------------------------
df_rmse <- df_sub %>%
  select(epsilon_pct, MLE = MLE_RMSE, Bayes = Bayes_RMSE, Rho = Rho_RMSE) %>%
  pivot_longer(cols = c(MLE, Bayes, Rho), names_to = "Method", values_to = "RMSE") %>%
  mutate(Method = factor(Method, levels = c("MLE", "Bayes", "Rho")))

p_rmse <- ggplot(df_rmse, aes(x = epsilon_pct, y = RMSE, colour = Method, linetype = Method)) +
  geom_line(linewidth = 1.1) + geom_point(size = 2.5) +
  scale_method_colour() + scale_method_linetype() +
  labs(x = TeX("Contamination rate $\\varepsilon$ (%)"),
       y = "RMSE", colour = "", linetype = "") +
  theme(legend.position = "bottom")

ggsave("figures/rmse_pois.pdf", p_rmse, width = 5.5, height = 4.5)
cat("Saved figures/rmse_pois.pdf\n")

# ------------------------------------------------------------------
# 3. Density plot
# ------------------------------------------------------------------
eps_target <- 0.10
lam0       <- 3.0
a_gamma    <- 1.0
b_gamma    <- 1.0

row0 <- df_trials %>%
  filter(epsilon == eps_target, tau == tau_ref, n == n_ref) %>%
  slice(1)

lam_mle   <- row0$lam_mle
lam_bayes <- row0$lam_bayes_mean
lam_rho   <- row0$lam_rho_mean
eta_std   <- row0$eta_std_rho
n_obs     <- row0$n

# Reconstruct approximate posterior parameters
a_post <- a_gamma + lam_bayes * n_obs
b_post <- b_gamma + n_obs

lam_grid <- seq(0.01, max(lam_mle, lam_bayes, lam_rho, lam0) * 2.5, length.out = 400)

df_dens <- tibble(
  lambda  = rep(lam_grid, 3),
  density = c(
    dgamma(lam_grid, shape = a_gamma, rate = b_gamma),
    dgamma(lam_grid, shape = a_post, rate = b_post),
    dlnorm(lam_grid, meanlog = log(lam_rho), sdlog = eta_std)
  ),
  Component = factor(rep(c("Prior", "Bayes", "Rho"), each = length(lam_grid)),
                     levels = c("Prior", "Bayes", "Rho"))
)

p_dens <- ggplot(df_dens, aes(x = lambda, y = density, fill = Component)) +
  geom_area(alpha = 0.35, colour = NA) +
  geom_line(aes(colour = Component), linewidth = 1.0) +
  scale_fill_grey(
    start = 0.85, end = 0.4,
    labels = c(TeX("$\\pi(\\lambda)$"),
               TeX("$\\pi(\\lambda | x)$"),
               TeX("$\\rho_\\psi(\\lambda | x)$"))
  ) +
  scale_colour_grey(start = 0.4, end = 0.1, guide = "none") +
  geom_vline(xintercept = lam0, linetype = "solid", linewidth = 1.0, colour = "red") +
  geom_vline(xintercept = lam_mle, linetype = "dashed", linewidth = 0.9, colour = "black") +
  geom_vline(xintercept = lam_rho, linetype = "dotted", linewidth = 0.9, colour = "black") +
  labs(x = TeX("$\\lambda$"), y = "Density", fill = NULL) +
  theme(legend.position = "bottom")

ggsave("figures/density_plot_pois.pdf", p_dens, width = 5.5, height = 4.5)
cat("Saved figures/density_plot_pois.pdf\n")
