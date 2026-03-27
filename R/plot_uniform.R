#!/usr/bin/env Rscript
# ============================================================
# Uniform scale model -- publication figures.
#
# Reads:   results/uniform_summary.csv, results/uniform_trials.csv
# Writes:  figures/posterior_risk_uniform.pdf
#          figures/rmse_uniform.pdf
#          figures/density_plot_uniform.pdf
#
# Usage:   Rscript R/plot_uniform.R
# ============================================================

source("R/theme.R")
set_jasa_theme()

df_summary <- read_csv("results/uniform_summary.csv", show_col_types = FALSE)
df_trials  <- read_csv("results/uniform_trials.csv", show_col_types = FALSE)

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

ggsave("figures/posterior_risk_uniform.pdf", p_risk, width = 5.5, height = 4.5)
cat("Saved figures/posterior_risk_uniform.pdf\n")

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

ggsave("figures/rmse_uniform.pdf", p_rmse, width = 5.5, height = 4.5)
cat("Saved figures/rmse_uniform.pdf\n")

# ------------------------------------------------------------------
# 3. Density plot
# ------------------------------------------------------------------
eps_target   <- 0.10
t0           <- 1.0
alpha_prior  <- 2.0
a_prior      <- 0.5

row0 <- df_trials %>%
  filter(epsilon == eps_target, tau == tau_ref, n == n_ref) %>%
  slice(1)

t_mle   <- row0$theta_mle
t_bayes <- row0$theta_bayes_mean
t_rho   <- row0$theta_rho_mean
u_std   <- row0$std_rho
n_obs   <- row0$n

# Limit grid to a readable range (excluding extreme outlier MLE)
t_grid <- seq(a_prior, min(max(t_rho, t0) * 3, 5.0), length.out = 400)

# Pareto prior (unnormalised) and posterior
prior_unnorm <- t_grid^(-alpha_prior)
prior_pdf    <- prior_unnorm / pracma::trapz(t_grid, prior_unnorm)

post_unnorm  <- t_grid^(-(n_obs + alpha_prior))
post_unnorm[t_grid < max(a_prior, t_mle)] <- 0
post_area    <- pracma::trapz(t_grid, post_unnorm)
post_pdf     <- if (post_area > 0) post_unnorm / post_area else post_unnorm

# Rho-posterior as lognormal
rho_pdf <- dlnorm(t_grid, meanlog = log(t_rho), sdlog = max(u_std, 0.01))

df_dens <- tibble(
  theta   = rep(t_grid, 3),
  density = c(prior_pdf, post_pdf, rho_pdf),
  Component = factor(rep(c("Prior", "Bayes", "Rho"), each = length(t_grid)),
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
  geom_vline(xintercept = t0, linetype = "solid", linewidth = 1.0, colour = "red") +
  geom_vline(xintercept = t_rho, linetype = "dotted", linewidth = 0.9, colour = "black") +
  labs(x = TeX("$\\theta$"), y = "Density", fill = NULL) +
  theme(legend.position = "bottom")

ggsave("figures/density_plot_uniform.pdf", p_dens, width = 5.5, height = 4.5)
cat("Saved figures/density_plot_uniform.pdf\n")
