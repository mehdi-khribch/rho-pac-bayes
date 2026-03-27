#!/usr/bin/env Rscript
# ============================================================
# Regression experiments -- publication figures.
#
# Reads:   results/fourier_summary.csv
#          results/correlated_summary.csv
#          results/correlated_predictions.csv
# Writes:  figures/fourier_bayes_risk.pdf
#          figures/fourier_rmse.pdf
#          figures/posterior_risk_regression.pdf
#          figures/rmseregression.pdf
#          figures/predictedvsfitted.pdf
#
# Usage:   Rscript R/plot_regression.R
# ============================================================

source("R/theme.R")
set_jasa_theme()

# ==============================================================
# Fourier basis regression
# ==============================================================

df_fourier <- read_csv("results/fourier_summary.csv", show_col_types = FALSE)

df_f <- df_fourier %>%
  arrange(epsilon) %>%
  mutate(epsilon_pct = 100 * epsilon)

# Fourier: Bayes risk
df_fr <- df_f %>%
  select(epsilon_pct, MLE = MLE_BayesRisk, Bayes = Bayes_BayesRisk, Rho = Rho_BayesRisk) %>%
  pivot_longer(cols = c(MLE, Bayes, Rho), names_to = "Method", values_to = "Risk") %>%
  mutate(Method = factor(Method, levels = c("MLE", "Bayes", "Rho")))

p_fourier_risk <- ggplot(df_fr, aes(x = epsilon_pct, y = Risk, colour = Method, linetype = Method)) +
  geom_line(linewidth = 1.1) + geom_point(size = 2.5) +
  scale_method_colour() + scale_method_linetype() +
  labs(x = TeX("Contamination rate $\\varepsilon$ (%)"),
       y = "Posterior risk", colour = "", linetype = "") +
  theme(legend.position = "bottom")

ggsave("figures/fourier_bayes_risk.pdf", p_fourier_risk, width = 6, height = 4.5)
cat("Saved figures/fourier_bayes_risk.pdf\n")

# Fourier: RMSE
df_frmse <- df_f %>%
  select(epsilon_pct, MLE = MLE_RMSE, Bayes = Bayes_RMSE, Rho = Rho_RMSE) %>%
  pivot_longer(cols = c(MLE, Bayes, Rho), names_to = "Method", values_to = "RMSE") %>%
  mutate(Method = factor(Method, levels = c("MLE", "Bayes", "Rho")))

p_fourier_rmse <- ggplot(df_frmse, aes(x = epsilon_pct, y = RMSE, colour = Method, linetype = Method)) +
  geom_line(linewidth = 1.1) + geom_point(size = 2.5) +
  scale_method_colour() + scale_method_linetype() +
  labs(x = TeX("Contamination rate $\\varepsilon$ (%)"),
       y = "RMSE", colour = "", linetype = "") +
  theme(legend.position = "bottom")

ggsave("figures/fourier_rmse.pdf", p_fourier_rmse, width = 6, height = 4.5)
cat("Saved figures/fourier_rmse.pdf\n")

# ==============================================================
# Correlated design regression
# ==============================================================

df_corr <- read_csv("results/correlated_summary.csv", show_col_types = FALSE)

df_c <- df_corr %>%
  arrange(epsilon) %>%
  mutate(epsilon_pct = 100 * epsilon)

# Correlated: posterior risk
df_cr <- df_c %>%
  select(epsilon_pct, OLS = MLE_BayesRisk, Bayes = Bayes_BayesRisk, Rho = Rho_BayesRisk) %>%
  pivot_longer(cols = c(OLS, Bayes, Rho), names_to = "Method", values_to = "Risk") %>%
  mutate(Method = factor(Method, levels = c("OLS", "Bayes", "Rho")))

p_corr_risk <- ggplot(df_cr, aes(x = epsilon_pct, y = Risk, colour = Method, linetype = Method)) +
  geom_line(linewidth = 1.1) + geom_point(size = 2.5) +
  scale_method_colour() + scale_method_linetype() +
  labs(x = TeX("Contamination rate $\\varepsilon$ (%)"),
       y = "Posterior risk", colour = "", linetype = "") +
  theme(legend.position = "bottom")

ggsave("figures/posterior_risk_regression.pdf", p_corr_risk, width = 5.5, height = 4.5)
cat("Saved figures/posterior_risk_regression.pdf\n")

# Correlated: RMSE
df_crmse <- df_c %>%
  select(epsilon_pct, OLS = MLE_RMSE, Bayes = Bayes_RMSE, Rho = Rho_RMSE) %>%
  pivot_longer(cols = c(OLS, Bayes, Rho), names_to = "Method", values_to = "RMSE") %>%
  mutate(Method = factor(Method, levels = c("OLS", "Bayes", "Rho")))

p_corr_rmse <- ggplot(df_crmse, aes(x = epsilon_pct, y = RMSE, colour = Method, linetype = Method)) +
  geom_line(linewidth = 1.1) + geom_point(size = 2.5) +
  scale_method_colour() + scale_method_linetype() +
  labs(x = TeX("Contamination rate $\\varepsilon$ (%)"),
       y = "RMSE", colour = "", linetype = "") +
  theme(legend.position = "bottom")

ggsave("figures/rmseregression.pdf", p_corr_rmse, width = 5.5, height = 4.5)
cat("Saved figures/rmseregression.pdf\n")

# Correlated: predicted vs true
df_pred <- read_csv("results/correlated_predictions.csv", show_col_types = FALSE)

df_pred_sub <- df_pred %>% filter(epsilon == 0.10)

p_pred <- ggplot(df_pred_sub, aes(x = y_true, y = y_pred_rho)) +
  geom_point(size = 1.5, alpha = 0.6, colour = "gray30") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "red", linewidth = 0.8) +
  labs(x = TeX("True $X\\beta^\\star$"),
       y = TeX("Predicted $X\\hat{\\beta}_\\rho$")) +
  theme(legend.position = "none")

ggsave("figures/predictedvsfitted.pdf", p_pred, width = 5.5, height = 5)
cat("Saved figures/predictedvsfitted.pdf\n")
