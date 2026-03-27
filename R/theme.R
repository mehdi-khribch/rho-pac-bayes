# ============================================================
# Shared ggplot theme for JASA publication figures.
#
# Usage: source("R/theme.R") at the top of each plotting script.
# ============================================================

library(tidyverse)
library(ggthemes)
library(latex2exp)

set_jasa_theme <- function() {
  theme_set(theme_tufte())
  theme_update(
    axis.text.x       = element_text(size = 16),
    axis.text.y       = element_text(size = 16),
    axis.title.x      = element_text(size = 20, margin = margin(15, 0, 0, 0)),
    axis.title.y      = element_text(size = 20, angle = 90, margin = margin(0, 15, 0, 0)),
    panel.grid.major   = element_line(linewidth = 0.25, linetype = "solid", colour = "gray85"),
    panel.grid.minor   = element_line(linewidth = 0.25, linetype = "solid", colour = "gray90"),
    legend.text        = element_text(size = 16),
    legend.title       = element_text(size = 16),
    strip.text         = element_text(size = 18),
    strip.background   = element_rect(fill = "white"),
    legend.position    = "bottom",
    plot.title         = element_blank(),
    plot.subtitle      = element_blank()
  )
}

# Standard colour and linetype scales used across all figures
scale_method_colour <- function() {
  scale_colour_grey(start = 0.2, end = 0.7)
}

scale_method_linetype <- function() {
  scale_linetype_manual(values = c("solid", "dashed", "solid"))
}
