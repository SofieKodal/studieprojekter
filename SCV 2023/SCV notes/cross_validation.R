library(MASS)
library(DAAG)

df <- mcycle[mcycle$times <= 40,]

# Define the Epanechnikov kernel function
epanechnikov_kernel <- function(x) {
  condition <- abs(x) <= 1
  result <- numeric(length(x))
  result[condition] <- 3/4 * (1 - x[condition]^2)
  return(result)
}

# Candidate values for p and h
candidate_p <- c(1, 2, 3)
candidate_h <- seq(3, 15, by = 1)  # Adjust the sequence as needed

# Initialize variables to keep track of the best values
best_p <- NA
best_h <- NA
min_cv_error <- Inf

# Perform cross-validation
for (p in candidate_p) {
  for (h in candidate_h) {
    # Fit the local polynomial smoother
    model <- loess(formula = accel ~ times, data = df, span = h, degree = p, surface = "direct", weights = epanechnikov_kernel((df$times - times) / h))
    
    # Calculate cross-validation error using cv.lm
    cv_error <- cv.lm(data = df, object = model)$cvm
    
    # Check if this combination has a lower CV error
    if (cv_error < min_cv_error) {
      min_cv_error <- cv_error
      best_p <- p
      best_h <- h
    }
  }
}

cat("Best p value:", best_p, "\n")
cat("Best h value:", best_h, "\n")
cat("Minimum CV error:", min_cv_error, "\n")

