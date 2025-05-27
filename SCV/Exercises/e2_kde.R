library(MASS)

# Define the mean vector and covariance matrix
mean_vector <- c(0, 0)  # Mean of the distribution
cov_matrix <- matrix(c(1, 0.3, 0.3, 1), nrow = 2)  # Covariance matrix

# Set the random seed for reproducibility (optional)
set.seed(123)

# Simulate data from the bivariate normal distribution
n_samples <- 100
bivariate_data <- mvrnorm(n = n_samples, mu = mean_vector, Sigma = cov_matrix)

x <- bivariate_data[,1]
y <- bivariate_data[,2]

#Epanechnikov
k_epa <- function(x){
  if(abs(x) <= 1){
    return(1-x^2)
  }
  else{
    return(0)
  }
}

h <- seq(0.1, 3, by= 0.15)
err <- c()

for(i in 1:length(h)){
  kde <- kde2d(x, y, h[i], n=n_samples)
  err[i] <- sqrt(1/n_samples*sum((y-kde$y)^2))
}

#forkert

for(i in 1:length(h)){
  contour_plot <- filled.contour(kde[[i]], xlab = "X-axis", ylab = "Y-axis", main = "2D KDE Contour Plot")
}


persp(kde[[1]]$x, kde[[1]]$y, kde[[1]]$z, theta = 30, phi = 20,
      col = "lightblue", border = NA,
      xlab = "X-axis", ylab = "Y-axis", zlab = "Density",
      main = "3D KDE Surface Plot")

