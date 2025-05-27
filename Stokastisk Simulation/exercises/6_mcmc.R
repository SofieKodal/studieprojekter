### Exercise 6

m <- 10
mean_service_time <- 8
mean_time_between_customers <- 1
A <- mean_time_between_customers * mean_service_time

num_costumers <- 10000
num_simulations <- 10

#### Functions ####
#### chi test 1D
chi_test_statistic <- function(data, n_classes){
  I <- seq(0,10,1)
  gI <- g(I)
  
  n_observed <- table(data)
  n_expected <- gI/sum(gI) * length(data)
  T = sum((n_observed-n_expected)^2/n_expected)
  return(T)
}

chi_test <- function(data, df, n_classes){ #df = degrees of freedom
  test_stat <- chi_test_statistic(data, n_classes)
  p_val <- 1 - pchisq(test_stat, df) 
  return(p_val)
}

#### chi test 2D
chi_test_statistic_2d <- function(data_matrix, n_classes){
  gij <- matrix(nrow=11,ncol=11)
  for(i in 0:10){
    for(j in 0:10){
      gij[i+1,j+1] <- g2(i,j)
      if(i+j>10){
        gij[i+1,j+1] <- 0
      }
    }
  }
  
  #gij_t=t(apply(gij,1,rev))
  #gij_vect=c(gij_t[upper.tri(gij_t,diag=TRUE)])
  
  #data_matrix_t <- t(apply(data_matrix,1,rev))
  #data_matrix_upper <- c(data_matrix[upper.tri(data_matrix,diag=TRUE)])
  
  n_observed <- data_matrix
  n_expected <- gij/sum(gij) * sum(data_matrix)
  matrix <- (n_observed-n_expected)^2/n_expected
  matrix[is.nan(matrix)] <- 0
  matrix[!is.finite(matrix)] <- 0
  T = sum(matrix)
  return(T)
}

chi_test_2d <- function(data, df, n_classes){ #df = degrees of freedom
  test_stat <- chi_test_statistic_2d(data, n_classes)
  p_val <- 1 - pchisq(test_stat, df) 
  return(p_val)
}

#### 1. Metropolis-Hastings ####
g <- function(i){
  A <- 1*8
  P <- A^i/factorial(i)
  return(P)
}

I <- seq(0,10,1)
gi <- g(I)

library(purrr)
x <- vector()
x[1] <- 1

for(i in 1:(num_costumers-1)){
  y <- rdunif(1,a=0,b=m)
  frac <- g(y)/g(x[i])
  prob <- runif(1)
  if(frac>=prob){
    x[i+1] <- y
  }
  if(frac<prob){
    x[i+1] <- x[i]
  }
}

# remove first 1000 for burn in period
state_list <- x[1001:10000]
hist(state_list)

test_stat_1 <- chi_test_statistic(state_list, n_classes=11)
chi_test_1 <- chi_test(state_list, df=11-1, n_classes=11)

#### 2. Joint ####
g2 <- function(i,j){
  A1 <- 4
  A2 <- 4
  P <- A1^i/factorial(i) * A2^j/factorial(j)
  return(P)
}

gij <- matrix(nrow=11,ncol=11)
for(i in 0:10){
  for(j in 0:10){
    gij[i+1,j+1] <- g2(i,j)
    if(i+j>10){
      gij[i+1,j+1] <- 0
    }
  }
}

# (a)
m <- 10
xi <- vector()
xj <- vector()

xi[1] <- 1
xj[1] <- 1

for(i in 1:(num_costumers-1)){
  yi <- m
  yj <- m
  while(yi+yj>=m){
    yi <- rdunif(1,a=0,b=m)
    yj <- rdunif(1,a=0,b=m)
  }
  frac <- g2(yi,yj)/g2(xi[i],xj[i])
  
  prob <- runif(1)
  # if(frac>1){           #fjernes da y
  #   xi[i+1] <- yi
  #   xj[i+1] <- yj
  # }
  if(frac>=prob){ 
    xi[i+1] <- yi
    xj[i+1] <- yj
  }
  if(frac<prob){
    xi[i+1] <- xi[i]
    xj[i+1] <- xj[i]
  }
}

state_list2a <- matrix(c(xi[1001:10000],xj[1001:10000]),ncol=2)
state_matrix2a <- matrix(table(factor(xi[1001:10000], levels=0:10), factor(xj[1001:10000], levels=0:10)), ncol=11)
colnames(state_list2a) <- c("xi", "xj")

hist(state_list2a[,"xi"])
hist(state_list2a[,"xj"])

test_stat_2a <- chi_test_statistic_2d(state_matrix2a, n_classes=11)
chi_test_2a <- chi_test_2d(state_matrix2a, df=11-1, n_classes=11)


# (b)
m <- 10
xi <- vector()
xj <- vector()

xi[1] <- 1
xj[1] <- 1

for(i in 1:(num_costumers-1)){
  yi <- m
  yj <- m
  while(yi+yj>=m){
    yi <- rdunif(1,a=0,b=m)
    yj <- rdunif(1,a=0,b=m)
  }
  frac_i <- g(yi)/g(xi[i])
  frac_j <- g(yj)/g(xj[i])
  
  prob <- runif(1)
  if(frac_i>=prob){
    xi[i+1] <- yi
  }
  if(frac_j>=prob){
    xj[i+1] <- yj
  }
  
  
  if(frac_i<prob){
    xi[i+1] <- xi[i]
  }
  if(frac_j<prob){
    xj[i+1] <- xj[i]
  }
}

state_list2b <- matrix(c(xi[1001:10000],xj[1001:10000]),ncol=2)
state_matrix2b <- matrix(table(factor(xi[1001:10000], levels=0:10), factor(xj[1001:10000], levels=0:10)), ncol=11)
colnames(state_list2b) <- c("xi", "xj")

hist(state_list2b[,"xi"])
hist(state_list2b[,"xj"])


test_stat_2b <- chi_test_statistic_2d(state_matrix2b, n_classes=11)
chi_test_2b <- chi_test_2d(state_matrix2b, df=11-1, n_classes=11)

## Summarize
test_stats <- matrix(c(test_stat_1, test_stat_2a, test_stat_2b),ncol=3)
colnames(test_stats) <- c("1D", "2D", "2D coodinate-wise")
rownames(test_stats) <- c("test stat")
print(test_stats)

chi_tests <- matrix(c(chi_test_1, chi_test_2a, chi_test_2b),ncol=3)
colnames(chi_tests) <- c("1D", "2D", "2D coodinate-wise")
rownames(chi_tests) <- c("p-value")
print(chi_tests)


# (c)


#### 3. wtf ####

# (a)
# standard_normal <- function(x,y,rho){
#   pdf <- 1/(2*pi*sqrt(1-rho^2)) * exp(-1/(2*(1-rho^2)) * (x^2-2*rho*x*y+y^2))
#   return(pdf)
# }

generate_X <- function(n,rho){
  z1 <- rnorm(n)
  z2 <- rnorm(n)

  xi <- z1
  gamma <- rho*z2 + sqrt(1-rho^2)*z2
  
  theta <- exp(xsi)
  psi <- exp(gamma)
  
  pairs <- matrix(c(xi,gamma), ncol=2)
  colnames(pairs) <- c("xsi", "gamma")
  
  return(pairs)
}


# (b)
rho <- 1/2
n <- 10
generate_X(n,rho)

# (c)

set.seed(123)  # For reproducibility

# Generate a pair (θ, ψ) from the prior distribution
xi <- rnorm(1)
gamma <- 
theta <- exp(xi)
psi <- exp(gamma)

# Generate n = 10 observations from N(θ, ψ)
rho <- 1/2
n <- 10
X <- generate_X(n,rho) #<- rnorm(n, mean=theta, sd=sqrt(psi))

# Compute sample mean and variance
x_bar <- mean(X)
s_squared <- var(X)

# Posterior distribution parameters
posterior_theta_mean <- x_bar
posterior_theta_sd <- sqrt(psi / n)
posterior_psi_scale <- n * s_squared / (n - 1)

# Sample from the posterior distribution using Gibbs sampling or other MCMC method
library(MCMCpack)

# Initial values
theta_post <- rnorm(1, mean=posterior_theta_mean, sd=posterior_theta_sd)
psi_post <- posterior_psi_scale / rchisq(1, df=n-1)

# Perform Gibbs sampling
iterations <- 10000
theta_samples <- numeric(iterations)
psi_samples <- numeric(iterations)

for (i in 1:iterations) {
  # Sample Θ given Ψ and data
  theta_post <- rnorm(1, mean=posterior_theta_mean, sd=sqrt(psi_post / n))
  
  # Sample Ψ given Θ and data
  psi_post <- (n * s_squared) / rchisq(1, df=n-1)
  
  # Store samples
  theta_samples[i] <- theta_post
  psi_samples[i] <- psi_post
}

# Summarize posterior samples
summary(theta_samples)
summary(psi_samples)

# Plot posterior samples
par(mfrow=c(2, 1))
hist(theta_samples, main="Posterior of Θ", xlab="Θ", breaks=30)
hist(psi_samples, main="Posterior of Ψ", xlab="Ψ", breaks=30)





