## Exercise 

#### 1. Bootstrap mean ####
bootstrap <- function(x,r,a,b){
  n <- length(x)
  mu <- mean(x)
  count_p <- 0

  bootstrap_rep <- vector()
  for(i in 1:r){
    boot_sample <- sample(x,size=n,replace=TRUE)
    boot_sample_mean <- mean(boot_sample)
    
    if (a < (boot_sample_mean - mu) && (boot_sample_mean - mu) < b) {
      count_p <- count_p + 1
    }
  }
  return(count_p/r)
}

n <- 10
X <- c(56,101,78,67,93,87,64,72,80,69)
mu <- mean(X)
a <- -5
b <- 5
r <- 10000 # num_bootstrap simulations

print(bootstrap(X,r,a,b))

#### 2. Bootstrap variance ####
n <- 15
X <- c(5,4,9,6,21,17,11,20,7,10,21,15,13,16,8)

bootstrap_variance_of_variance <- function(data,r){
  n <- length(data)
  boot_sample_var <- vector()
  
  for(i in 1:r){
    boot_sample <- sample(data,size=n,replace=TRUE)
    boot_sample_var[i] <- var(boot_sample)
  }
  return(var(boot_sample_var))
}

print(bootstrap_variance_of_variance(X,r))

#### 3.

bootstrap_variance_of_mean <- function(data,r){
  n <- length(data)
  boot_sample_mean <- vector()
  for(i in 1:r){
    boot_sample <- sample(data,size=n,replace=TRUE)
    boot_sample_mean[i] <- mean(boot_sample)
  }
  mean <- mean(data)
  mean_variance_est <- var(boot_sample_mean)
  return(c(mean, mean_variance_est))
}

bootstrap_variance_of_median <- function(data,r){
  n <- length(x)
  boot_sample_med <- vector()
  for(i in 1:r){
    boot_sample <- sample(data,size=n,replace=TRUE)
    boot_sample_med[i] <- median(boot_sample)
  }
  median <- median(data)
  median_variance_est <- var(boot_sample_med)
  return(c(median,median_variance_est))
}

k <- 1.05
beta <- 1
n <- 200
r <- 100

data <- rpareto(n, location=beta, shape=1.05)

# (a)
mean(data)
median(data)

# (b)
bootstrap_variance_of_mean(data,r)

# (c)
bootstrap_variance_of_median(data,r)

# (d)
bootstrap_mean_var <- vector()
bootstrap_median_var <- vector()

n_sims <- 100
for(i in 1:n_sims){
  bootstrap_mean_var[i] <- bootstrap_variance_of_mean(data,r)[2]
  bootstrap_median_var[i] <- bootstrap_variance_of_median(data,r)[2]
}

par(mfrow = c(2, 1))
plot(bootstrap_mean_var, type = "l", col = "blue", xlab = "Index", ylab = "Mean Estimate", main = "Bootstrap Mean Estimate")
plot(bootstrap_median_var, type = "l", col = "red", xlab = "Index", ylab = "Median Estimate", main = "Bootstrap Median Estimate")
par(mfrow = c(1, 1))  # Reset to default layout (optional)
