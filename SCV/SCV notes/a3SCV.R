library(mets)

h_AMISE <- function(n, N, alpha, beta){
  
  ## input:
  # n = sample size
  # N = block size
  # alpha, beta = parameters for the Beta(alpha, beta)-distribution

  
  ## output:
  # optimal bandwidth for selected parameters
  
  
  # Simulate data
  sd <- 1 #for e~N(0,sd^2)
  supp_X <- 1
  
  m <- function(x){
    sin((x/3 + 0.1)^(-1))
  }
  
  X <- rbeta(n, alpha, beta)
  e <- rnorm(n, 0, sd^2)
  Y <- m(X) + e
  df <- data.frame(X,Y)
  
  # Split data into N blocks
  df_split <- split(df, (seq(nrow(df))-1) %/% (length(X)/N))
  
  m_hat <- function(x, coef){
    return(coef[1] + coef[2]*x + coef[3]*x^2 + coef[4]*x^3 + coef[5]*x^4)
  }
  dm2_hat <- function(x, coef){
    return(2*coef[3] + 3*2*coef[4]*x + 4*3*coef[5]*x^2)
  }
  
  for(i in 1:n){
    for(j in 1:N){
      fit <- lm(Y~X + I(X^2) + I(X^3) + I(X^4), data=df_split[[j]])
      coef <- fit$coefficients
      
      phi_sum <- 0
      var_sum <- 0
      
      phi_sum_iteration <- (dm2_hat(X[i], coef) * dm2_hat(X[i], coef))
      phi_sum <- phi_sum + phi_sum_iteration
      
      var_sum_iteration <- (Y[i] - m_hat(X[i], coef))^2
      var_sum <- var_sum + var_sum_iteration
    }
  }
  
  phi_hat <- 1/n * phi_sum
  if((n-5*N) == 0){
    var_hat <- 1/(n-5*N+0.005) * var_sum
  }
  else{
    var_hat <- 1/(n-5*N) * var_sum
  }
  
  h_AMISE <- n^(-1/5) * ((35*var_hat^2 * abs(supp_X)) / phi_hat)^(1/5)
  return(h_AMISE)
}


n <- 100
N <- 5
alpha <- 5
beta <- 5

n_seq <- seq(50,1000, length.out=10)
N_seq <- seq(2,92, length.out=10) # 52
alpha_seq <- seq(1,10, length.out=10)
beta_seq <- seq(1,10, length.out=10)

h_opt_n <- c()
h_opt_N <- c()
h_opt_a <- c()
h_opt_b <- c()

for(i in 1:length(n_seq)){
  h_opt_n[i] <- h_AMISE(n_seq[i], N, alpha, beta)
  h_opt_N[i] <- h_AMISE(n, N_seq[i], alpha, beta)
  h_opt_a[i] <- h_AMISE(n, N, alpha_seq[i], beta)
  h_opt_b[i] <- h_AMISE(n, N, alpha, beta_seq[i])
}


library(ggplot2)
library(gridExtra)
plot1 <- ggplot(data = NULL, aes(x = n_seq, y = h_opt_n)) +
  geom_line() +
  labs(x = "Number of samples", y = NULL)

plot2 <- ggplot(data = NULL, aes(x = N_seq, y = h_opt_N)) +
  geom_line() +
  labs(x = "Number of blocks", y = NULL)

plot3 <- ggplot(data = NULL, aes(x = alpha_seq, y = h_opt_a)) +
  geom_line() +
  labs(x = "alpha-value", y = NULL)

plot4 <- ggplot(data = NULL, aes(x = alpha_seq, y = h_opt_b)) +
  geom_line() +
  labs(x = "beta-value", y = NULL)

grid.arrange(plot1, plot2, plot3, plot4, ncol = 2, left="Optimal bandwidth")



