# Exercise 3

#### 1. Generate simulated values ####

## Exponential distribution
n <- 10000
lambda <- 1/3
U <- runif(n=n, min=0, max=1)

X_exp <- -log(U)/lambda

par(mfrow=c(1,1))
hist(X_exp)

## Normal distribution
# Box-Muller method

Z <- vector()
for (i in 1:(n-1)){
  new_val <- sqrt( -2 * log(U[i]) ) * cos( 2 * pi * U[i+1] )
  Z <- c(Z, new_val)
}
hist(Z)


## Pareto distribution

beta <- 1
k_values <- c(2.05, 2.5, 3, 4)

library(EnvStats)

pareto <- function(U,beta,k){
  return(beta*( U^(-1/k) ))
}

X_r_df <- data.frame(X_r = numeric(n))
X_pareto_df <- data.frame(X_pareto = numeric(n))

par(mfrow=c(2,2))
for (k in k_values){
  X_r <- rpareto(n=n, location=beta, shape=k)
  X_pareto <- pareto(U,beta,k)
  #hist([X_r,X_pareto], breaks=20, main=paste("k =", k), xlab="Values", ylab="Frquency")
  
  hist(X_r, breaks = 20, main = paste("k =", k), xlab = "Values", ylab = "Frequency", col = "blue", xlim = range(c(X_r, X_pareto)), ylim = range(c(hist(X_r, breaks=20, plot=FALSE)$counts, hist(X_pareto, breaks=20, plot=FALSE)$counts)))
  hist(X_pareto, breaks = 20, col = "red", add = TRUE)
  legend("topright", c("R", "Pareto"), col=c("blue", "red"), lwd=2, cex=0.6)

  X_r_df <- cbind(X_r_df, X_r) 
  X_pareto_df <- cbind(X_pareto_df, X_pareto) 
}

X_r_df <- X_r_df[-1]
X_pareto_df <- X_pareto_df[-1]

colnames(X_r_df) <- paste0("X_r_", k_values)
colnames(X_pareto_df) <- paste0("X_pareto_", k_values)

#### 2. Mean and variance of Pareto ####

E_analytical <- vector()
var_analytical <- vector()

for (k in k_values){
  E_analytical <- c(E_analytical, beta * (k/(k-1)))
  var_analytical <- c(var_analytical, beta^2 * (k / ( (k-1)^2 * (k-2) )))
}

E_sims <- colMeans(X_pareto_df)
var_sims <- apply(X_pareto_df, 2, var)

E_table <- data.frame(E_sims, E_analytical)
var_table <- data.frame(var_sims, var_analytical)

print(E_table)
print(var_table)

#### 3. Normal distribution ####


library(DescTools)
conf_lower=c()
conf_upper=c()
conf_lower_var=c()
conf_upper_var=c()

for (i in 1:100){
  X_norm=c()
  U=runif(20)
  for (i in seq(from=1,to=20,by=2)){
    X_norm=c(X_norm,sqrt(-2*log(U[i]))*cos(2*pi*U[i+1]))
  }
  stat=t.test(X_norm)
  conf_lower=c(conf_lower,stat$conf.int[1])
  conf_upper=c(conf_upper,stat$conf.int[2])
  conf_lower_var=c(conf_lower_var,VarCI(X_norm)[2])
  conf_upper_var=c(conf_upper_var,VarCI(X_norm)[3])
}

which(conf_lower>=0)
which(conf_upper<=0)

which(conf_lower_var>=1)
which(conf_upper_var<=1)


#### 4. Simulate from the Pareto distribution using composition ####





