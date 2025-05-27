### Exercise 1

# 1. Implement LCG (linear congruential generator)
 
lcg <- function(a,c,M,x0,n){
  u <- vector(length=n)
  x <- vector(length=n)
  u[1] <- x0/M
  x[1] <- x0
  for(i in 2:n){
    x[i] <- (a*x[i-1] + c) %% M
    u[i] <- x[i]/M
  }
  #return(c(x,u))
  return(list(x=x, u=u))
}

"a. Generate 10.000 (pseudo-) random numbers and present these numbers in a 
histogramme (e.g. 10 classes)."
a <- 5
c <- 1
M <- 16
x0 <- 3
n <- 10000

x <- lcg(a,c,M,x0,n)$x
u <- lcg(a,c,M,x0,n)$u

r1 <- lcg(a=129,c=26461,M=64499,x0=30,n=10000)$u
r2 <-lcg(a=5,c=1,M=16,x0=3,n=10000)$u

par(mfrow=c(1,1))
hist(x, breaks=10, main="Randomly generated numbers from LCG", xlab="Values", ylab="Frequency")

"b. Evaluate the quality of the generator by graphical
descriptive statistics (histogrammes, scatter plots) and
statistical tests - χ2,Kolmogorov-Smirnov, run-tests, and
correlation test. "

# Scatter plot
plot(u[-n],u[-1], main="Plot of Ui againt U(i-1)", xlab="U(i-1)", ylab="Ui")

############ chi^2 test ##########
chi_test_statistic <- function(data, n_classes){
  n_observed <- table(data)
  n_expected <- length(data) / n_classes
  T = sum((n_observed-n_expected)**2/n_expected)
  return(T)
}

chi_test <- function(u, df, n_classes){
  test_stat = chi_test_statistic(u, n_classes)
  p_val <- 1 - pchisq(test_stat, df)
  return(p_val)
}

n_classes = 100 # max 2000
m = 0 # number of estimated parameters
df = n_classes - 1 - m

chi_test(u,df,n_classes)

######## Kolmogorov-Smirnov test ####
ks_ecdf <- function(u){
  n <- length(u)
  u_sorted <- sort(u)
  
  ecdf <- rep(0,n)
  for(i in 1:n){
    ecdf[i] <- sum(u_sorted<=u[i]) / n
  }
  return(ecdf)
}


# make function for ks_test
ks_test <- function(u){
  n <- length(u)
  
  F <- (1:n) / n
  Fn <- ks_ecdf(u)
  
  Dn <- max(abs(Fn - F))
  
  k <- 1:n
  p_val <- 1 - 2 * sum((-1)^(k-1) * exp(-2*k^2*Dn^2))
  return(p_val)
}

ks_test(u)

###### Run-tests #####

## 1. Wald-Wolfowitz run test
runtest_wald_wolf <- function(u){
  med <- median(u)
  n1 <- sum(u>med)
  n2 <- sum(u<=med)
  
  mu <- 2 * (n1*n2) / (n1+n2) + 1
  var <- 2 * (n1*n2*(2*n1*n2-n1-n2)) / ((n1+n2)^2*(n1+n2-1))
  
  runs_above = rle(u>med)
  runs_below = rle(u<=med)
  
  Ra = sum(runs_above$values & runs_above$lengths > 0)
  Rb = sum(runs_below$values & runs_below$lengths > 0)
  
  T = Ra + Rb
  
  return ((1 - pnorm(abs(T), mean=mu, sd=sqrt(var)) ))
}

runtest_wald_wolf(u)

## 2. Up/Down test
runtest_up_down <- function(u){
  n <- length(u)
  runs = rle(u[1:(n-1)]<u[2:n])
  R = matrix(
    c(sum(runs$values & runs$lengths == 1),
    sum(runs$values & runs$lengths == 2),
    sum(runs$values & runs$lengths == 3),
    sum(runs$values & runs$lengths == 4),
    sum(runs$values & runs$lengths == 5),
    sum(runs$values & runs$lengths >= 6)),
    nrow=6, ncol=1)
  
  A <- matrix(
    c(4529.4, 9044.9, 13568, 18091, 22615, 27892,
         9044.9, 18097, 27139, 36187, 45234, 55789,
         13568, 27139, 40721, 54281, 67852, 83685,
         18091, 36187, 54281, 72414, 90470, 111580,
         22615, 45234, 67852, 90470, 113262, 139476,
         27982, 55789, 83685, 111580, 139476, 172860),
    nrow=6, ncol=6)
  B <- matrix(
    c(1/6, 5/24, 11/120, 19/720, 29/5040, 1/840),
    nrow=6, ncol=1)
  
  Z <- 1/(n-6) * t(R-n*B) %*% A %*% (R-n*B)
  p_val <- 1 - pchisq(Z, 6)
  
  return(p_val)
  
}
runtest_up_down(u)


## 3. Up and Down test
runtest_up_down2 <- function(u){
  n <- length(u)
  runs = rle(u[1:(n-1)]<u[2:n])
  x = sum(runs$values & runs$lengths > 0)
  
  z = (x-((2*n-1)/3))/(sqrt(16*n-29)/90)
  return ((1 - pnorm(abs(z), mean=0, sd=1) ))
}


## Correlation test
cor_test <- function(u,lags){
  c <- vector()
  n <- length(u)
  
  for(h in 1:floor(n/lags)){
    sum_term <- 0
    for(i in 1:(n-h)){
      sum_term <- sum_term + u[i]*u[i+h]
    }
    c[h] <- 1/(n-h) * sum_term
  }
  
  x <- seq(min(c), max(c), length=length(c))
  y <- dnorm(x, mean=0.25, sd=sqrt(7/(144*n)))
  
  hist(c, breaks=10, probability=TRUE, col="lightgray", main="Estimated expected correlation", xlab="Correlation")
  curve(dnorm(x, mean=0.25, sd=7/(144*7)), col="darkblue", lwd=2, add=TRUE)
  legend("topright", legend=c("Estimated", "Expected"), col=c("lightgray","darkblue"), lwd=2, cex=.8)
}

cor_test(r2,10)


