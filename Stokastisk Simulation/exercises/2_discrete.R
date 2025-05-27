# Exercise 2

#### 1. Geometric distribution ####

p <- 1/3
n <- 10000

# Metode fra undervisning
U <- runif(n=n, min=0, max=1)
X <- floor((log(U) / log(1-p))) + 1
hist(X, breaks=10, main="Geometric distribution", xlab="Values", ylab="Frequency")

# Direkte i R
rgeom(n=n,prob=p)

#### 2. Simulate the 6 point distribution ####

X <- c(1, 2, 3, 4, 5, 6)
p <- c(7/48, 5/48, 1/8, 1/16, 1/4, 5/16)

barplot(p)

## direct (crude) method

# make the acummulated sum of p to find interals
intervals <- c(0, cumsum(p))
# put U into those intervals
X_direct <- as.numeric(cut(U, intervals, labels=X))

## rejection method

k <- length(p)
c <- max(p)
X_rej <- vector()

for(i in 1:(n-1)){
  I <- floor(k*U[i]) + 1
  if(U[i+1] < p[I]/c){
    X_rej <- c(X_rej, I)
  }
}

hist(X_sim)


## alias method

L <- seq(1,k)
F <- k*p
G <- which(F >= 1)
S <- which(F < 1)
eps <- 0.01

while (length(S) > 0) {
  # a
  i <- G[1]
  j <- S[1]
  
  # b
  L[j] <- i
  F[i] <- F[i] - (1 - F[j])
  
  # c
  if (F[i] < 1 - eps) {
    G <- G[-1] #fjern første element
    S <- c(S, i)
  }
  
  # d
  S <- S[-1]
}

print(L) # 1,2,3 sender til 5 og 4,5,6 sender til 6
print(F) # proportionen af sig selv og alias

X_alias <- vector()

for(i in 1:(n-1)){
  I <- floor(k*U[i]) + 1
  if(U[i+1] < F[I]){
    X_alias <- c(X_alias, I)
  }
  else{
    X_alias <- c(X_alias, L[I])
  }
}

## Plot of all methods
par(mfrow = c(1, 3))
hist(X_direct, main = "Direct method", xlab = "Value")
hist(X_rej, main = "Rejection method", xlab = "Value")
hist(X_alias, main = "Alias method", xlab = "Value")
par(mfrow=c(1,1))
