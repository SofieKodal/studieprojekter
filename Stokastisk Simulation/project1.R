#### Functions ####
square_matrix <- function(matrix,power){
  if(potens==1){
    return(matrix)
  }
  dim <- dim(matrix)
  result <- matrix
  for(i in 1:(potens-1)){
    result <- result %*% matrix
    i <- i+1
  }
  return(result)
}

conf_interval <- function(alpha, x){
  n = length(x)
  mu_est = mean(x)
  s_theta_sq = abs(1/(n-1)*(sum(x^2)-n*mu_est^2))
  s_theta = sqrt(s_theta_sq)
  conf_int = c(mu_est + s_theta/sqrt(n)*qt(alpha/2,n-1),mu_est + s_theta/sqrt(n)*qt(1-alpha/2,n-1))
  
  cat("mu_est", round(mu_est,4), "\n")
  cat("conf_int", round(conf_int,4), "\n")
  
  return(c(mu_est, conf_int))
}
####################       Part 1       ####################
#### 1.1 ####

P <- matrix(c(0.9915,0.005,0.0025,0,0.001,
            0,0.986,0.005,0.004,0.005,
            0,0,0.992,0.003,0.005,
            0,0,0,0.991,0.009,
            0,0,0,0,1), nrow=5, byrow=TRUE)

num_simulations <- 1000
lifetime_distribution <- vector()
reappeared_locally <- vector()

for (i in 1:num_simulations) {
  state <- 1
  time <- 0
  reappeared <- FALSE
  
  while (state != 5) {
    time <- time + 1
    state <- sample(1:5, 1, prob = P[state, ])
    if (state == 2) {
      reappeared <- TRUE
    }
  }
  
  lifetime_distribution[i] <- time
  reappeared_locally[i] <- reappeared
}

hist(lifetime_distribution)
frac_reappered <- sum(reappeared_locally) / length(reappeared_locally)
print(frac_reappered)

#### 1.2 ####
p0 <- P[1,]
analytical_p120 <- p0 %*% square_matrix(P,120)

num_simulations <- 1000
state_at_120 <- numeric(num_simulations)

for (i in 1:num_simulations) {
  state <- 1
  for (t in 1:120) {
    state <- sample(1:5, 1, prob = P[state, ])
  }
  state_at_120[i] <- state
}

simulated_p120 <- table(state_at_120) / num_simulations

observed <- as.numeric(simulated_p120)
expected <- as.numeric(analytical_p120)
chisq_test <- chisq.test(observed, p = expected, rescale.p = TRUE)

#### 1.3 ####
library(matlib)

Ps <- P[1:4,1:4]
pi <- P[1,1:4]
ps <- P[1:4,5]
one_vec <- c(1,1,1,1)
I <- diag(4)
expected_lifetime <- pi %*% solve(I-Ps) %*% one_vec

months <- 1200
t <- 1:1200
Pt <- vector()
for(i in 1:months){
  Pt[i] <- pi %*% square_matrix(Ps,i) %*% ps
}
par(mfrow=c(1,2))
plot(Pt, type='l', main="Theoretical pdf")
hist(lifetime_distribution, main="Simulated distribution")
par(mfrow=c(1,1))

#### Task 4 ####

keep <- 0
lifetime_distribution <- vector()
while (keep < 1000) {
  state <- 1
  time <- 0
  valid_sample <- TRUE
  
  while (state != 5) {
    time <- time + 1
    state <- sample(1:5, 1, prob = P[state, ])
    if (time == 12 && state == 1) {
      valid_sample <- FALSE
      break
    }
  }
  if(valid_sample){
    lifetime_distribution <- append(lifetime_distribution, time)
    keep <- keep + 1
  }
}

par(mfrow=c(1,1))
hist(lifetime_distribution)

#### Task 5 ####
num_simulations <- 100
num_women <- 200
fraction_died <- vector()
fraction_Z <- vector()

for(i in 1:num_simulations){
  died <- 0
  lifetime_distribution <- vector()
  for(j in 1:num_women){
    state <- 1
    time <- 0
    
    while(state != 5) {
      time <- time + 1
      state <- sample(1:5, 1, prob = P[state, ])
    }
    lifetime_distribution[j] <- time
    if(time <= 350){
      died <- died + 1
    }
  }
  X[i] <- died / num_women
  Y[i] <- mean(lifetime_distribution)
}
muY <- expected_lifetime
const <- -cov(X,Y)/var(Y)
Z <- X + const*(Y - c(muY))

(var(X)-var(Z))/var(X) # 69% reduction
conf_interval(0.05,X) # crude MC: [0.7339, 0.7455]
conf_interval(0.05,Z) # control variates: [0.7353, 0.7417]


####################       Part 2       ####################
#### Task 7 ####

Q <- matrix(c(-0.0085, 0.005, 0.0025, 0, 0.001,
              0, -0.014, 0.005, 0.004, 0.005,
              0,0, -0.008, 0.003, 0.005,
              0,0,0,-0.009, 0.009,
              0,0,0,0,0), nrow=5, byrow=TRUE)

# Make probability matrix
Prob <- matrix(rep(0,25),nrow=5)
for(row in 1:4){
  Prob[row,] <- -Q[row,]/diag(Q)[row]
}
diag(Prob) <- c(0,0,0,0,0)

num_simulations <- 1000
lifetime_distribution <- vector()

for (i in 1:num_simulations) {
  state <- 1
  time <- 0
  distant_reappearance <- FALSE
  distant_reappearance_count <- 0
  
  while (state != 5) {
    time <- time + rexp(1, rate = -Q[state,state])
    state <- sample(1:5, 1, prob = Prob[state, ])
    if (time <= 30.5 && (state == 3 || state == 4)) {
      distant_reappearance <- TRUE
    }
  }
  lifetime_distribution[i] <- time
  if (distant_reappearance) {
    distant_reappearance_count <- distant_reappearance_count + 1
  }
}

hist(lifetime_distribution)
conf_interval(0.05,lifetime_distribution)
frac_distant_appearance <- distant_reappearance_count / num_simulations

#### Task 8 ####
library(expm)
Ft_func <- function(p0,Qs,t){
  one_vec <- rep(1,dim(Qs)[2])
  return(1 - p0 %*% expm(Qs*t) %*% one_vec)
}

Qs <- Q[1:4,1:4]
p0 <- c(1,0,0,0)

Ft <- vector()
t_range <- floor(max(lifetime_distribution))
for(t in 1:t_range){
  Ft[t] <- Ft_func(p0,Qs,t)
}

cdf_estimate <- ecdf(lifetime_distribution)

par(mfrow=c(1,2))
plot(cdf_estimate)
plot(Ft,type="l")

par(mfrow = c(1, 1))
plot(cdf_estimate, main = "Empirical vs. Theoretical CDF", xlab = "Lifetime (Months)", ylab = "Cumulative Probability", col = "blue")
lines(1:t, Ft, col = "red", lwd = 2)
legend("bottomright", legend = c("Empirical CDF", "Theoretical CDF"), col = c("blue", "red"), lwd = 2)

F_empirical <- cdf_estimate(1:t_range)

## Statistical tests
t.test(F_empirical, Ft)

# Kolmogorov Smirnov test
n <- length(Ft)
Dn <- max(abs(F_empirical - Ft))
ks_p_val <- (sqrt(n) + 0.12 + 0.11/sqrt(n))*Dn

#### Task 9 ####

S_est <- (N - n_died)/N
Q9 <- matrix(c(0,0.0025,0.00125,0,0.001,
               0,0,0,0.002,0.005,
               0,0,0,0.003,0.005,
               0,0,0,0,0.009,
               0,0,0,0,0),nrow=5, byrow=TRUE)
diag(Q9) <- -rowSums(Q9)

P9 <- matrix(rep(0,25),nrow=5)
for(row in 1:4){
  P9[row,] <- -Q9[row,]/diag(Q9)[row]
}
diag(P9) <- c(0,0,0,0,0)

num_simulations <- 1000
lifetime_distribution_treatment <- vector()
for (i in 1:num_simulations) {
  state <- 1
  time <- 0
  while (state != 5) {
    time <- time + rexp(1, rate = -Q9[state,state])
    state <- sample(1:5, 1, prob = P9[state, ])
  }
  lifetime_distribution_treatment[i] <- time
}

St <- function(t,values){
  N <- length(values)
  dt <- sum(values<t)
  return((N - dt)/N)
}

kaplan_treatment <- vector()
kaplan_no_treatment <- vector()
for(t in 1:t_range){
  kaplan_treatment[t] <- St(t,lifetime_distribution_treatment)
  kaplan_no_treatment[t] <- St(t,lifetime_distribution)
}

par(mfrow = c(1, 1))
plot(kaplan_treatment, main = "Treatment vs no treatment", xlab = "Lifetime (Months)", ylab = "Cumulative Probability", col = "blue", type="l")
lines(1:t, kaplan_no_treatment, col = "red", lwd = 2)
legend("topright", legend = c("With treatment", "Without treatment"), col = c("blue", "red"), lwd = 2)

####################       Part 3       ####################
#### Task 12 ####


Q <- matrix(c(-0.0085, 0.005, 0.0025, 0, 0.001,
              0, -0.014, 0.005, 0.004, 0.005,
              0,0, -0.008, 0.003, 0.005,
              0,0,0,-0.009, 0.009,
              0,0,0,0,0), nrow=5, byrow=TRUE)

num_simulations <- 1000
#lifetime_distribution <- vector()
observed_states <- vector("list", num_simulations)

for (i in 1:num_simulations) {
  state <- 1
  time <- 0
  observed_state <- c(1) # X(0) = 1
  time_spent <- 0
  
  while (state != 5) {
    time_spent <- time_spent + rexp(1, rate = -Q[state, state]) # accumulate time_spent, if time_spent<48
    time <- time + time_spent
    num_states <- 0
    
    if(time_spent>=48){
      time_spent <- time_spent - 48
      while(time_spent >= 0){
        num_states <- num_states + 1
        time_spent <- time_spent - 48
      }
      states <- rep(state, num_states)
      observed_state <- c(observed_state, states)
      time_spent <- 0
    }
    state <- sample(1:5, 1, prob = P9[state, ])
  }
  if(length(observed_state) > 0 && observed_state[length(observed_state)] != 5){
    observed_state <- c(observed_state, 5)
  }
  if(length(observed_state) == 0){
    observed_state <- c(5)
  }
  observed_states[[i]] <- observed_state
  #lifetime_distribution[i] <- time
}

#### Task 13 ####


# Initialize
Q <- matrix(c(-0.0085, 0.005, 0.0025, 0, 0.001,
              0, -0.014, 0.005, 0.004, 0.005,
              0,0, -0.008, 0.003, 0.005,
              0,0,0,-0.009, 0.009,
              0,0,0,0,0), nrow=5, byrow=TRUE)

Y <- observed_states
#lifetime_distribution <- vector()

jump_names <- c("1-1", "1-2", "1-3", "1-4", "1-5",
                "2-1", "2-2", "2-3", "2-4", "2-5",
                "3-1", "3-2", "3-3", "3-4", "3-5",
                "4-1", "4-2", "4-3", "4-4", "4-5",
                "5-1", "5-2", "5-3", "5-4", "5-5")

convergence_threshold <- 1e-3
criterion <- 10
i <- 1
while (criterion >= convergence_threshold) {
  y <- Y[[i]]
  state <- y[1]
  states <- c(state)
  observed_state <- vector()
  num_states <- 0
  time_spent <- 0
  time <- 0
  time_per_state <- rep(0,5)
  valid_sample <- TRUE
  
  N <- matrix(0, nrow = 5, ncol = 5, byrow=TRUE)
  S <- rep(0,5)
  
  # Calculate probability matrix
  P <- matrix(rep(0,25),nrow=5)
  for(row in 1:4){
    P[row,] <- -Q[row,]/diag(Q)[row]
  }
  diag(P) <- c(0,0,0,0,0)
  
  while (state != 5 && valid_sample == TRUE) {
    time_spent <- time_spent + rexp(1, rate = -Q[state, state])
    time <- time + time_spent
    
    time_per_state[state] <- time_per_state[state] + time_spent
    
    # Check states, if time_spent>48
    if(time_spent >= 48){
      time_spent <- time_spent - 48
      while(time_spent >= 0){
        num_states <- num_states + 1
        time_spent <- time_spent - 48
      }
      states <- append(states, rep(state, num_states))
      
      # Check if simulated states are valid
      if(length(states) > length(y)){
        valid_sample <- FALSE
        break
      }
      for(i in 1:length(states)){
        if(y[i] != states[i]){
          valid_sample <- FALSE
          break
        }
      }
      
      # Store the observed states and reset time_spent
      observed_state <- c(observed_state, states)
      time_spent <- 0
    }
    state <- sample(1:5, 1, prob = P[state, ])
  }
  
  if(valid_sample){
    jumps <- paste(y[-length(y)], y[-1], sep = "-")
    jump_count <- table(factor(jumps, levels = jump_names))
    N <- N + matrix(jump_count, nrow = 5, ncol = 5, byrow=TRUE)
    S <- S + time_per_state
    diag(N) <- 0
  
    
    #lifetime_distribution <- append(lifetime_distribution, time)
    
  }
  Q_new <- matrix(0, nrow = 5, ncol = 5)
  for (i in 1:5) {
    if (S[i] != 0) {
      Q_new[i, ] <- N[i, ] / S[i]
    }
  }
  diag(Q_new) <- -rowSums(Q_new)
  criterion <- max(abs(Q-Q_new))
  Q <- Q_new
  i <- i + 1
}





######
get_states <- function(Q, y){
  state <- y[1]
  states <- c(state)
  observed_state <- vector()
  num_states <- 0
  time_spent <- 0
  time <- 0
  time_per_state <- rep(0,5)
  valid_sample <- TRUE
  
  while (state != 5 || length(observed_state) < length(y)) {
    
    if (-Q[state, state] <= 0) {
      return(FALSE)
    }
    
    time_spent <- time_spent + rexp(1, rate = -Q[state, state])
    time <- time + time_spent
    
    if(time_spent < 48){
      time_per_state[state] <- time_per_state[state] + time_spent
      time_spent <- 0
    }
    else{ #(time_spent >= 48){
      time_spent <- time_spent - 48
      while(time_spent >= 0){
        num_states <- num_states + 1
        time_spent <- time_spent - 48
      }
      states <- append(states, rep(state, num_states))
      
      # Check if simulated states are valid
      if(length(states) > (length(y) - length(observed_state))){
        return(FALSE)
      }
      for(i in 1:length(states)){
        if(y[length(observed_state) + i] != states[i]){
          return(FALSE)
        }
      }
      
      # Store the observed states and reset time_spent
      observed_state <- c(observed_state, states)
      time_spent <- 0
    }
    state <- sample(1:5, 1, prob = P[state, ])
  }
  return(list(observed_state, time_per_state))
}

bool <- FALSE
sum <- 0
while(length(bool) <= 1){
  state_test <- get_states(Q,y)
  bool <- state_test[[1]]
  sum <- sum + 1
}
print(sum)





##############################

rates <- c(-0.0085, 0.005, 0.0025, .Machine$double.eps, 0.001,0, -0.014, 0.005, 0.004, 0.005,.Machine$double.eps ,.Machine$double.eps,-0.008, 0.003, 0.005,.Machine$double.eps ,.Machine$double.eps ,.Machine$double.eps, -0.009 ,0.009,.Machine$double.eps, .Machine$double.eps,.Machine$double.eps,.Machine$double.eps,.Machine$double.eps)
Q <- matrix(rates,ncol=5,byrow=TRUE)
n <- 1000
checkin_states_list <-  list()
for (i in 1:n){
  all_states <- 1:5
  cur <- 1
  save_states <- c()
  time_in_state <- c()
  while (cur!=5){
    save_states <- c(save_states,cur)
    time_in_state <- c(time_in_state,rexp(n=1,rate=-P[cur,cur]))
    pt <- -Q[cur,]/Q[cur,cur]
    pt[cur] <- .Machine$double.eps
    state=table(cut(runif(1),c(cumsum(pt))))
    rest_of_states <- all_states[-cur]
    cur <- rest_of_states[as.numeric(which(state==1))]
  }
  cumulative_time <- cumsum(time_in_state)
  total_time <- sum(time_in_state)
  if (total_time <48){
    checkin_states <- c(5)
    checkin_states_list <- append(checkin_states_list,list(checkin_states))
  }
  else {
    checkin_times <- seq(48, total_time, by = 48)
    state_indices <- findInterval(checkin_times, cumulative_time) + 1
    checkin_states <- save_states[state_indices]
    checkin_states <- c(checkin_states,5)
    checkin_states_list <- append(checkin_states_list,list(checkin_states))
    
  }
}


