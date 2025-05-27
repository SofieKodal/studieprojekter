### Exercise 4
library(EnvStats)

#### Functions ####

# Theoretical solution
erlang <- function(lambda,s,m){
  sum_term <- 0
  A <- lambda * s
  for(i in 1:m){
    sum_term <- sum_term +(A^i / factorial(i))
  }
  B <- (A^m / factorial(m)) / sum_term
  return(B)
}

# Function to simulate the blocking system
simulate_blocking_system <- function(num_costumers, m, mean_service_time, mean_time_between_customers, inter_distribution, service_distribution) {
  blocked <- 0
  num_costumers_processed <- 0
  arrival_time <- 0
  service_end_times <- numeric(m)
  
  for (i in 1:num_costumers) {
    # Time until next arrival
    if(inter_distribution=="exp"){
      interarrival_time <- rexp(1, rate = 1/mean_time_between_customers)
    }
    if(inter_distribution=="erlang"){
      interarrival_time <- rgamma(1, shape=1, rate=1/mean_time_between_customers)
    }
     if(inter_distribution=="hyper"){
       interarrival_time=runif(num_costumers)
       low_ind=which(interarrival_time<=0.2)
       high_ind=which(interarrival_time>0.2)
       interarrival_time[low_ind]=-log(U[low_ind])/5.0
       interarrival_time[high_ind]=-log(U[high_ind])/0.8333
       #interarrival_time <- rexp(1, rate=0.8333)*0.8 + rexp(1, rate=5.0)*0.2
     }
    arrival_time <- arrival_time + interarrival_time
    
    # Check if there is an available server
    available_server <- which(service_end_times <= arrival_time)
    
    if (length(available_server) > 0) {
      if(service_distribution=="exp"){
        service_time <- rexp(1, rate = 1/mean_service_time)
      }
      if(service_distribution=="constant"){
        service_time <- mean_service_time
      }
      if(service_distribution=="pareto1.05"){
        k <- 1.05
        beta <- mean_service_time*(k-1)/k
        service_time <- rpareto(1, location=beta, shape=1.05)
      }
      if(service_distribution=="pareto2.05"){
        k <- 2.05
        beta <- mean_service_time*(k-1)/k
        service_time <- rpareto(1, location=beta, shape=2.05)
      }
      
      # Assign to the first available server
      service_end_times[available_server[1]] <- arrival_time + service_time
      num_costumers_processed <- num_costumers_processed + 1
    } else {
      # No available servers, customer is blocked
      blocked <- blocked + 1
    }
  }
  
  fraction_blocked <- blocked / num_costumers
  return(fraction_blocked)
}

conf_interval <- function(alpha, fractions_blocked){
  n = length(fractions_blocked)
  mu_est = mean(fractions_blocked)
  s_theta_sq = abs(1/(n-1)*(sum(fractions_blocked^2)-n*mu_est^2))
  s_theta = sqrt(s_theta_sq)
  conf_int = c(mu_est + s_theta/sqrt(n)*qt(alpha/2,n-1),mu_est + s_theta/sqrt(n)*qt(1-alpha/2,n-1))
  mu_true <- erlang(lambda=mean_time_between_customers, s=mean_service_time, m=m)
  
  cat("mu_est", round(mu_est,4), "\n")
  cat("conf_int", round(conf_int,4), "\n")
  cat("mu_erlang", round(mu_true,4), "\n")
}

#### Parameters ####

set.seed(42)  # For reproducibility

# Parameters
alpha <- 0.05
m <- 10
mean_service_time <- 8
mean_time_between_customers <- 1
num_costumers <- 10000
num_simulations <- 10


#### 1. Fraction of blocked costumers #### √
fractions_blocked <- numeric(num_simulations)
inter_distribution="exp"
service_distribution="exp"
for (i in 1:num_simulations) {
  fractions_blocked[i] <- simulate_blocking_system(num_costumers, m, mean_service_time, mean_time_between_customers, inter_distribution, service_distribution)
}
conf_interval(alpha, fractions_blocked)

#### 2. Different interarrival_time distributions ####

# (a) Erlang √
fractions_blocked <- numeric(num_simulations)
inter_distribution="erlang"
service_distribution="exp"
for (i in 1:num_simulations) {
  fractions_blocked[i] <- simulate_blocking_system(num_costumers, m, mean_service_time, mean_time_between_customers, inter_distribution, service_distribution)
}
conf_interval(alpha, fractions_blocked)

# (b) Hyper exponential (√ ca.)
fractions_blocked <- numeric(num_simulations)
inter_distribution="hyper"
service_distribution="exp"
for (i in 1:num_simulations) {
  fractions_blocked[i] <- simulate_blocking_system(num_costumers, m, mean_service_time, mean_time_between_customers, inter_distribution, service_distribution)
}
conf_interval(alpha, fractions_blocked)

#### 3. Different service_time distributions ####

# (a) Constant √
fractions_blocked <- numeric(num_simulations)
inter_distribution="exp"
service_distribution="constant"
for (i in 1:num_simulations) {
  fractions_blocked[i] <- simulate_blocking_system(num_costumers, m, mean_service_time, mean_time_between_customers, inter_distribution, service_distribution)
}
conf_interval(alpha, num_simulations, fractions_blocked)

# (b) Pareto 1.05 (nope)
fractions_blocked <- numeric(num_simulations)
inter_distribution="exp"
service_distribution="pareto1.05"
for (i in 1:num_simulations) {
  fractions_blocked[i] <- simulate_blocking_system(num_costumers, m, mean_service_time, mean_time_between_customers, inter_distribution, service_distribution)
}
conf_interval(alpha, fractions_blocked)


# (b) Pareto 2.05 √
fractions_blocked <- numeric(num_simulations)
inter_distribution="exp"
service_distribution="pareto2.05"
for (i in 1:num_simulations) {
  fractions_blocked[i] <- simulate_blocking_system(num_costumers, m, mean_service_time, mean_time_between_customers, inter_distribution, service_distribution)
}
conf_interval(alpha, fractions_blocked)





