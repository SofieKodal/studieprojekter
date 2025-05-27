### Exercise 5


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


set.seed(43)
n <- 100
U <- runif(n=n)

#### 1. Crude Monte Carlo ####
X <- exp(U)
conf_inf_est1 <- conf_interval(0.05,X)

####. 2. Antithetic variables ####
Y <- (exp(U) + exp(1-U))/2
conf_inf_est2 <- conf_interval(0.05,Y)

#### 3. Control variable ####
c <- -cov(X,U)/var(U)
Z <- X + c*(U-1/2)
conf_inf_est3 <- conf_interval(0.05,Z)

#### 4. Stratified sampling ####
strata <- 10
W <- vector()
for(i in 1:strata){
  sum_term <- 0
  U_small <- U[((i-1)*10+1):(i*10)]
  print(U_small)
  for(j in 1:strata){
    sum_term <- sum_term + (exp((j-1)/strata + U_small[j]/strata))
  }
  W[i] <- sum_term/strata
}
conf_inf_est4 <- conf_interval(0.05,W)


## Summary
conf_intervals <- matrix(c(conf_inf_est1, conf_inf_est2, conf_inf_est3, conf_inf_est4),
                         nrow=4, ncol=3, byrow=TRUE)
colnames(conf_intervals) = c('Estimated mean', 'Lower', 'Upper')
rownames(conf_intervals) <- c('Crude Monte Carlo', 'Antithetic variables', 'Control variable', 'Stratisfied sampling')

conf_intervals_table <- as.table(conf_intervals)
conf_intervals_table

#### 5. Control variates for exercise 4 ####

# Function to simulate the blocking system table

simulate_blocking_system_control <- function(num_customers, m, mean_service_time, mean_time_between_customers) {
  blocked <- vector()
  num_customers_processed <- 0
  interarrival_time <- vector()
  arrival_time <- 0
  service_end_times <- numeric(m)

  for (i in 1:num_customers) {
    # Time until next arrival
    interarrival_time[i] <- rexp(1, rate = 1/mean_time_between_customers)
    arrival_time <- arrival_time + interarrival_time[i]
    
    # Check if there is an available server
    available_server <- which(service_end_times <= arrival_time)
    
    if (length(available_server) > 0) {
      # Time the service takes
      service_time <- rexp(1, rate = 1/mean_service_time)
      
      # Assign to the first available server
      service_end_times[available_server[1]] <- arrival_time + service_time
      num_customers_processed <- num_customers_processed + 1
      
      blocked[i] <- 0
    } else {
      # No available servers, customer is blocked
      blocked[i] <- 1
    }
  }
  
  X <- blocked
  Y <- interarrival_time
  c <- -cov(X,Y)/var(Y)
  
  Z <- blocked + c*(interarrival_time - mean_time_between_customers)
  fraction_blocked <- sum(blocked) / num_customers
  fraction_Z <- sum(Z) / num_customers
  
  #fraction_blocked <- sum(blocked) / num_customers
  return(list(X=X, Y=Y, Z=Z, c=c, fraction_blocked=fraction_blocked, fraction_Z=fraction_Z))
}

# Parameters
alpha <- 0.05
m <- 10
mean_service_time <- 8
mean_time_between_customers <- 1
num_costumers <- 10000
num_simulations <- 10

# Run simulation and get values for X, Y and c
blocked_sim <- simulate_blocking_system_control(num_costumers, m, mean_service_time, mean_time_between_customers)
X <- blocked_sim$X
Y <- blocked_sim$Y
Y <- blocked_sim$Z
c <- blocked_sim$c
fractions_blocked <- blocked_sim$fraction_blocked
fraction_Z <- blocked_sim$fraction_Z


# Run simulations
blocked_sim_list <- list()
for (i in 1:num_simulations) {
  blocked_sim_list[[i]] <- simulate_blocking_system_control(num_costumers, m, mean_service_time, mean_time_between_customers)
}

# Extract original blockec value and Z value
fractions_blocked <- vector()
fractions_Z <- vector()
for(i in 1:length(blocked_sim_list)){
  fractions_blocked[i] <- blocked_sim_list[[i]]$fraction_blocked
  fractions_Z[i] <- blocked_sim_list[[i]]$fraction_Z
}

conf_orig <- conf_interval(alpha, fractions_blocked)
conf_control <- conf_interval(alpha, fractions_Z)
conf_orig_interval <- conf_orig[3]-conf_orig[2]
conf_control_interval <- conf_control[3]-conf_control[2]

conf_Z <- matrix(round(c(conf_orig, conf_orig_interval, conf_control, conf_control_interval),4),
                         nrow=2, ncol=4, byrow=TRUE)
colnames(conf_Z) <- c('Estimated mean', 'Lower', 'Upper', 'Interval')
rownames(conf_Z) <- c('Original', 'Control variate')
as.table(conf_Z)

abs(var(fractions_blocked)-var(fractions_Z))/var(fractions_Z)

#### 6. Common random numbers. ####

# Function to simulate the blocking system table
simulate_blocking_system6 <- function(U, service_time, num_customers, m, mean_service_time, mean_time_between_customers, inter_distribution) {
  blocked <- vector()
  num_customers_processed <- 0
  arrival_time <- 0
  service_end_times <- numeric(m)
  
  if(inter_distribution=="exp"){
    interarrival_time <- -log(U) / (1/mean_time_between_customers)
  }
  if(inter_distribution=="hyper"){
    interarrival_time <- runif(num_costumers)
    low_ind <- which(interarrival_time<=0.2)
    high_ind <- which(interarrival_time>0.2)
    interarrival_time[low_ind] <- -log(U[low_ind])/5.0
    interarrival_time[high_ind] <- -log(U[high_ind])/0.8333
  }
  
  #service_time <- rexp(num_costumers, 1/mean_service_time)
  #service_time <- -log(U) / (1/mean_service_time)
  
  for (i in 1:num_customers) {
    # Time until next arrival
    arrival_time <- arrival_time + interarrival_time[i]
    
    # Check if there is an available server
    available_server <- which(service_end_times <= arrival_time)
    
    if (length(available_server) > 0) {
      # Assign to the first available server
      service_end_times[available_server[1]] <- arrival_time + service_time[i]
      num_customers_processed <- num_customers_processed + 1
      
      blocked[i] <- 0
    } else {
      # No available servers, customer is blocked
      blocked[i] <- 1
    }
  }
  
  fraction_blocked <- sum(blocked) / num_customers
  return(fraction_blocked)
}
# Parameters
alpha <- 0.05
m <- 10
mean_service_time <- 8
mean_time_between_customers <- 1
num_costumers <- 10000
num_simulations <- 10


# Run using different U
fractions_blocked_exp1 <- numeric(num_simulations)
fractions_blocked_hyper1 <- numeric(num_simulations)

fractions_blocked_exp2 <- numeric(num_simulations)
fractions_blocked_hyper2 <- numeric(num_simulations)

for (i in 1:num_simulations) {
  service_time <- rexp(num_costumers, 1/mean_service_time)
  
  U_common <- runif(num_costumers, min=0, max=1)
  fractions_blocked_exp1[i] <- simulate_blocking_system6(U_common, service_time, num_costumers, m, mean_service_time, mean_time_between_customers, "exp")
  fractions_blocked_hyper1[i] <- simulate_blocking_system6(U_common, service_time, num_costumers, m, mean_service_time, mean_time_between_customers, "hyper")
  
  U1 <- runif(num_costumers, min=0, max=1)
  U2 <- runif(num_costumers, min=0, max=1)
  fractions_blocked_exp2[i] <- simulate_blocking_system6(U1, service_time, num_costumers, m, mean_service_time, mean_time_between_customers, "exp")
  fractions_blocked_hyper2[i] <- simulate_blocking_system6(U2, service_time, num_costumers, m, mean_service_time, mean_time_between_customers, "hyper")
}
fractions_diff_common <- fractions_blocked_exp1 - fractions_blocked_hyper1
conf_diff_common <- conf_interval(alpha, fractions_diff_common)
var_common <- var(fractions_diff_common)

fractions_diff_diff <- fractions_blocked_exp2 - fractions_blocked_hyper2
conf_diff_diff <- conf_interval(alpha, fractions_diff_diff)
var_diff <- var(fractions_diff_diff)

# Create conf table
conf_common <- matrix(round(c(conf_diff_common, conf_diff_diff),4),
                      nrow=2, ncol=3, byrow=TRUE)
colnames(conf_common) <- c('Estimated difference', 'Lower', 'Upper')
rownames(conf_common) <- c('Common U', 'Different U')
as.table(conf_common)

# Create var table
# var_table <- matrix(round(c(var_common, var_diff),8),
#                       nrow=1, ncol=2, byrow=TRUE)
var_table <- matrix(c(var_common, var_diff),
                    nrow=1, ncol=2, byrow=TRUE)
colnames(var_table) <- c('Common U', 'Different U')
rownames(var_table) <- c('Variance of estimated differeces')
as.table(var_table)

(var_common-var_diff)/var_diff

