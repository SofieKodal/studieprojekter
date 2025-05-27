#### Exercise 3

# m service units
# no waiting room
# A = mean(arrival_rate) * mean(service_time)

#### 1. Poisson process modelling ####

# theta_bar <- vector()
# for(i in 1:m){
#   theta_hat <- vector()
#     for(j in 1:n){
#       # accumulated times between arrivals to get arrival times
#       arrival_time <- cumsum(rpois(k, lambda=time_between_cost))
#       service_time <- rexp(k, rate=1/mean_service_time)
#       departure_time <- arrival_time + service_time
#       
#       times <- c(arrival_time, departure_time)
#       
#       # Create a corresponding vector of names
#       boolean <- c(rep(1, length(arrival_time)), rep(-1, length(departure_time)))
#       
#       # Order the combined values and apply that order to the names
#       sorted_order <- order(times)
#       arr_or_dep <- boolean[sorted_order]
#       
#       num_costumers <- vector()
#       num_costumers[1] <- arr_or_dep[1]
#       blocked <- 0
#       for(h in 2:k){
#         num_costumers[h] <- num_costumers[h-1] + arr_or_dep[h]
#         if(num_costumers[h] > max_services){
#           num_costumers[h] <- max_services
#           blocked <- blocked + 1
#         }
#       }
#     
#       theta_hat[j] <- blocked/k
#     }
#   theta_bar[i] <- sum(theta_hat)/n
# }

k <- 10000 # costumers per for loop
n <- 10 # number of loop loop
m <- 10 # number of simulations

mean_service_time <- 8
time_between_cost <- 1

max_costumers <- 10
blocked <- 0
num_costumers <- 0
total_costumers <- 0

arrival_time <- 0
departure_time <- vector()

k <- 100




while(total_costumers < k){
  new_arrival <- rpois(1, lambda=time_between_cost)
  arrival_time <- arrival_time + new_arrival
  
  new_depature <- arrival_time + rexp(1, rate=1/mean_service_time)
  departure_time <- c(departure_time, new_depature)
  
  min_dep <- which.min(departure_time)
  
  # Store is full and new costumer arrives
  if(arrival_time < min(departure_time) && num_costumers >= max_costumers){
    num_costumers <- max_costumers
    
    # reject costumer
    arrival_time <- arrival_time - new_arrival # remove newest arrival
    departure_time <- departure_time[-length[departure_time]] # remove new_departure
    blocked <- blocked + 1
  }
  
  # One leaves
  if(arrival_time > min(departure_time) && num_costumers >= max_costumers){
    num_costumers <- num_costumers - 1
    departure_time <- departure_time[-min_dep] # remove min depature_time
  }
  
  # One arrives and there is space
  if(arrival_time < min(departure_time) && num_costumers < max_costumers){
    num_costumers <- num_costumers + 1
    
    #new_arrival <- rpois(1, lambda=time_between_cost)
    #arrival_time <- arrival_time + new_arrival
    
    #new_departure <- arrival_time + rexp(1, rate=1/mean_service_time)
    #departure_time <- c(departure_time, new_departure)
  }
  
  # if(arrival_time > min(departure_time) && num_costumers < max_costumers){
  #   num_costumers <- num_costumers + 1
  #   new_departure <- arrival_time + rexp(1, rate=1/mean_service_time)
  #   departure_time <- c(departure_time, new_departure)
  #}
  total_costumers <- total_costumers + 1
}

print(blocked)

