# Initialize
Q <- matrix(c(0,0.0025,0.00125,0,0.001,
              0,0,0,0.002,0.005,
              0,0,0,0.003,0.005,
              0,0,0,0,0.009,
              0,0,0,0,0),nrow=5, byrow=TRUE)
diag(Q) <- -rowSums(Q)


# #Simulate trajectory
# simulate_trajectory <- function(Q, initial_state, max_time) {
#   current_state <- initial_state
#   times <- seq(0, max_time, by = 48)
#   trajectory <- numeric(length(times))
#   trajectory[1] <- current_state
#   for (t in 2:length(times)) {
#     if (current_state == 5) break
#     rates <- Q[current_state, ]
#     rates[current_state] <- 0
#     rates <- rates / -Q[current_state, current_state]
#     current_state <- sample(1:5, 1, prob = rates)
#     trajectory[t] <- current_state
#   }
#   return(trajectory)
# }
# 
# # Simulate 1000 trajectories
# simulate_all_trajectories <- function(Q, num_trajectories, max_time) {
#   trajectories <- list()
#   for (i in 1:num_trajectories) {
#     trajectories[[i]] <- simulate_trajectory(Q, 1, max_time)
#   }
#   return(trajectories)
# }
# 
# num_trajectories <- 1000
# max_time <- 240 # Simulate for 20 years (0, 48, 96, 144, 192, 240)
# trajectories <- simulate_all_trajectories(Q, num_trajectories, max_time)

simulate_trajectories <- function(Q, num_simulations, observation_interval=48) {
  trajectories <- vector("list", num_simulations)
  
  for (i in 1:num_simulations) {
    state <- 1
    time <- 0
    trajectory <- c(1)  # X(0) = 1
    time_spent <- 0
    
    while (state != 5) {
      time_spent <- time_spent + rexp(1, rate = -Q[state, state])  # accumulate time_spent
      time <- time + time_spent
      num_states <- 0
      
      if (time_spent >= observation_interval) {
        time_spent <- time_spent - observation_interval
        while (time_spent >= 0) {
          num_states <- num_states + 1
          time_spent <- time_spent - observation_interval
        }
        states <- rep(state, num_states)
        trajectory <- c(trajectory, states)
        time_spent <- 0
      }
      
      next_state_probs <- Q[state, ]
      next_state_probs[state] <- 0
      next_state_probs <- next_state_probs / sum(next_state_probs)
      state <- sample(1:5, 1, prob = next_state_probs)
    }
    
    if (length(trajectory) > 0 && trajectory[length(trajectory)] != 5) {
      trajectory <- c(trajectory, 5)
    }
    if (length(trajectory) == 0) {
      trajectory <- c(5)
    }
    
    trajectories[[i]] <- trajectory

  }
  
  return(trajectories)
}

# Example usage
Q <- matrix(c(0,0.0025,0.00125,0,0.001,
              0,0,0,0.002,0.005,
              0,0,0,0.003,0.005,
              0,0,0,0,0.009,
              0,0,0,0,0),nrow=5, byrow=TRUE)
diag(Q) <- -rowSums(Q)

trajectories <- simulate_trajectories(Q,1000)


### Summarize trajectory
######## !!!! this is wrong, we need the time in each state simulated
summarize_trajectories <- function(trajectories) {
  N <- matrix(0, nrow = 5, ncol = 5)
  S <- numeric(5)
  for (trajectory in trajectories) {
    for (t in 2:length(trajectory)) {
      i <- trajectory[t-1]
      j <- trajectory[t]
      if (i != j && i != 5) {
        N[i, j] <- N[i, j] + 1
        S[i] <- S[i] + 48
      }
    }
  }
  return(list(N = N, S = S))
}


# Update Q(k+1)
update_Q <- function(N, S) {
  Q_new <- matrix(0, nrow = 5, ncol = 5)
  for (i in 1:5) {
    for (j in 1:5) {
      if (i != j) {
        Q_new[i, j] <- N[i, j] / S[i]
      }
    }
    Q_new[i, i] <- -sum(Q_new[i, ])
  }
  Q_new[is.na(Q_new)] <- 0
  return(Q_new)
}


# Together
estimate_Q <- function(Q_initial, max_time, threshold=0.001) {
  criterion <- 10
  time <- 0
  Q <- Q_initial
  while (criterion > threshold && time <= max_time) {
    trajectories <- simulate_trajectories(Q,num_simulations)
    summary <- summarize_trajectories(trajectories)
    N <- summary$N
    S <- summary$S
    Q_new <- update_Q(N, S)
    
    criterion <- max(rowSums(abs(Q-Q_new)))
    print(Q)
    Q <- Q_new
    time <- time + 1
  }
  return(Q)
}

Q_estimated <- estimate_Q(Q, 1000, 0.001)
print(Q_estimated)


