#### Exercise 7

#### 1. Simulated annealing for the travelling salesman ####

# (a)
euclidean_distance <- function(x1, y1, x2, y2) {
  sqrt((x2 - x1)^2 + (y2 - y1)^2)
}

calculate_cost <- function(route,x,y) {
  x <- x[route]
  y <- y[route]
  n <- length(x)
  
  total_cost <- euclidean_distance(x[n], y[n], x[1], y[1])
  for (i in 1:(n-1)) {
    total_cost <- total_cost + euclidean_distance(x[i], y[i], x[i+1], y[i+1])
  }
  return(total_cost)
}


library(purrr)
sim_annealing <- function(x,y,num_simulations){ # n=length of walk
  n <- length(x)
  
  current_route <- sample(n)
  current_cost <- calculate_cost(route, x, y)
  
  best_route <- current_route
  best_cost <- current_cost
  
  for(i in 1:num_simulations){
    k <- 1
    T_k <- 1 / sqrt(1 + k)
    new_route <- current_route
    swap_indices <- sample(1:n, 2)
    new_route[swap_indices[1]] <- current_route[swap_indices[2]]
    new_route[swap_indices[2]] <- current_route[swap_indices[1]]
    
    new_cost <- calculate_cost(new_route, x, y)
    cost_diff <- new_cost - current_cost
    
    if (cost_diff < 0 || runif(1) < exp(-cost_diff / T_k)) {
      current_route <- new_route
      current_cost <- new_cost
    }
    
    if (current_cost < best_cost) {
      best_route <- current_route
      best_cost <- current_cost
    }
    k <- k+1
  }
  return(list(route = best_route, cost = best_cost))
}

x0 <- runif(n,0,10)
y0 <- runif(n,0,10)
num_simulations <- 100
tsp_simulated <- sim_annealing(x0,y0,num_simulations)

best_route <- tsp_simulated$route
best_cost <- tsp_simulated$cost

# Create a data frame for plotting
stations <- data.frame(x = x0, y = y0)
best_route_points <- stations[best_route, ]
best_route_points <- rbind(best_route_points, best_route_points[1, ])  # Close the loop

# Plot the resulting route
ggplot(stations, aes(x = x, y = y)) +
  geom_path(data = best_route_points, aes(group = NULL), color = "blue") +  # Line connecting points
  geom_point(color = "red", size = 3) +  # Points on the route
  geom_text(data = stations, aes(label = 1:n), vjust = -1, hjust = -0.5) +  # Labels for points
  labs(title = "Traveling Salesman Route (Simulated Annealing)",
       x = "X coordinate",
       y = "Y coordinate") +
  coord_fixed() +  # Equal aspect ratio
  theme_minimal()
