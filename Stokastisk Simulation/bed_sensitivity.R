### Making event list
confint_and_mean <- function(lp_for_runs){
  return (c(t.test(lp_for_runs)$conf.int[1],as.numeric(t.test(lp_for_runs)$estimate),t.test(lp_for_runs)$conf.int[2]))}


create_event_list <- function(arrivals_pr_day,n_types_of_patients, stop_time = 365){
  total_patients <- 10000
  arrivals = c()
  for (i in 1:n_types_of_patients){
    arrivals = c(arrivals,cumsum(rexp(total_patients,rate=arrivals_pr_day[i])))
  }
  
  arrivals <- data.frame(time = arrivals,patient_type = rep(1:n_types_of_patients,each=total_patients))
  arrivals["event_type"] <- "arrival"
  arrivals["ward_type"] <- NaN
  
  stop_time <- 365
  
  event_list <- arrivals
  event_list <- event_list[order(event_list$time,event_list$patient_type), ]
  event_list <- event_list[(event_list$time)<stop_time,]
  
  return (event_list)
}



update_event_list <- function(event_list,patient_type,ward_type,event_type,arrival_time){
  
  stay_time <- rexp(1,rate=stay_rate[patient_type])
  departure_event <- data.frame(time = (arrival_time + stay_time),patient_type = patient_type,ward_type = ward_type,event_type = event_type)
  event_list <- rbind(event_list,departure_event)
  event_list <- event_list[order(event_list$time, event_list$patient_type), ]
  event_list <- event_list[-1,]
  return (event_list)
}

simulate_bedding <- function(total_bed_capacity,reloc_probs,event_list){
  ##### Defining control parameters
  patients_processed <- 1
  total_patients <- 2000
  arrived <- rep(0,n_types_of_patients)
  relocated_patients <- rep(0,n_types_of_patients)
  lost_patients <- rep(0,n_types_of_patients)
  bed_capacity <- total_bed_capacity
  
  while (patients_processed < total_patients){
    next_event = head(event_list,1)
    if (next_event$event_type == "arrival") {
      patient_type = next_event$patient_type
      ward = next_event$patient_type
      arrival_time = next_event$time
      arrived[patient_type] = arrived[patient_type] + 1
      free_bed = (bed_capacity[ward]>0)
      
      if (free_bed) {
        bed_capacity[ward] <- bed_capacity[ward] - 1
        event_list <- update_event_list(event_list,
                                        patient_type = patient_type,
                                        ward_type = ward,
                                        arrival_time = arrival_time,
                                        event_type = "departure")
      } 
      else {
        arrived[patient_type] = arrived[patient_type] - 1
        arrived[ward] = arrived[ward] +1
        ward <- which(table(cut(runif(1),c(0,cumsum(reloc_probs[patient_type,1:n_types_of_patients]))))==1)
        free_bed <- (bed_capacity[ward]>0)
        if (free_bed){
          relocated_patients[patient_type] <- relocated_patients[patient_type] + 1
          bed_capacity[ward] <- bed_capacity[ward] - 1
          event_list <- update_event_list(event_list,
                                          patient_type = patient_type,
                                          ward_type = ward,
                                          arrival_time = arrival_time,
                                          event_type = "departure")} 
        else {
          lost_patients[patient_type] <- lost_patients[patient_type] + 1
          event_list <- event_list[-1,]}}} 
    
    if (next_event$event_type == "departure") {
      ward = next_event$ward_type
      bed_capacity[ward] <- min(bed_capacity[ward] + 1,total_bed_capacity[ward])
      event_list <- event_list[-1,]}
    patients_processed <- patients_processed + 1
  }
  
  return (list(lost_patients,relocated_patients,arrived))
}


cooling_scheme <- function(k){
  return (1/sqrt(1+k))
}

simulated_annealing <- function(cooling_scheme) {
  n <- 5
  break_points <- sort(sample(1:(165-1), n-1))
  
  current_bed_distribution <- diff(c(0, break_points, 165))
  new_bed_distribution <- current_bed_distribution
  lp_for_runs <- c()
  for (j in 1:10){
    event_list <- create_event_list(arrivals_pr_day = arrivals_pr_day,n_types_of_patients = n_types_of_patients)
    sim = simulate_bedding(new_bed_distribution,reloc_probs,event_list)
    new_lost_patients <- sum(sim[[1]])
    lp_for_runs <- c(lp_for_runs,new_lost_patients) 
  }
  
  current_lost_patients <- confint_and_mean(lp_for_runs)
  best_lost_patients <- current_lost_patients
  
  temp <- 10
  i <- 1
  while (temp > 0.07) {
    new_bed_distrubution <- current_bed_distribution
    
    swap_indices <- sample(1:n, 2)
    swap_1 <- swap_indices[1]
    swap_2 <- swap_indices[2]
    new_bed_distrubution[swap_1] <- new_bed_distrubution[swap_1]+1
    new_bed_distrubution[swap_2] <- new_bed_distrubution[swap_2]-1
    
    event_list <- create_event_list(arrivals_pr_day = arrivals_pr_day,n_types_of_patients = n_types_of_patients)
    lp_for_runs <- c()
    
    for (j in 1:10){
      event_list <- create_event_list(arrivals_pr_day = arrivals_pr_day,n_types_of_patients = n_types_of_patients)
      sim = simulate_bedding(new_bed_distrubution,reloc_probs,event_list)
      new_lost_patients <- sum(sim[[1]])
      lp_for_runs <- c(lp_for_runs,new_lost_patients) 
    }
    
    new_lost_patients <- confint_and_mean(lp_for_runs)
    
    if (new_lost_patients[2] < current_lost_patients[2] || runif(1) < exp(-(new_lost_patients[2] - current_lost_patients[2]) / temp)) {
      current_bed_distribution <- new_bed_distrubution
      current_lost_patients <- new_lost_patients
      
      if (new_lost_patients[2] < best_lost_patients[2]) {
        best_bed_distribution <- new_bed_distrubution
        best_lost_patients <- new_lost_patients
      }
    }
    
    temp <- cooling_scheme(i)
    i <- i + 1
    print(i)
    print(best_lost_patients)
    print(best_bed_distribution)
    print(temp)
    print("-----------------")
  }
  
  return (list(route = best_bed_distribution, distance = best_lost_patients))
}


#### Defining parameters
arrivals_pr_day <- c(14.5,11,8,6.5,5)
length_of_stay <- c(2.9,4.0,4.5,1.4,3.9)
arrival_rate <- arrivals_pr_day
stay_rate <- 1/length_of_stay
n_types_of_patients <- 5
reloc_probs <- c(0,0.05,0.1,0.05,0.8,
                 0.2, 0, 0.5, 0.15,0.15,
                 0.3,0.2,0,0.2,0.3,
                 0.35,0.3,0.05,0,0.3,
                 0.2,0.1,0.6,0.1,0)
reloc_probs <- matrix(reloc_probs,ncol=5,byrow=TRUE)
reloc_probs[reloc_probs==0] <- .Machine$double.eps
lost_patients_list <- c()
# initializing parameters
simulated_annealing(cooling_scheme)