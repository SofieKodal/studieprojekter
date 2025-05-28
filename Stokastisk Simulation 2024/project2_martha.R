#### Iteratively adding beds to ward 6
add_bed_to_ward_6 <- function(urgency_points,relocated_patients,lost_patients,arrived){
  relocated_patients <- relocated_patients[1:5]
  arrived <- arrived[1:5]
  lost_patients <- lost_patients[1:5]
  
  bed_value <- (urgency_points/sum(urgency_points)*lost_patients/sum(lost_patients))
  bed_value_normalized <- (bed_value/sum(bed_value))
  steal <- which.min(bed_value_normalized)
  print(paste0("Bed stolen from ward ", steal))
  total_bed_capacity[6] <- total_bed_capacity[6] + 1
  total_bed_capacity[steal] <- total_bed_capacity[steal] - 1 
  return (total_bed_capacity)
}

### Making event list
create_event_list <- function(arrivals_pr_day,n_types_of_patients, stop_time = 365,seed=1){
  total_patients <- 10000
  set.seed(seed)
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
  set.seed(1)
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


#### Defining parameters
arrivals_pr_day <- c(14.5,11,8,6.5,5,13)
length_of_stay <- c(2.9,4.0,4.5,1.4,3.9,2.2)
arrival_rate <- arrivals_pr_day
stay_rate <- 1/length_of_stay
urgency_points <- c(7,5,2,10,5)
n_types_of_patients <- 6
reloc_probs <- c(0,0.05,0.1,0.05,0.8,0,
                 0.2, 0, 0.5, 0.15,0.15,0,
                 0.3,0.2,0,0.2,0.3,0,
                 0.35,0.3,0.05,0,0.3,0,
                 0.2,0.1,0.6,0.1,0,0,
                 0.2,0.2,0.2,0.2,0.2,0)
reloc_probs <- matrix(reloc_probs,ncol=6,byrow=TRUE)
reloc_probs[reloc_probs==0] <- .Machine$double.eps

#### Creating event_list

# initializing parameters
total_bed_capacity <- c(55,40,30,20,20,0)
ward_6_relocation_rate = 1

while (ward_6_relocation_rate > 0.05){
  print(ward_6_relocation_rate)
  # simulate bedding allocation given total_bed_capacity
  event_list <- create_event_list(arrivals_pr_day = arrivals_pr_day,n_types_of_patients = n_types_of_patients,seed=1)
  sim = simulate_bedding(total_bed_capacity,reloc_probs,event_list)
  lost_patients = sim[[1]]
  relocated_patients = sim[[2]]
  arrived = sim[[3]]
  ward_6_relocation_rate = (relocated_patients[6]+lost_patients[6])/arrived[6]
  total_bed_capacity <- add_bed_to_ward_6(urgency_points,relocated_patients,lost_patients,arrived)
}

total_bed_capacity
relocated_patients/arrived
lost_patients/arrived
sum(lost_patients)


event_list <- create_event_list(arrivals_pr_day = arrivals_pr_day,n_types_of_patients = n_types_of_patients,seed=1)
res <- simulate_bedding(total_bed_capacity,reloc_probs,event_list)
lost_patients = res[[1]]
relocated_patients = res[[2]]
arrived = res[[3]]