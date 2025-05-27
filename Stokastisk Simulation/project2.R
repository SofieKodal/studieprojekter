## Project 2

#### Parameters ####
bed_capacity <- c(55,40,30,20,20)
arrivals_pr_day <- c(14.5,11,8,6.5,5,13)
length_of_stay <- c(2.9,4.0,4.5,1.4,3.9,2.2)
arrival_rate <- arrivals_pr_day
stay_rate <- 1/length_of_stay
urgency_points <- c(7,5,2,10,5)

reloc_probs <- c(0,0.05,0.1,0.05,0.8,0,
                 0.2, 0, 0.5, 0.15,0.15,0,
                 0.3,0.2,0,0.2,0.3,0,
                 0.35,0.3,0.05,0,0.3,0,
                 0.2,0.1,0.6,0.1,0,0,
                 0.2,0.2,0.2,0.2,0.2,0)
reloc_probs <- matrix(reloc_probs,ncol=6,byrow=TRUE)
reloc_probs_small <- reloc_probs[1:5,1:5]
n_types_of_patients <- 5

#### 1. Simulation of patient flow ####

#### Event list
total_patients <- 10000
event_queue = list()
stop_time <- 365
n_types_of_patients <- 5

#set.seed(1)
arr_time1 = cumsum(rexp(total_patients,rate=arrivals_pr_day[1]))
arr_time2 = cumsum(rexp(total_patients,rate=arrivals_pr_day[2]))
arr_time3 = cumsum(rexp(total_patients,rate=arrivals_pr_day[3]))
arr_time4 = cumsum(rexp(total_patients,rate=arrivals_pr_day[4]))
arr_time5 = cumsum(rexp(total_patients,rate=arrivals_pr_day[5]))

dept_time1 = arr_time1 + rexp(total_patients,rate=stay_rate[1])
dept_time2 = arr_time2 + rexp(total_patients,rate=stay_rate[2])
dept_time3 = arr_time3 + rexp(total_patients,rate=stay_rate[3])
dept_time4 = arr_time4 + rexp(total_patients,rate=stay_rate[4])
dept_time5 = arr_time5 + rexp(total_patients,rate=stay_rate[5])


arrivals <- data.frame(time = c(arr_time1,arr_time2,arr_time3,arr_time4,arr_time5),patient_type = rep(1:n_types_of_patients,each=total_patients), patient_no=1:(total_patients*n_types_of_patients))
arrivals["type"] <- "arrival"

departures <- data.frame(time = c(dept_time1,dept_time2,dept_time3,dept_time4,dept_time5),patient_type = rep(1:n_types_of_patients,each=total_patients), patient_no=1:(total_patients*n_types_of_patients))
departures["type"] <- "departure"

event_list <- rbind(arrivals,departures)
event_list <- event_list[order(event_list$time), ]
event_list <- event_list[event_list$time < stop_time,]

#### Simulation

simulate_patient_flow <- function(event_list, num_simulations, bed_capacity, reloc_probs, n_patient_types) {
  arrived <- rep(0,n_patient_types)
  departed <- rep(0,n_patient_types)
  relocated <- rep(0,n_patient_types)
  located_in_own_ward <- rep(0,n_patient_types) # patients getting a bed in their own ward
  relocated_success <- rep(0,n_patient_types) # patients getting a bed in a different ward
  relocated_bed <- rep(0,n_patient_types) # beds taken by relocated patients
  lost <- rep(0,n_patient_types)
  lost_patients <- vector()
  cur_bed_capacity <- bed_capacity
  patient_type_updates <- data.frame(patient_no = NULL, new_patient_type = NULL)
  
  for (i in 1:num_simulations) {
    event <- event_list[i, ]
    
    # Check if patient is lost
    if (event$patient_no %in% lost_patients) {
      next
    }
    
    patient_type <- event$patient_type
    
    # Arrival
    if (event$type == "arrival") {
      arrived[patient_type] <- arrived[patient_type] + 1
      
      # Bed in own ward
      free_bed <- cur_bed_capacity[patient_type] > 0
      if (free_bed & cur_bed_capacity[patient_type] < bed_capacity[patient_type]) {
        cur_bed_capacity[patient_type] <-  cur_bed_capacity[patient_type] - 1
        located_in_own_ward[patient_type] <- located_in_own_ward[patient_type] + 1
      }
      # Bed in alternative ward
      else {
        relocated[patient_type] <- relocated[patient_type] + 1
        alternative_bed <- sample(1:n_patient_types, size=1, prob = reloc_probs[patient_type, ])
        if (cur_bed_capacity[alternative_bed] > 0 & cur_bed_capacity[alternative_bed] < bed_capacity[alternative_bed]) {
          cur_bed_capacity[alternative_bed] <- cur_bed_capacity[alternative_bed] - 1
          relocated_success[patient_type] <- relocated_success[patient_type] + 1
          relocated_bed[alternative_bed] <- relocated_bed[alternative_bed] + 1
          
          # Update patient type
          patient_type_update <- data.frame(patient_no = event$patient_no, new_patient_type = alternative_bed)
          patient_type_updates <- rbind(patient_type_updates, patient_type_update)
        } else {
          lost[patient_type] <- lost[patient_type] + 1
          lost_patients <- c(lost_patients, event$patient_no)
        }
      }
    }
    
    # Departure
    if (event$type == "departure") {
      departed[patient_type] <- departed[patient_type] + 1
      
      # Update patient type, if they got an alterntive bed
      if (event$patient_no %in% patient_type_updates$patient_no) {
        patient_type <- patient_type_updates[patient_type_updates$patient_no == event$patient_no, "new_patient_type"]
      }
      
      cur_bed_capacity[patient_type] <- cur_bed_capacity[patient_type] + 1
    }
  }
  
  return(list(arrived = arrived, departed = departed, relocated = relocated,
              located_in_own_ward = located_in_own_ward, relocated_success = relocated_success, relocated_bed = relocated_bed, 
              lost = lost, cur_bed_capacity = cur_bed_capacity))
}

n_patient_types <- 5
num_simulations_burn_in <- 2000
num_simulations <- nrow(event_list) - num_simulations_burn_in - 4000

burn_in <- simulate_patient_flow(event_list, num_simulations_burn_in, bed_capacity, reloc_probs_small,n_patient_types) 
patient_flow <- simulate_patient_flow(event_list[(num_simulations_burn_in+1):nrow(event_list),], num_simulations, burn_in$cur_bed_capacity, reloc_probs_small, n_patient_types) 

arrived <- patient_flow$arrived
departed <- patient_flow$departed

lost <- patient_flow$lost
percent_lost <- lost/arrived

relocated_succes <- patient_flow$relocated_success

relocated_bed <- patient_flow$relocated_bed
relocated_bed_relative <- relocated_bed/bed_capacity

print(percent_lost)
print(lost)

sum(lost)/num_simulations


#### 2. New ward F ####


#### Event list
total_patients <- 10000
event_queue = list()
stop_time <- 365
n_types_of_patients <- 6

set.seed(1)
arr_time1 = cumsum(rexp(total_patients,rate=arrivals_pr_day[1]))
arr_time2 = cumsum(rexp(total_patients,rate=arrivals_pr_day[2]))
arr_time3 = cumsum(rexp(total_patients,rate=arrivals_pr_day[3]))
arr_time4 = cumsum(rexp(total_patients,rate=arrivals_pr_day[4]))
arr_time5 = cumsum(rexp(total_patients,rate=arrivals_pr_day[5]))
arr_time6 = cumsum(rexp(total_patients,rate=arrivals_pr_day[6]))

dept_time1 = arr_time1 + rexp(total_patients,rate=stay_rate[1])
dept_time2 = arr_time2 + rexp(total_patients,rate=stay_rate[2])
dept_time3 = arr_time3 + rexp(total_patients,rate=stay_rate[3])
dept_time4 = arr_time4 + rexp(total_patients,rate=stay_rate[4])
dept_time5 = arr_time5 + rexp(total_patients,rate=stay_rate[5])
dept_time6 = arr_time6 + rexp(total_patients,rate=stay_rate[6])


arrivalsF <- data.frame(time = c(arr_time1,arr_time2,arr_time3,arr_time4,arr_time5,arr_time6),patient_type = rep(1:n_types_of_patients,each=total_patients), patient_no=1:(total_patients*n_types_of_patients))
arrivalsF["type"] <- "arrival"

departuresF <- data.frame(time = c(dept_time1,dept_time2,dept_time3,dept_time4,dept_time5,dept_time6),patient_type = rep(1:n_types_of_patients,each=total_patients), patient_no=1:(total_patients*n_types_of_patients))
departuresF["type"] <- "departure"

event_listF <- rbind(arrivalsF,departuresF)
event_listF <- event_listF[order(event_listF$time), ]
event_listF <- event_listF[event_listF$time < stop_time,]

# High low urgency and low number of lost patients gives a low importance to the wards

simulate_ward_F <- function(event_list, num_simulations_burn_in, num_simulations, bed_capacity, reloc_probs, n_patient_types, urgency_points) {
  
  urgency <- urgency_points/sum(urgency_points)
  relocation_rateF <- 1
  
  while(relocation_rateF > 0.05){
    
    print(bed_capacity)
    patient_flow <- simulate_patient_flow(event_list, num_simulations, bed_capacity, reloc_probs, n_patient_types)
    
    lost <- patient_flow$lost
    print(lost/sum(lost))
    arrived <- patient_flow$arrived
    relocated <- patient_flow$relocated
    
    importance <- (urgency * lost/sum(lost))[1:5]
    bed_to_steal <- which.min(importance)
    
    if(bed_capacity[bed_to_steal] > 0){
      bed_capacity[bed_to_steal] <- bed_capacity[bed_to_steal] - 1
      bed_capacity[6] <- bed_capacity[6] + 1
    }
    relocation_rateF <- relocated[6]/arrived[6]
  }
  return(list(bed_capacity = bed_capacity, relocation_rateF = relocation_rateF))
}

bed_capacity_F <- c(55,40,30,20,20,0)
num_simulations_burn_in <- 2000
num_simulations <- nrow(event_listF) - num_simulations_burn_in - 4000
n_patient_types <- 6
urgency_points <- c(7,5,2,10,5,0)


simulated_bed_capacity <- simulate_ward_F(event_listF, num_simulations_burn_in, num_simulations, bed_capacity_F, reloc_probs, n_patient_types, urgency_points)





