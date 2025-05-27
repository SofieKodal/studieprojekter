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

update_event_list <- function(event_list,patient_type,ward_type,event_type,arrival_time,stay_type){
  
  if(stay_type==1){
    stay_time <- rexp(1,rate=stay_rate[patient_type])
  }
  else{
    mu <- length_of_stay[patient_type]
    var <- stay_type/mu^2
    log_mu <- log(mu^2/(sqrt(var+mu^2)))
    log_var <- log(var/mu^2 + 1)
    stay_time <- rlnorm(1,meanlog=log_mu, sdlog=sqrt(log_var))
  }
  
  departure_event <- data.frame(time = (arrival_time + stay_time),patient_type = patient_type,ward_type = ward_type,event_type = event_type)
  event_list <- rbind(event_list,departure_event)
  event_list <- event_list[order(event_list$time, event_list$patient_type), ]
  event_list <- event_list[-1,]
  return (event_list)
}



simulate_bedding <- function(total_bed_capacity,reloc_probs,event_list,stay_type){
  ##### Defining control parameters
  #set.seed(1)
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
                                        event_type = "departure",
                                        stay_type = stay_type)
      } 
      else {
        arrived[patient_type] = arrived[patient_type] - 1
        ward <- which(table(cut(runif(1),c(0,cumsum(reloc_probs[patient_type,1:n_types_of_patients]))))==1)
        arrived[ward] = arrived[ward] +1
        free_bed <- (bed_capacity[ward]>0)
        if (free_bed){
          relocated_patients[patient_type] <- relocated_patients[patient_type] + 1
          bed_capacity[ward] <- bed_capacity[ward] - 1
          event_list <- update_event_list(event_list,
                                          patient_type = patient_type,
                                          ward_type = ward,
                                          arrival_time = arrival_time,
                                          event_type = "departure",
                                          stay_type = stay_type)} 
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
arrivals_pr_day <- c(14.5,11,8,6.5,5)
length_of_stay <- c(2.9,4.0,4.5,1.4,3.9)
arrival_rate <- arrivals_pr_day
stay_rate <- 1/length_of_stay
urgency_points <- c(7,5,2,10,5)
n_types_of_patients <- 5
reloc_probs <- c(0,0.05,0.1,0.05,0.8,0,
                 0.2, 0, 0.5, 0.15,0.15,0,
                 0.3,0.2,0,0.2,0.3,0,
                 0.35,0.3,0.05,0,0.3,0,
                 0.2,0.1,0.6,0.1,0,0,
                 0.2,0.2,0.2,0.2,0.2,0)
reloc_probs <- matrix(reloc_probs,ncol=6,byrow=TRUE)
reloc_probs[reloc_probs==0] <- .Machine$double.eps
reloc_probs <- reloc_probs[1:5,1:5]

#### Creating event_list

# initializing parameters
total_bed_capacity <- c(55,40,30,20,20)
#ward_6_relocation_rate = 1

# while (ward_6_relocation_rate > 0.05){
#   print(ward_6_relocation_rate)
#   # simulate bedding allocation given total_bed_capacity
#   event_list <- create_event_list(arrivals_pr_day = arrivals_pr_day,n_types_of_patients = n_types_of_patients,seed=1)
#   sim = simulate_bedding(total_bed_capacity,reloc_probs,event_list)
#   lost_patients = sim[[1]]
#   relocated_patients = sim[[2]]
#   arrived = sim[[3]]
#   ward_6_relocation_rate = (relocated_patients[6]+lost_patients[6])/arrived[6]
#   total_bed_capacity <- add_bed_to_ward_6(urgency_points,relocated_patients,lost_patients,arrived)
# }
# 
# total_bed_capacity
# relocated_patients/arrived
# lost_patients/arrived
# sum(lost_patients)


event_list <- create_event_list(arrivals_pr_day = arrivals_pr_day,n_types_of_patients = n_types_of_patients,seed=1)

stay_types <- c(1,2,3,4)

lost_patients <- vector(mode="list", length=length(stay_types))
relocated_patients <- vector(mode="list", length=length(stay_types))
arrived_patients <- vector(mode="list", length=length(stay_types))


i = 1
for(stay_type in stay_types){
  while (i <= 10){
    print(i)
    res <- vector(mode="list", length=length(stay_types))
    for(stay_type in stay_types){
      print(stay_type)
      res <- simulate_bedding(total_bed_capacity, reloc_probs, event_list, stay_type)
      
      lost_patients[[stay_type]] <- append(lost_patients[[stay_type]], res[[1]])
      relocated_patients[[stay_type]] <- append(relocated_patients[[stay_type]], res[[2]])
      arrived_patients[[stay_type]] <- append(arrived_patients[[stay_type]], res[[3]])
      
      print(lost_patients)
    }
    i = i + 1
  }
}



lost_patients_matrix <- vector(mode="list", length=length(stay_types))
relocated_patients_matrix <- vector(mode="list", length=length(stay_types))
arrived_patients_matrix <- vector(mode="list", length=length(stay_types))

for(stay_type in stay_types){
  lost_patients_matrix[[stay_type]] <- matrix(lost_patients[[stay_type]], ncol=5, byrow=TRUE)
  relocated_patients_matrix[[stay_type]] <- matrix(relocated_patients[[stay_type]], ncol=5, byrow=TRUE)
  arrived_patients_matrix[[stay_type]] <- matrix(arrived_patients[[stay_type]], ncol=5, byrow=TRUE)
}


calculate_confidence_interval <- function(data, conf_level = 0.95) {
  t_test_result <- t.test(data, conf.level = conf_level)
  return(c(mean(data), t_test_result$conf.int))
}

lost_conf <- vector(mode="list", length=length(stay_types))
relocated_conf <- vector(mode="list", length=length(stay_types))
arrived_conf <- vector(mode="list", length=length(stay_types))

for(stay_type in stay_types){
  lost_conf[[stay_type]] <- matrix(rep(0,15),ncol=5)
  relocated_conf[[stay_type]] <- matrix(rep(0,15),ncol=5)
  arrived_conf[[stay_type]] <- matrix(rep(0,15),ncol=5)
  for(i in 1:5){
    lost_conf[[stay_type]][,i] <- calculate_confidence_interval(lost_patients_matrix[[stay_type]][,i])
    relocated_conf[[stay_type]][,i] <- calculate_confidence_interval(relocated_patients_matrix[[stay_type]][,i])
    arrived_conf[[stay_type]][,i] <- calculate_confidence_interval(arrived_patients_matrix[[stay_type]][,i])
  }
}

df_lost <- do.call(rbind, lapply(seq_along(lost_conf), function(i) {
  mat <- lost_conf[[i]]
  data.frame(
    Type = i,
    Class = rep(1:ncol(mat), each = 3),
    Statistic = rep(c("Mean", "Lower", "Upper"), times = ncol(mat)),
    Value = as.vector(mat)
  )
}))

df_reloc <- do.call(rbind, lapply(seq_along(relocated_conf), function(i) {
  mat <- relocated_conf[[i]]
  data.frame(
    Type = i,
    Class = rep(1:ncol(mat), each = 3),
    Statistic = rep(c("Mean", "Lower", "Upper"), times = ncol(mat)),
    Value = as.vector(mat)
  )
}))

df_arrived <- do.call(rbind, lapply(seq_along(arrived_conf), function(i) {
  mat <- arrived_conf[[i]]
  data.frame(
    Type = i,
    Class = rep(1:ncol(mat), each = 3),
    Statistic = rep(c("Mean", "Lower", "Upper"), times = ncol(mat)),
    Value = as.vector(mat)
  )
}))

df_lost$Type <- factor(df_lost$Type)
df_lost$Class <- factor(df_lost$Class)
df_lost$Statistic <- factor(df_lost$Statistic, levels = c("Lower", "Mean", "Upper"))

df_reloc$Type <- factor(df_reloc$Type)
df_reloc$Class <- factor(df_reloc$Class)
df_reloc$Statistic <- factor(df_reloc$Statistic, levels = c("Lower", "Mean", "Upper"))

df_arrived$Type <- factor(df_arrived$Type)
df_arrived$Class <- factor(df_arrived$Class)
df_arrived$Statistic <- factor(df_arrived$Statistic, levels = c("Lower", "Mean", "Upper"))


library(ggplot2)
par(mfrow=c(1,3))
ggplot(df_lost, aes(x = Class, y = Value, color = Type)) +
  geom_errorbar(aes(ymin = ifelse(Statistic == "Lower", Value, NA), ymax = ifelse(Statistic == "Upper", Value, NA)), width = 0.4, position = position_dodge(width = 0.5)) +
  geom_point(aes(y = ifelse(Statistic == "Mean", Value, NA)), size = 2, position = position_dodge(width = 0.5)) +
  geom_line(aes(group = interaction(Type, Class)), position = position_dodge(width = 0.5)) + 
  labs(title = "Lost patients", x = "Patient type", y = "Count") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")

ggplot(df_reloc, aes(x = Class, y = Value, color = Type)) +
  geom_errorbar(aes(ymin = ifelse(Statistic == "Lower", Value, NA), ymax = ifelse(Statistic == "Upper", Value, NA)), width = 0.4, position = position_dodge(width = 0.5)) +
  geom_point(aes(y = ifelse(Statistic == "Mean", Value, NA)), size = 2, position = position_dodge(width = 0.5)) +
  geom_line(aes(group = interaction(Type, Class)), position = position_dodge(width = 0.5)) + 
  labs(title = "Relocated", x = "Patient type", y = "Count") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")

ggplot(df_arrived, aes(x = Class, y = Value, color = Type)) +
  geom_errorbar(aes(ymin = ifelse(Statistic == "Lower", Value, NA), ymax = ifelse(Statistic == "Upper", Value, NA)), width = 0.4, position = position_dodge(width = 0.5)) +
  geom_point(aes(y = ifelse(Statistic == "Mean", Value, NA)), size = 2, position = position_dodge(width = 0.5)) +
  geom_line(aes(group = interaction(Type, Class)), position = position_dodge(width = 0.5)) + 
  labs(title = "Arrived", x = "Patient type", y = "Count") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")



# conf_total_lost <- vector(mode="list", length=length(stay_types))
# for(stay_type in stay_types){
#   lost_per_iteration <- rowSums(lost_patients_matrix[[stay_type]])
#   conf_total_lost[[stay_type]] <- calculate_confidence_interval(lost_per_iteration)
# }


total_lost <- sapply(lost_patients_matrix, function(mat) rowSums(mat))

for(i in 1:4){
  print(calculate_confidence_interval(total_lost[,i]))
  print(calculate_confidence_interval(total_lost[,i])[3] - calculate_confidence_interval(total_lost[,i])[2])
}
conf_total_lost <- sapply(total_lost, calculate_confidence_interval)





