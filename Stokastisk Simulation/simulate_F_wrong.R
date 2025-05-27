simulate_ward_F <- function(event_list, num_simulations, bed_capacity, reloc_probs, urgency_points) {
  arrived <- rep(0,6)
  departed <- rep(0,6)
  relocated <- rep(0,6)
  lost <- rep(0,6)
  lost_patients <- vector()
  cur_bed_capacity <- bed_capacity
  patient_type_updates <- data.frame(patient_no = NULL, new_patient_type = NULL)
  
  urgency <- urgency_points/sum(urgency_points)
  relocation_rateF <- 1
  i <- 0
  
  #for (i in 1:num_simulations) {
  while(relocation_rateF > 0.05){
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
      if (free_bed) {
        cur_bed_capacity[patient_type] <-  cur_bed_capacity[patient_type] - 1
      }
      # Bed in alternative ward
      else {
        relocated[patient_type] <- relocated[patient_type] + 1
        alternative_bed <- sample(1:n_patient_types, size=1, prob = reloc_probs[patient_type, ])
        if (cur_bed_capacity[alternative_bed] > 0) {
          cur_bed_capacity[alternative_bed] <- cur_bed_capacity[alternative_bed] - 1
          
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
    if(sum(lost) != 0){
      importance <- urgency * lost/sum(lost)
      bed_to_steal <- which.min(importance)
      if(bed_capacity[bed_to_steal] > 0){
        bed_capacity[bed_to_steal] <- bed_capacity[bed_to_steal] - 1
        bed_capacity[6] <- bed_capacity[6] + 1
      }
    }
    relocation_rateF <- relocated[6]/arrived[6]
    i <- i + 1
  }
  
  return(list(bed_capacity = bed_capacity, cur_bed_capacity = cur_bed_capacity))
}