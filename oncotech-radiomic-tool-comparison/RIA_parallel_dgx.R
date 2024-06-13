folder <- "/home/lebrahimpour/IUCPQ-Radiomics/data/Nifti" #Location of folder containing individual folders per patient which contain nifti files for the image and m$

out <- "/home/lebrahimpour/IUCPQ-Radiomics/deliver/RIA/out_test_parallel_bincount8/" #Location of folder where the results will be dumped

patients <- list.dirs(folder, recursive = FALSE, full.names = FALSE) #Name of patient folders

#print(patients)
patients_full <- list.dirs(folder, recursive = FALSE, full.names = TRUE) #Name of patient folders with full file path name
#print(patients_full)
#print(list.files(patients_full[1]))

library(foreach); library(doParallel) #Load required packages
doParallel::registerDoParallel(200) #Define how many threads to use, usually use the number of threads-1

#Use parallelized for cycle to cycle through all the patients
data_out_paral <- foreach (i = 1:length(patients), .combine="rbind", .inorder=FALSE,
                           .packages=c('RIA'), .errorhandling = c("pass"), .verbose=FALSE) %dopar% {
                             
                             file_image <- list.files(file.path(patients_full[i], '/CT')) #Name of the CT image file in the current patient folder
                             file_mask <- list.files(file.path(patients_full[i], '/SEG')) #Name of the mask file in the current patient folder
                             #                             image <- grep("IUCPQ_ct.nii", files, ignore.case = T, value = T) #Full name of the image file
                             #                             masks <- grep("IUCPQ_seg.nii", files, ignore.case = T, value = T) #Full name of the mask files
                             
                             #RUN RIA
                             IMAGE <- RIA::load_nifti(filename = paste0(patients_full[i], "/CT/", file_image),
                                                      mask_filename = paste0(patients_full[i], "/SEG/", file_mask), switch_z = FALSE)
                             #                             IMAGE <- RIA::first_order(IMAGE)
                             IMAGE <- RIA::radiomics_all(IMAGE,  equal_prob = FALSE , bins_in= 8) #Calculate radiomic features
                             RIA::save_RIA(IMAGE, save_to = out, save_name = patients[i], group_name = patients[i]) #Export results into csv
                           }
