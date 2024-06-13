library(RIA)

Nifti = load_nifti(filename = "/home/lebrahimpour/IUCPQ-Radiomics/data/Nifti/001-IUCPQ/CT/001-IUCPQ_ct.nii",
                 mask_filename = "/home/lebrahimpour/IUCPQ-Radiomics/data/Nifti/001-IUCPQ/SEG/001-IUCPQ_seg.nii")

#Nifti_first_order = first_order(RIA_data_in = Nifti)
#RIA:::list_to_df(Nifti_first_order$stat_fo$orig)
#Nifti_glcm = glcm(RIA_data_in = Nifti, off_right = 1, off_down = 2, off_z = 2)
#Nifti_glrlm = glrlm(RIA_data_in = Nifti)

#Nifti_all <- radiomics_all(Nifti, bins_in = c(8, 16, 32), equal_prob = "both")
Nifti_all <- radiomics_all(Nifti, bins_in = 25, equal_prob = "both")
save_RIA(Nifti_all, save_to = "/home/lebrahimpour/IUCPQ-Radiomics/deliver/RIA/001-IUCPQ/", save_name = 
"radiomics_RIA", group_name = "Case")
