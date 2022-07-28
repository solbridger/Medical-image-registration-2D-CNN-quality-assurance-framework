import os 
import glob
import File_Struct as FI
import Image_registration as REG
import metrics
import numpy as np
import nibabel as nib
import datetime
import sys
selection = sys.argv[1:]#This takes parameters form the comand line and can use them to set parameters within the code 

#file to automate the process of image registion

# '''THIS SCRIPT WILL LOOP OVER ALL THE PAIRS OF PATIENTS AND PERFORM A REGISTATION SO THAT WE CAN START CREATING THE DATABASE TO TRAIN THE CNNS. THE CODE ACOMPLISHIS THIS BY TAKING A PATH TO THE PATIENTS AND THEN SAVING THE NAME OF THE PATHS TO EACH OF THE RELEVENT FILES. WE CAN THEN USE THIS TO FEED THE RELEVENT PATHS INTO THE CODE THAT PERFORMS THE REGISTRATION. '''


#defines the path to the patients folder so that we can extract the patient numbers. 
MainPath = '/hepgpu9-data1/callum/DeepRegQA/data/patients/*'
OutPath = '/hepgpu9-data1/sbridger/Auto_Registration/registration_store/new_registration'
#extract the file names of the files in the main path, i.e the numbers of each patient. 

def Setup(Args):
        Out = []
        if len(Args) != 5:
                sys.exit()
        #check if the inputs are integers
        for i in range(0,len(Args)):
                try: 
                        Out.append(int(Args[i]))
                except: 
                        print('one of the inputs is not a integer')
                        sys.exit()
        if (Out[0] < Out[1]) and (Out[2] < Out[3]):
                if Out[1] > 94:
                        Out[1] = 94 
                if Out[3] > 94:
                        Out[3] = 94
                if Out[4]==1: # 5th argument = 1 for odd number pairs
                        OutPath = '/hepgpu9-data1/sbridger/Auto_Registration/registration_store/new_registration/test_even'
                if Out[4]==2: # 5th argument = 2 for even number pairs
                        OutPath = '/hepgpu9-data1/sbridger/Auto_Registration/registration_store/new_registration/train_odd'
                return Out       
        else: 
                sys.exit()

def GetPath(MainPath): 
        patient_list = glob.glob(MainPath)
        Patient_Paths = [] 
        for patient in patient_list:
                study_list = sorted(glob.glob(os.path.join(patient, '*'))) #Each element of this will contain the path to one of the files in the folder
                patient_id = os.path.basename(patient) #this wil contain the patient id for each patient, basename takes the last component of the directory
                #save the file path to the file structure 
                Patient_Paths.append(FI.File_struct(patient_id, study_list))
        print(Patient_Paths)
        return Patient_Paths


def Write_File(Fname, Flocation, Fcontent):
        print('Writing file: {}/{}'.format(Flocation, Fname))
        File = open(Flocation + '/' + Fname, 'w')
        File.write(str(datetime.datetime.now()))
        File.write('\n\n\n')
        for i in range(0, len(Fcontent[0])):
                File.write(str(Fcontent[0][i]) + ': ' + str(Fcontent[1][i]))
                File.write('\n')
        File.close()

#########################################List Functions#########################################

def CompList(a, b):
        return any([i in b for i in a])

def list_sort(a,b,c): #sorts two lists based on a third list.  
        A = [] 
        B = []
        AB = [A,B]
        for k in range (0, len(c)):
                for i in range(0, len(a)):
                        for j in range(0, len(b)):
                                if (c[k] in b[j]) and  (c[k] in a[i]):
                                        A.append(a[i])
                                        B.append(b[j])
        return AB

def Resampling(File_Paths1, File_Paths2, OutPath, Input):
        Base_Structures = ['BRAIN_STEM','SPINAL_CORD','PAROTID_LT','PAROTID_RT']
        Struc_list = list_sort(File_Paths1.Structures, Base_Structures, Base_Structures)
        for i in range(0,len(Struc_list[0])):
                REG.Resample(Struc_list[0][i], File_Paths2.IMG, (OutPath + '_' + Struc_list[1][i] + '.nii.gz'), Input) 
        return None

def Evaluate_Metrics(Structures_0, Structures_1):
                                #check that the files are of the same size before performing the calculations of the similarity metrics. 
                                if metrics.CheckShape(Structures_0, Structures_1):
                                        #print(Structures[1]) 
                                        #print(Structures[0])
                                        Data_names = []
                                        Data_values = []
                                        Data = [Data_names,Data_values]
                                        groundtruth_nii = nib.load(Structures_1)
                                        registration_nii = nib.load(Structures_0)
                                        groundtruth = groundtruth_nii.get_fdata()
                                        registration = registration_nii.get_fdata()
                                        vdim = groundtruth_nii.header.get_zooms()
                                        surface_distance = metrics.surfd(groundtruth, registration, vdim, 2)
                                        #calculate metrics such as dice coefficient
                                        dice = metrics.getDice(groundtruth, registration)
                                        Data_names.append('Dice coefficient')
                                        Data_values.append(dice)
                                        # median surface distance
                                        mediansd = np.median(surface_distance)
                                        Data_names.append('Median surface distance')
                                        Data_values.append(mediansd)
                                        # mean surface distance
                                        msd = np.mean(surface_distance)
                                        Data_names.append('Mean surface distance')
                                        Data_values.append(msd)
                                        # root mean square residual
                                        rms = np.sqrt(((surface_distance - msd)**2).mean())
                                        Data_names.append('RMS distance')
                                        Data_values.append(rms)
                                        #Hausdorff distance
                                        hd  = surface_distance.max()
                                        Data_names.append('Hausdorff distance')
                                        Data_values.append(hd)
                                        # 95th percentile surface distance of the 3D structures
                                        perc = np.percentile(surface_distance, 95)
                                        Data_names.append('95 pecentile surface distance')
                                        Data_values.append(perc)
                                return Data

if __name__ == '__main__':

        Var = Setup(selection)

        #define the parameters for logging information about the running of the code
        Log_name = []
        Log_info = []
        Log = [Log_name, Log_info]


        #log point
        Log_name.append('START_RUN')
        Log_info.append(datetime.datetime.now())
        Log_name.append('MAIN_PATH')
        Log_info.append(MainPath)
        Log_name.append('OUT_PATH')
        Log_info.append(OutPath)


        PF = GetPath(MainPath)
        # perform the pairwise registation of the patients by looping over all possible pairs of patients.  
        odd_even=Var[4]
        range1=np.arange(Var[0], Var[1])
        range2=np.arange(Var[2], Var[3])
        range_idx1, range_idx2=[],[]
        for i in range1:
            for j in range2:
                if (i != j) and (j>i):
                    if odd_even==2:
                        if (i%2==0) and (j%2==0):
                            range_idx1.append(i)
                            range_idx2.append(j)
                    if odd_even==1:
                        if (i%2==1) and (j%2==1):
                            range_idx1.append(i)
                            range_idx2.append(j)
                                                
        for p in range(len(range_idx1)):
                #print("I'M IN!")
                i=range_idx1[p] # Floating number 
                j=range_idx2[p] # Reference number
                print("Now registering pair(s):")
                print("(flo,ref)=", (i,j))
                #log Point 
                Log_name.append('REGISTATION_START_TIME')
                Log_info.append(str(datetime.datetime.now()))
                Log_name.append('REF_ID')
                Log_info.append(PF[j].Pat_no)
                Log_name.append('FLO_ID')
                Log_info.append(PF[i].Pat_no)

                FID = 'Flo_' + PF[i].Pat_no + '_Ref_' + PF[j].Pat_no
                if odd_even==1: # 5th argument = 1 for odd number pairs                                                                                                                              
                        OutPath = '/hepgpu9-data1/sbridger/Auto_Registration/registration_store/new_registration/train_odd'
                if odd_even==2: # 5th argument = 2 for even number pairs                                                                                                                               
                        OutPath = '/hepgpu9-data1/sbridger/Auto_Registration/registration_store/new_registration/test_even'
                Out_Path = OutPath + '/' + FID
                Base_Structures = ['BRAIN_STEM','SPINAL_CORD','PAROTID_LT','PAROTID_RT']
                Date = datetime.datetime.now().strftime ("%d-%m-%Y")
                #create a folder for the images their associated files (if the file exits then it will not try to create it again)
                if not os.path.exists(Out_Path):
                    os.makedirs(Out_Path)			
                    #This will perform the affine refistration		

                    Resampling(PF[i],PF[j],(Out_Path + '/' + FID ),(''))
                    PRE_Structures = []
                    for k in range(0,len(PF[i].Structures)):
                          PRE_Structures.append(Out_Path + '/' + FID + '_' + PF[i].Files[k])							
                    Structures_1 = list_sort(PRE_Structures,PF[j].Structures,Base_Structures)
                    for k in range(0,len(Structures_1[0])):
                        Initial_Metric = Evaluate_Metrics(Structures_1[0][k],Structures_1[1][k])
                        Write_File(Base_Structures[k] + '_PreReg_Metrics.txt',Out_Path,Initial_Metric)



                    Affine = REG.Reg_Affine((PF[i].IMG),(PF[j].IMG),(Out_Path + '/Affine_' + FID),(Out_Path + '/Affine_' + FID + '.txt'))	
                    Log_name.append('CMD_LINE')
                    Log_info.append(Affine)
                    #Register the images present following the affine transformation.
                    Resampling(PF[i],PF[j],(Out_Path + '/Affine_' + FID ),(Out_Path + '/Affine_' + FID + '.txt')) 


                    AFF_Structures = []
                    for k in range(0,len(PF[i].Structures)):
                          AFF_Structures.append(Out_Path + '/Affine_' + FID + '_' + PF[i].Files[k])							
                    Structures_2 = list_sort(AFF_Structures,PF[j].Structures,Base_Structures)
                    for k in range(0,len(Structures_2[0])):			
                        AFF_Metric = Evaluate_Metrics(Structures_2[0][k],Structures_2[1][k])
                        Write_File(Base_Structures[k] + '_Affine_Metrics.txt',Out_Path,AFF_Metric)





                    #perform a deformable registration on the output image
                    Deform = REG.Reg_Deform((PF[i].IMG),(PF[j].IMG),(Out_Path+ '/Def_' + FID + '.nii'),(Out_Path + '/Def_' + FID + '_cpp.nii'),(Out_Path + '/Affine_' + FID + '.txt')	)
                    Log_name.append('CMD_LINE')
                    Log_info.append(Deform)				
                    #Register the images present following the affine transformation.
                    Resampling((PF[i]),(PF[j]),(Out_Path + '/Def_' + FID),(Out_Path + '/Def_' + FID + '_cpp.nii')) 

                    REG.Deformation_field((Out_Path + '/Def_' + FID + '_cpp.nii'),(PF[j].IMG),(Out_Path + '/Def_Vector_Field'))
                    REG.Displacement_field((Out_Path + '/Def_' + FID + '_cpp.nii'),(PF[j].IMG),(Out_Path + '/Disp_Vector_Field'))


                    #calculate the metrics for the registered patients
                    #The First step is to ensure that the two lists are in the correct order
                    #there is a strong possibility that the lists are already in the correct order however it doesnt hurt to be careful with this 
                    #especially since there will not be a easy way to check if it goes wrong.  


                    DEF_Structures = []
                    for k in range(0,len(PF[i].Structures)):
                          DEF_Structures.append(Out_Path + '/Def_' + FID + '_' + PF[i].Files[k])							
                    Structures_3 = list_sort(DEF_Structures,PF[j].Structures,Base_Structures)		
                    for k in range(0,len(Structures_3[0])):				
                        DEF_Metric = Evaluate_Metrics(Structures_3[0][k],Structures_3[1][k])
                        Write_File(Base_Structures[k] + '_Def_Metrics.txt',Out_Path,DEF_Metric)





        Log_name.append('END')
        Log_info.append(datetime.datetime.now())
        Write_File(str(Date) + '_RUNLOG.txt', OutPath, Log)
