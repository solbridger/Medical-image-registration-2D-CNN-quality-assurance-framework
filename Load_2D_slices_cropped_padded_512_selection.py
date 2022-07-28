import tensorflow
#print(tensorflow.version.VERSION)
import sys, os
# Important to assign the processing to the GPU with the most vRAM available
GPU = 1
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU) #"0"
physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
config = tensorflow.compat.v1.ConfigProto(device_count = {'GPU': GPU})
#Important to run this in order to not overload the GPU
config = tensorflow.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction=0.23 # don't hog all vRAM
#config.operation_timeout_in_ms=15000   # terminate on long hangs
#config.gpu_options.allow_growth = True
sess = tensorflow.compat.v1.InteractiveSession("", config=config)

import glob
import numpy as np
import nibabel as nib
import pandas as pd
import importlib
# import prepare_all_inputs_cropped_xyz_3D_512 as pic
import dice_and_sds_2d_from_array
import random
import time

selection = sys.argv[1:]

def Setup(Args, n):
        Out = []
        if len(Args) != 3:
                sys.exit()
        #check if the inputs are integers
        for i in range(0,len(Args)):
                try: 
                        Out.append(int(Args[i]))
                except: 
                        print('one of the inputs is not a integer')
                        sys.exit()
        if (Out[0] < Out[1]):
                if Out[1] > n:
                        Out[1] = n
                if Out[2]==1: # 3rd argument = 1 for odd number pairs
                        train_test = 'train_odd'
                if Out[2]==2: # 3rd argument = 2 for even number pairs
                        train_test = 'test_even'
                return Out, train_test       
        else: 
                sys.exit()
                
train_test = Setup(selection, 2200)[1] # 2200 is arbitrary, just any number > number of registrations (2162 possible) will work
# print("train_test: ", train_test)
# 'BRAIN_STEM', 'SPINAL_CORD', 'PAROTID_RT', 'PAROTID_LT'
organ = 'PAROTID_LT'
machine = 'hepgpu9'
# array_path = '/hepgpu9-data1/sbridger/mphys_2d_network_2021-22/slice_arrays/' + organ + '/'

# if(machine == 'hepgpu9'):
patient_filepath = '/hepgpu9-data1/sbridger/94_patient_data/patient_filepath/*'
registration_filepath = '/hepgpu9-data1/sbridger/Auto_Registration/registration_store/new_registration/' + train_test + '/' 
single_patient_filepath = '/hepgpu9-data1/sbridger/94_patient_data/patient_filepath/'

# path_size = os.path.getsize(registration_filepath)
# print(path_size)
# print(patient_filepath)
# print(registration_filepath)
# print(single_patient_filepath)

registration_list = glob.glob(registration_filepath+'*/')  #Get all reference image paths
registration_number_list = [i.split(train_test+'/',1)[1] for i in registration_list]  #Extract the numbers
flo_ref_number_list = [i.split(train_test+'/Flo_',1)[1] for i in registration_list]
flo_number_list = [i.split('_',1)[0] for i in flo_ref_number_list]
ref_number_list0 = [i.split('Ref_',1)[1] for i in flo_ref_number_list]
ref_number_list = [i.split('/',1)[0] for i in ref_number_list0]

n_reg=len(registration_number_list)

Var = Setup(selection, n_reg)[0]
# print("Var: ",Var)
del ref_number_list0, flo_ref_number_list, registration_number_list, registration_list
#Note: for the moment we are only using files in which the x- and y-dimensions are 512. This is being checked for in the
#prepare inputs cropped file by raising a ValueError which is catched in the while loop block below upon calling the
#prepareMetricsArrays function.

global_start = time.time() # Variable to then measure the total time taken to run this whole block of code
# i = 0 #Variable to loop over the possible patient pairings

#If you want to get rid of slices with Dice Score of 0, then set zero_bool = False, otherwise zero_bool = True
zero_bool = True

#If you want to get rid of slices in which either (or both) def and ref images have no organ contour, then False
mid_zero_bool = False

#Choose slicing direction (0:x, 1:y, 2:z)
n = 2

slice_wrong_shape = 0 #Number of times the slice x and y dimensions were not 512
contour_wrong_shape = 0 #Number of times the contour x and y dimensions were not 512
corrupted = 0 #Number of times a corrupted nifty file is encountered

full_zeros = [] #Keep track of slices which are all zero
#Shape of full_zeros elements is [flo,ref,slice index], where
#slice index is counted from the very bottom of the scans
# reg_track_filename = os.path.join(array_path, 'registration_track')
# registration_track = np.memmap(reg_track_filename, dtype='float16', mode='w+')
registration_track = [] #Keep track of how many slices are obtained for each successful registration
#Each element's form is [flo,ref,#slices with metric,#total slices]
#Note the length of this array gives the total number of successful registrations

image_array_list = []
dice_list = []
hd_95_list = []
hd_75_list = []
hd_50_list = []
msd_list = []
rms_list = []
# msd_3D_list = []
ref_contour_slice_list = []

indexes = str(Var[0])+'-'+str(Var[1]-1)

pic = importlib.import_module('prepare_all_inputs_cropped_xyz_3D_512_'+train_test)
i = Var[0]
while Var[0] <= i < Var[1]: #n_reg
    start = time.time()
    print(i)
    #print the current combination
    flo_no = flo_number_list[i]
    ref_no = ref_number_list[i]
    print('Floating patient number: ' + flo_no + '     Reference patient number: ' + ref_no)
    
    if os.path.isdir(registration_filepath + 'Flo_' + flo_no + '_Ref_' + ref_no) and \
       os.path.isfile(single_patient_filepath + flo_no + '/' + organ +'.nii.gz') and \
       os.path.isfile(single_patient_filepath + ref_no + '/' + organ +'.nii.gz') and \
       os.path.isfile(single_patient_filepath+ref_no+'/NoBackground_0522c0'+ref_no+'.nii.gz') and \
       os.path.isfile(registration_filepath + 'Flo_'+flo_no+'_Ref_'+ref_no+'/Disp_Vector_Field.nii') and \
       os.path.isfile(registration_filepath + 'Flo_'+flo_no+'_Ref_'+ref_no+'/Def_Flo_'+flo_no+'_Ref_'+ref_no+'.nii'):
        print('All relevant files exist.')

        try:
            #Process the metrics arrays
            dice_array, hd_95_array, hd_75_array, hd_50_array, rms_array, msd_array, min_slice_index, max_slice_index, zero_slices, mid_zero_slices, ref_contour_slice_array = pic.prepareMetricsArrays(zero_bool, mid_zero_bool, flo_no, ref_no, machine, organ, n)
#             ref_contour_slice_array = np.reshape(ref_contour_slice_array, (len(ref_contour_slice_array),512,512))
                
            if not zero_bool:
                print("Slices with zero Dice Score: ", zero_slices)
            if not mid_zero_bool:
                print("Slices with no contour on either (or both) images: ", mid_zero_slices)
#             if min_slice_index <= max_slice_index:
#                 ref_contour_slice_list.append(ref_contour_slice_array)
                
        except ValueError:
            #The ValueError is raised from within the prepare input cropped (pic) python file
            print("The shape of the contour images was not 512x512xZ.")
            contour_wrong_shape += 1
            i += 1
            continue #go straight to the next iteration in the loop
        except EOFError:
            print("Some nifty files for this registation were corrupt.")
            corrupted += 1
            i+=1
            continue

        if min_slice_index <= max_slice_index:

            try:
                #Calculate the image array and input shape
                patient_image_array, blank_slices = pic.prepareImageArray(zero_bool, mid_zero_bool, flo_no, ref_no, machine, min_slice_index, max_slice_index, zero_slices, mid_zero_slices, n)
#                 print(patient_image_array.shape)
            except ValueError:
                #The ValueError is raised from within the prepare input cropped (pic) python file
                print("The shape of the input images was not 512x512xZ.")
                slice_wrong_shape += 1
                i += 1
                continue
            except EOFError:
                print("Some nifty files for this registation were corrupt.")
                corrupted += 1
                i+=1
                continue
            
            print("Blank slices were: ",blank_slices)
#             print("Blank slices shape: ",blank_slices.shape)     
            #Delete those elements of the metric arrays which had a corresponding blank slice
            delete_indices = []
            extra_zero_slices = 0 #The extra slice added at bottom (min_index-1) and top (max_index+1) may be a zero-slice 
            for i in blank_slices[2]:
                if 0<i<len(hd_95_array)+1: #blank_slices[i][2] goes from -1 (if min_index-1 is zero-slice) to
                    delete_indices.append(i)                               #len(hd)+1 (if max_slice+1 is zero-slice).
                else:
                    extra_zero_slices +=1
            
#             print("delete_indices:", delete_indices)
            dice_array = np.delete(dice_array, delete_indices)
            hd_95_array = np.delete(hd_95_array, delete_indices) 
            hd_75_array = np.delete(hd_75_array, delete_indices)
            hd_50_array = np.delete(hd_50_array, delete_indices)
            msd_array = np.delete(msd_array, delete_indices)
            rms_array = np.delete(rms_array, delete_indices)
#             msd_3D_array = np.delete(msd_3D_array, delete_indices)
            patient_image_array = np.delete(patient_image_array, delete_indices, axis=0)
            print(patient_image_array.shape)
#             print("length patient image array: {}, length ref contour slice array: {}".format(len(patient_image_array), len(ref_contour_slice_array)))
#             ref_contour_slice_array = np.delete(ref_contour_slice_array, delete_indices, axis=0)
            
        
            if len(hd_95_array) - extra_zero_slices == patient_image_array.shape[0]:
                print('Length of metric array: ', len(hd_95_array))
                print('Length of image array: ', patient_image_array.shape[0])
#                 print('Length of ref contour image array: ', ref_contour_slice_array.shape[0])
                
                #Add the metric arrays to the lists
                dice_list.append(dice_array)
                hd_95_list.append(hd_95_array)
                hd_75_list.append(hd_75_array)
                hd_50_list.append(hd_50_array)
                rms_list.append(rms_array)
                msd_list.append(msd_array)
#                 msd_3D_list.append(msd_3D_array)
                image_array_list.append(patient_image_array)
#                 ref_contour_slice_list.append(ref_contour_slice_array)

                registration_track.append([flo_no, ref_no, len(hd_95_array), patient_image_array.shape[0]])
                full_zeros.append(blank_slices + min_slice_index)
                
                print('The registration was processed successfully.')

                stop = time.time()
                print("This iteration took: " + str(stop-start) + " seconds." )
                i += 1

            else:
                print('The length of the metric arrays and the number of image slices do not have the correct correspondence.')
                i += 1

        else:
            print('Either or both images had no contours in them.')
            i += 1
        
    else:
        print('There are relevant files missing.')
        i += 1
          
global_stop = time.time()
print("The total time taken was: " + str(global_stop-global_start) + " seconds.")
#Default of np.concatenate() is axis=0
#The image_array returned by prepareImageArray, stored above in patient_image_array, has shape
#(number of slices from that registration, 512, 512, 4)
#Hence when appending every image_array into image_array_list, and then concatenating them in axis=0,
#we get multiple_patient_image_array which has shape (total number of slices from all registrations, 512,512,4)

multiple_patient_image_array = np.concatenate(image_array_list)
# multiple_ref_contour_image_array = np.concatenate(ref_contour_slice_list)
multiple_dice_array = np.concatenate(dice_list)
multiple_hd_95_array = np.concatenate(hd_95_list)
multiple_hd_75_array = np.concatenate(hd_75_list)
multiple_hd_50_array = np.concatenate(hd_50_list)
multiple_msd_array = np.concatenate(msd_list)
multiple_rms_array = np.concatenate(rms_list)
# multiple_msd_3D_array = np.concatenate(msd_3D_list)

print("Number of times the contours had the wrong shape: "+str(contour_wrong_shape))
print("Number of times the slice had the wrong shape: "+str(slice_wrong_shape))
print("Number of times the files were corrupted: "+str(corrupted))
print("The number of successful registrations was: ", len(registration_track))

# desired directory in which to save the arrays to
folder_arrays = '/hepgpu9-data1/sbridger/mphys_2d_network_2021-22/slice_arrays/new_registrations/' + organ + '/' + train_test + '/'

if not os.path.exists(folder_arrays):
    os.makedirs(folder_arrays)

suffix = train_test + '_' + organ + '_' + indexes

image_array_name = folder_arrays + 'image_dvf_array_' + suffix
# ref_contour_image_array_name = folder_arrays + 'ref_contour_image_array_' + suffix
dice_array_name = folder_arrays + 'dice_array_' + suffix
hd_95_array_name = folder_arrays + 'hd_95_array_' + suffix
hd_75_array_name = folder_arrays + 'hd_75_array_' + suffix
hd_50_array_name = folder_arrays + 'hd_50_array_' + suffix
msd_array_name = folder_arrays + 'msd_array_' + suffix
rms_array_name = folder_arrays + 'rms_array_' + suffix
# msd_3D_array_name = folder_arrays + 'msd_3D_array_' + suffix
# hd_95_array_name_3D = folder_arrays + 'hd_95_array_3D_' + suffix
# rms_array_name_3D = folder_arrays + 'rms_array_3D_' + suffix

registration_track_name = folder_arrays + 'registration_track_' + suffix
blank_slices_name = folder_arrays + 'blank_slices_' + suffix

print(multiple_patient_image_array.shape)
np.save(image_array_name, multiple_patient_image_array)
print("image array saved")
# print(multiple_ref_contour_image_array.shape)
# np.save(ref_contour_image_array_name, multiple_ref_contour_image_array)
# print("ref contour image array saved")
np.save(dice_array_name, multiple_dice_array)
print("dice array saved")
#np.save(hd_array_name, multiple_hd_array)
#print("hd array saved")
np.save(hd_95_array_name, multiple_hd_95_array)
print("95 percentile hd array saved")
np.save(hd_75_array_name, multiple_hd_75_array)
print("75 percentile hd array saved")
np.save(hd_50_array_name, multiple_hd_50_array)
print("50 percentile hd array saved")
np.save(msd_array_name, multiple_msd_array)
print("msd array saved")
#np.save(msd_array_name, multiple_msd_array)
#print("mean surface distance array saved")
np.save(rms_array_name, multiple_rms_array)
print("root mean square distance array saved")
# np.save(hd_95_array_name_3D, multiple_hd_95_array_3D)
# print("95 percentile 3D hd array saved")
# np.save(msd_3D_array_name, multiple_msd_3D_array)
# print("3D mean surface distance array saved")
# np.save(rms_array_name_3D, multiple_rms_array_3D)
# print("3D root mean square distance array saved")

np.save(registration_track_name, registration_track)
print("Registration track array saved")

if np.any(full_zeros):
    np.save(blank_slices_name, full_zeros)
    print("Blank slices array saved")
else:
    print("There were no mid_zero_slices")
