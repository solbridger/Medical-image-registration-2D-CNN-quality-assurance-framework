import numpy as np
import nibabel as nib
import dice_and_sds_2d_from_array as dfa
from scipy.ndimage import zoom, morphology

#RIGHT NOW THIS FILE CONTAINS 3D METRICS FOR z-DIRECTION SLICES, WITH ZOOMING IN TO THE "NATURAL" FACTOR.
train_test='test_even'

def prepareImageArray(zero_bool, mid_zero_bool, flo_no, ref_no, machine, min_slice_index, max_slice_index, zero_slices, mid_zero_slices, n): 
#input the patient numbers of the floating image and reference image as strings
#input the machine as string 'hepgpu3' or 'hepgpu4'
#input the min and max slice indexes for def and ref from the prepareDiceArray function output 
#function to prepare the array of images used to train the CNN
#n is the direction you chose to take slices in (n={0,1,2} corresponding to x,y,z)
    
    #Define filepaths of reference image and DVF 
    if (machine == 'hepgpu9'):
        if train_test=='train_odd' or train_test=='test_even':
            img_filepath = '/hepgpu9-data1/sbridger/94_patient_data/patient_filepath/' + ref_no + '/NoBackground_0522c0' + ref_no +'.nii.gz'
            dvf_filepath = '/hepgpu9-data1/sbridger/Auto_Registration/registration_store/new_registration/' + train_test + '/Flo_' + flo_no + '_Ref_' + ref_no + '/Disp_Vector_Field.nii'

    #Find the reference image and DVF and convert them into numpy arrays
    img_nii = nib.load(img_filepath)
    input_img = img_nii.get_fdata()    
    img_dimension = input_img.shape
    dvf_nii = nib.load(dvf_filepath)
    input_dvf = dvf_nii.get_fdata()

    no_time_dvf = input_dvf[:,:,:,0,:] #Remove the time dimension from the dvf array
    #no_time_dvf now has shape (x-dimension, y-dimension, z-dimension, 3), where 3 corresponds to the 3 components of the DVF.
    
#     print("input_img_original shape:  ", input_img_original.shape)
#     print("input_dvf shape:  ", input_dvf.shape)

#     affine_matrix = img_nii.affine
#     slice_spacing_z = abs(affine_matrix[2,2])
#     slice_spacing_xy = abs(affine_matrix[1,1])
#     z_factor = slice_spacing_z//slice_spacing_xy
#     z_dimension = int(z_factor*input_img_original.shape[2])
#     print('z_factor: {} , z_dimension: {}'.format(z_factor, z_dimension))

#     input_img = np.zeros(shape=(512, 512, z_dimension))
#     no_time_dvf = np.zeros(shape=(512, 512, z_dimension, 3))

#     for j in range(0, 512):
#         input_img[:,j,:] = zoom(input_img_original[:,j,:], (1, z_factor)) # zoom in by scale of 'z_factor' in z-axis
#         no_time_dvf[:,j,:,:] = zoom(no_time_dvf_original[:,j,:,:], (1, z_factor, 1))

    input_img = input_img.astype(np.float16)
    no_time_dvf = no_time_dvf.astype(np.float16)

    if input_img.shape[2] <= 512:
        input_img = np.pad(input_img, [(0,0),(0,0),(0,512-input_img.shape[2])], mode='constant', constant_values=((0,0),(0,0),(0,np.min(input_img))))
        no_time_dvf = np.pad(no_time_dvf, [(0,0),(0,0),(0,512-no_time_dvf.shape[2]),(0,0)], mode='constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
    else: #CROPPING OUT SLICES AT THE BOTTOM (image shape > 512)
        input_img = input_img[:,:,input_img.shape[2]-512:input_img.shape[2]]
        no_time_dvf = no_time_dvf[:,:,no_time_dvf.shape[2]-512:no_time_dvf.shape[2],:]
        
#     print("input_img shape:  ", input_img.shape)
#     print("no_time_dvf shape:  ", no_time_dvf.shape)

    slices = {
        0: slice(0, input_img.shape[0]),
        1: slice(0, input_img.shape[1]),
        2: slice(0, input_img.shape[2])
    }

    #Create a numpy array to store the inputs
    #The array has dimensions n x m where n is the number of training samples (i.e. the number of slices) and m is the dimensions of each input (i.e. x and y dimensions of original image and 4 z dimensions for the 4 channels, which are the image brightness and the 3 components of the DVF)
    image_array = np.zeros(shape=(max_slice_index-min_slice_index-len(zero_slices)-len(mid_zero_slices)+1, 512, 512, 4))
    blank_slices=[[],[],[]]
    for i in range(min_slice_index, max_slice_index+1):
        
        if (i in zero_slices):
            blank_slices[2].append(i)
        #If zero_bool is set to False, then you would only go into the following "if" block if the Dice value is non-zero.
        #If mid_zero_bool is set to False, then you would only go into the following "if" block if there are no missing organs in the slices.
        if (i not in zero_slices or zero_bool == True) and (i not in mid_zero_slices or mid_zero_bool == True):
            #Extract the slice from the reference image and the DVF
            slices[n] = slice(i, i+1)

            img_slice = input_img[slices[0],slices[1],slices[2]]
            dvf_slice = no_time_dvf[slices[0],slices[1],slices[2], :]

            #Eliminate the extra dimension in the dvf slice, since using slice() leaves it with (512,512,1,3)
            shaped_dvf_slice = np.reshape(dvf_slice, (512, 512, 3))

            #Add the vectors from the DVF slice as channels in the reference image
            train_slice = np.concatenate((img_slice, shaped_dvf_slice), axis = 2)

            if( (mid_zero_bool == False) or (zero_bool == False) ):
                #Calculate how many slices with 0 Dice Score or no contour there were between the min and the current slice
                position = 0
                for mid_zero_slice in mid_zero_slices:
                    if i > mid_zero_slice:
                        position += 1
                for zero_slice in zero_slices:
                    if i > zero_slice:
                        position += 1
                #Add this input to the array of inputs
                image_array[i-min_slice_index-position, :, :, :] = train_slice
            else:
                image_array[i-min_slice_index,:,:,:] = train_slice
    blank_slices = np.array(blank_slices)
    image_array_less_digits = np.array(image_array).astype(np.float16)
    return image_array_less_digits, blank_slices

def prepareMetricsArrays(zero_bool, mid_zero_bool, flo_no, ref_no, machine, organ, n): #n={0,1,2} corresponding to slicing direction x, y or z
#function to prepare the array of Dice scores used to train the CNN
    slicingDirection = {
        0: "x",
        1: "y",
        2: "z"
    }
    print("You have chosen to take slices in the " + slicingDirection[n] + "-direction")

    if(machine == 'hepgpu9'):
        if train_test =='train_odd' or train_test == 'test_even':               
            def_contour_filepath = '/hepgpu9-data1/sbridger/Auto_Registration/registration_store/new_registration/' + train_test + '/Flo_' + flo_no + '_Ref_' + ref_no + '/Def_Flo_' + flo_no + '_Ref_' + ref_no + '_' + organ + '.nii.gz'               
            ref_contour_filepath = '/hepgpu9-data1/sbridger/94_patient_data/patient_filepath/' + ref_no + '/' + organ + '.nii.gz'

    #Find the def and ref images and convert them into numpy arrays
    def_contour_nii = nib.load(def_contour_filepath)
    def_contour_array = def_contour_nii.get_fdata()    
    ref_contour_nii = nib.load(ref_contour_filepath)
    ref_contour_array = ref_contour_nii.get_fdata()
#     print(def_contour_nii.shape[0], def_contour_nii.shape[1])

    if def_contour_nii.shape[0] == def_contour_nii.shape[1] != 512:
        raise ValueError('The images x- and y-dimensions were not 512.')

#     affine_matrix = def_contour_nii.affine
#     slice_spacing_z = abs(affine_matrix[2,2])
#     slice_spacing_xy = abs(affine_matrix[1,1])
#     z_factor = slice_spacing_z//slice_spacing_xy
#     z_dimension = int(z_factor*def_contour_array_original.shape[2])
#     print('z_factor: {} , z_dimension: {}'.format(z_factor, z_dimension))

    
#     def_contour_array = np.zeros(shape=(512, 512, z_dimension))
#     ref_contour_array = np.zeros(shape=(512, 512, z_dimension))
    
#     for j in range(0, 512):
#         def_contour_array[:,j,:] = zoom(def_contour_array_original[:,j,:], (1, z_factor))
#         ref_contour_array[:,j,:] = zoom(ref_contour_array_original[:,j,:], (1, z_factor))

    def_contour_array = np.round(def_contour_array)
    ref_contour_array = np.round(ref_contour_array)

    if def_contour_array.shape[2] <= 512:
        def_contour_array = np.pad(def_contour_array, [(0,0),(0,0),(0,512-def_contour_array.shape[2])], mode='constant', constant_values=((0,0),(0,0),(0,0)))
        ref_contour_array = np.pad(ref_contour_array, [(0,0),(0,0),(0,512-ref_contour_array.shape[2])], mode='constant', constant_values=((0,0),(0,0),(0,0)))
    else: #CROPPING OUT SLICES AT THE BOTTOM
        def_contour_array = def_contour_array[:,:,def_contour_array.shape[2]-512:512]
        ref_contour_array = ref_contour_array[:,:,ref_contour_array.shape[2]-512:512]
    
#     print("def_contour_array shape:  ", def_contour_array.shape)
#     print("ref_contour_array shape:  ", ref_contour_array.shape)
    
    #Create variables to find the range of slices within which both the def and ref images have contours    
    min_slice_index = 0
    max_slice_index = 511

    slices = {
        0: slice(0, 512),
        1: slice(0, 512),
        2: slice(0, 512)
    }

    slices[n] = slice(0,1)

    ref_slice = ref_contour_array[slices[0], slices[1], slices[2]]
    def_slice = def_contour_array[slices[0], slices[1], slices[2]]

    summation_def = np.sum(def_slice)
    summation_ref = np.sum(ref_slice)
    while (summation_def == 0 or summation_ref == 0) and min_slice_index < max_slice_index:
        min_slice_index +=1
        slices[n] = slice(min_slice_index, min_slice_index+1)

        ref_slice = ref_contour_array[slices[0], slices[1], slices[2]]
        def_slice = def_contour_array[slices[0], slices[1], slices[2]]

        summation_def = np.sum(def_slice)
        summation_ref = np.sum(ref_slice)

    slices[n] = slice(max_slice_index, max_slice_index+1)

    ref_slice = ref_contour_array[slices[0], slices[1], slices[2]]
    def_slice = def_contour_array[slices[0], slices[1], slices[2]]

    summation_def = np.sum(def_slice)
    summation_ref = np.sum(ref_slice)
    while (summation_def == 0 or summation_ref == 0) and max_slice_index >= min_slice_index:
        max_slice_index -=1
        slices[n] = slice(max_slice_index, max_slice_index+1)

        ref_slice = ref_contour_array[slices[0], slices[1], slices[2]]
        def_slice = def_contour_array[slices[0], slices[1], slices[2]]

        summation_def = np.sum(def_slice)
        summation_ref = np.sum(ref_slice)

    #Create the arrays for the different metrics
    dice_array = []
    hd_95_array = []
    hd_75_array = []
    hd_50_array = []
    rms_array = []
#     hd_95_array_3D = []
    msd_array = []
#     msd_array_3D = []
#     rms_array_3D = []
    ref_slice_array = []
    def_slice_array = []

    zero_slices = [] #To record the index of the slices with zero dice score (or with the wrong shape)

    print("Min: ", min_slice_index)
    print("Max: ", max_slice_index)


    if min_slice_index > max_slice_index: #If there had been no contours (in all the slices of either or both of the images), then the max index would have gone smaller than the min index to exit the while loop which starts from the top
        print("min_slice_index is bigger than max_slice_index")
        mid_zero_slices = []
    else:
        #CHECK IF THE IMAGES HAVE NO CONTOUR IN THE SAME SLICE WITHIN THE SELECTED RANGE
        #THIS IS UNLIKELY TO HAPPEN BUT IT WOULD GIVE A "DIVISION BY 0" ERROR WHEN CALCULATING THE DICE SCORE
        #THE INDEXES OF THE SLICES IN WHICH THIS HAPPENS ARE RECORDED IN THE ARRAY mid_zero_slices_both
        mid_zero_slices_def = []
        mid_zero_slices_ref = []
        mid_zero_slices_both = []
        mid_zero_slices = [] #Record all the slices in which either (or both) images have no contour

        #Create 3D penalty arrays for later calculations of 3D surface distance
#         boolean_def_array = np.atleast_1d(def_contour_array.astype(np.bool))
#         boolean_ref_array = np.atleast_1d(ref_contour_array.astype(np.bool))

#         penalty_array_def = morphology.distance_transform_edt(~boolean_def_array, 1)
#         penalty_array_ref = morphology.distance_transform_edt(~boolean_ref_array, 1)

        for i in range(min_slice_index, max_slice_index+1):
            slices[n] = slice(i, i+1)
            def_slice = def_contour_array[slices[0],slices[1],slices[2]]
            ref_slice = ref_contour_array[slices[0],slices[1],slices[2]]
            No_contour_def = No_contour_ref = False

            if np.sum(def_slice) == 0:
                mid_zero_slices_def.append(i)
                No_contour_def = True

            if np.sum(ref_slice) == 0:
                mid_zero_slices_ref.append(i)
                No_contour_ref = True

            if No_contour_def and No_contour_ref:
                mid_zero_slices_both.append(i)

            if (No_contour_def or No_contour_ref) and not mid_zero_bool: #For the moment not using those slices in which the organs have no contour
                mid_zero_slices.append(i)
                continue

            #Calculate the metrics
            dice = dfa.getDice2D(def_slice, ref_slice)
            sds = dfa.getSurfaceDistance(def_slice, ref_slice, 1.2)
#             sds = dfa.getSurfaceDistance(def_slice, ref_slice, 2.76) for x or y slices

            #Calculate the 3D surface distance array

            #From ref 2D slice to 3D def contour
#             penalty_array_def_slice = penalty_array_def[slices[0],slices[1],slices[2]]
#             sds_ref_to_3D_def = np.ravel(penalty_array_def_slice[ref_slice!=0])

            #From def 2D slice to 3D ref contour
#             penalty_array_ref_slice = penalty_array_ref[slices[0],slices[1],slices[2]]
#             sds_def_to_3D_ref = np.ravel(penalty_array_ref_slice[def_slice!=0])

#             sds_3D = np.concatenate([sds_ref_to_3D_def, sds_def_to_3D_ref])

            ######if dice == 0 and zero_bool == False: #zero_bool is set to False if we want to get rid of dice scores of 0
            ######	zero_slices.append(i)
            ######else:
            dice_array.append(dice)
                ########hd_array.append( sds.max() )
            hd_95_array.append( np.float32 ( np.percentile(sds, 95) ) )
            hd_75_array.append( np.float32 ( np.percentile(sds, 75) ) )
            hd_50_array.append( np.float32 ( np.percentile(sds, 50) ) )
            rms_array.append( np.sqrt( (sds**2).mean() ) )
#             hd_95_array_3D.append( np.float32 ( np.percentile(sds_3D, 95) ) )
            msd_array.append( sds.mean() )
#             msd_array_3D.append( sds_3D.mean() )
#             rms_array_3D.append( np.sqrt( (sds_3D**2).mean() ) )
#             ref_slice_array.append(ref_slice)
            def_slice_array.append(def_slice)

    return dice_array, hd_95_array, hd_75_array, hd_50_array, rms_array, msd_array, min_slice_index, max_slice_index, zero_slices, mid_zero_slices, def_slice_array#, ref_slice_array #, rms_array_3D
# 	return dice_array, hd_95_array_3D, mds_array_3D, min_slice_index, max_slice_index, zero_slices, mid_zero_slices
