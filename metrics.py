# metrics.py
# Calculates DICE and Mean Surface Distance from an input of two binary images (for the two contours)

import numpy as np
import nibabel as nib


## Defining functions to calculate metrics

# Surface distance
def surfd(input1, input2, sampling=1, connectivity=1):
  from scipy.ndimage import morphology
  # surface_distance = surfd()
  # msd = surface_distance.mean()
  # rms = np.sqrt((surface_distance**2).mean())
  # hd  = surface_distance.max()
  # from https://mlnotebook.github.io/post/surface-distance-function/
  input_1 = np.atleast_1d(input1.astype(np.bool))
  input_2 = np.atleast_1d(input2.astype(np.bool))

  conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

  S = np.bitwise_xor(input_1,morphology.binary_erosion(input_1, conn))
  Sprime = np.bitwise_xor(input_2, morphology.binary_erosion(input_2, conn))

  dta = morphology.distance_transform_edt(~S,sampling)
  dtb = morphology.distance_transform_edt(~Sprime,sampling)
  
  sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
      
  return sds

# Dice similarity coefficients
def getDice(input1, input2):
  #function to compute 3D DICE score
  input_1 = np.atleast_1d(input1.astype(np.int))
  input_2 = np.atleast_1d(input2.astype(np.int))
  # Add the two binary images together
  intersection = input_1 + input_2 
  # not actually intersection : DSC is defined as 
  # intersection

  # score of zero where the two don't overlap 
  intersection[np.where(intersection==1)] = 0
  intersection[np.where(intersection==2)] = 1
  dice = 2 * np.sum(intersection).astype(np.float) / (np.sum(input_1) + np.sum(input_2))
  return dice


# Check to see if the two images have the same dimensions before calculating the metrics. 
def CheckShape(GT_Path,REG_Path):
	GT_nii = nib.load(GT_Path)
	REG_nii = nib.load(REG_Path)
	GT = GT_nii.get_fdata()
	REG = REG_nii.get_fdata()
	if GT.shape != REG.shape :
		return False
	else: 
		return True





# whyd you comment all this out


#if __name__=='__main__':
  # load nifti images using nibabel: http://nipy.org/nibabel/manual.html#manual 
  #registeredROI must be a propagated binary image (e.g. PAROTID_R.nii) that 
  # has been reg_resample'd into the refrence space of the ground truth image
  # check that this is the case by checking they both 
  # have the same vdim's and the same shapes.
 # groundtruth_nii = nib.load('/hepgpu3-data1/deepreg/data/nifti/cancer_imaging_archive/head_and_neck/patients/806/SPINAL_CORD.nii')
  #registeredROI_nii = nib.load('/hepgpu3-data1/summer_students/Zak/Test_Registration/Flo_871_Ref_806/Def_Flo_077_Ref_806_SPINAL_CORD.nii')
  #groundtruth = groundtruth_nii.get_data()
  #registeredROI = registeredROI_nii.get_data()
  # get the physical voxel dimensions
  #vdim = groundtruth_nii.header.get_zooms()
  #print('Physical voxel dimensions of reference: {}'.format(vdim)) 
  # check that these two images have the same shape
  #if groundtruth.shape != registeredROI.shape:
    
  #else:
    #surface_distance = surfd(groundtruth, registeredROI, vdim, 2)
    # calculate metrics such as
    # dice coefficient
    #dice = getDice(groundtruth, registeredROI)
    #print("DICE Similarity Coefficent = %.3f" % (dice))
    # median surface distance
    #mediansd = np.median(surface_distance)
    #print("Median Surface Distance = %.3f" % (mediansd))
    # mean surface distance
    #msd = np.mean(surface_distance)
    #print("Mean Surface Distance = %.3f" % (msd))
    # root mean square residual
    #rms = np.sqrt(((surface_distance - msd)**2).mean())
    #print("Root Mean Squared Residual = %.3f" % (rms))
    # Hausdorff distance
    #hd  = surface_distance.max()
    #print("Hausdorff Distance = %.3f" % (hd)) 
    # 95th percentile surface distance of the 3D structures
    #perc = np.percentile(surface_distance, 95)
    #print("95th Percentile Surface Distance = %.3f" % (perc))

