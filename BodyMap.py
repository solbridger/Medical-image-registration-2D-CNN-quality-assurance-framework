from deepregtools.utility import * 
import nibabel as nib
import numpy as np



def RemoveBackground(Image):	
	Path = Image.split('/')

	saveBodyMask(Image,'BodyMask')
	img = nib.load(Image)
	image_data = np.array(img.get_data())
	bodymap = nib.load('BodyMask.nii')
	bodymap_data = np.array(bodymap.get_data())
	null = []
	Min = 0
	for i in range(0,len(bodymap_data)):
		Zeroes = []
		for j in range(0,len(bodymap_data[i])):
			zero = []
			for k in range(0,len(bodymap_data[i][j])):
				if image_data[i][j][k] < Min: 
					Min = image_data[i][j][k]
				zero.append(0)
			Zeroes.append(zero)				
		null.append(Zeroes)
	no_bg = np.where(bodymap_data == null, bodymap_data, (image_data + abs(Min)))
	NoBackNii = nib.Nifti1Image(no_bg, affine = img.affine, header = img.header)
	
	
	nib.save(NoBackNii,'NoBackground_' + Path[len(Path)-1] )



			


RemoveBackground('/afs/hep.man.ac.uk/u/zaklaw/Downloads/Flo_248_Ref_081/Def_Flo_248_Ref_081.nii.gz')

