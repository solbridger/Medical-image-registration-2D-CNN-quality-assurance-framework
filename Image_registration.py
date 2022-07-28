from nipype.interfaces import niftyreg
#this file defines all the registration algorithms used in the project 

#we may want to look at changing some of the input parameters so that we canbetter optimis the transformations

#affine registration 
def Reg_Affine(Flo_path,Ref_path,Res_path,Aff_path):
	node = niftyreg.RegAladin()
	node.inputs.flo_file = Flo_path #define the floting image and the reference image. 
	node.inputs.ref_file = Ref_path
	node.inputs.res_file = Res_path #where we want to put the resulting image and name it
	node.inputs.aff_file = Aff_path #where we want to put the resulting affine transformation 	
	node.inputs.args = '-speeeeed'
	node.inputs.verbosity_off_flag = True 
	node.run()
	return node.cmdline

#deformable registration 
def Reg_Deform(Flo_path,Ref_path,Res_path,Cpp_path,Aff_path):
	node = niftyreg.RegF3D()
	node.inputs.flo_file = Flo_path
	node.inputs.ref_file = Ref_path
	node.inputs.res_file = Res_path
	node.inputs.cpp_file = Cpp_path
	node.inputs.aff_file = Aff_path 
	node.inputs.verbosity_off_flag = True 
	node.inputs.be_val = 0.002	
	node.inputs.sx_val = 12
	node.inputs.sy_val = 12
	node.inputs.sz_val = 12
	node.run()
	return node.cmdline

#The resampling of the images, takes a path for each of the files as a input. 
def Resample(Flo_path,Ref_path,Res_path,Trans_path):
	node = niftyreg.RegResample()
	node.inputs.flo_file = Flo_path
	node.inputs.ref_file = Ref_path	
	node.inputs.out_file = Res_path
	try:
		node.inputs.trans_file = Trans_path
	except: 
		pass
	node.run()
	return node.cmdline


#Calculation of the Displacement and the Deformation fileds are here 
def Displacement_field(Trans_path,Ref_path,Out_path):
	node = niftyreg.RegTransform()
	node.inputs.disp_input = Trans_path
	node.inputs.ref1_file = Ref_path
	node. inputs.out_file = Out_path
	node.run()
	return None


def Deformation_field(Trans_path,Ref_path,Out_path):
	node = niftyreg.RegTransform()
	node.inputs.disp_input = Trans_path
	node.inputs.ref1_file = Ref_path
	node. inputs.out_file = Out_path
	node.run()
	return None
