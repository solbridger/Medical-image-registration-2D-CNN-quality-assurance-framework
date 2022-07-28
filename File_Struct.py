import os

class File_struct:

  def __init__(self, Pat_no, IN): #IN will be a array of the files that are in the file                                                                                                                                              
        #check what data we have been given                                                                                                                                                                                          
        self.Pat_no = Pat_no #stores the patient number                                                                                                                                                                              

        Structures = []
        Structure_files = []
        for paths in IN:
                Files = os.path.split(paths)

#                 check which file each one is                                                                                                   
                if ('NoBackground_0522'in Files[1]) and ('.nii'  in Files[1]):#identifies the nifty image                                                 
#                 if ('0522'in Files[1]) and ('.nii'  in Files[1]):#identifies the nifty image                                                                                                                                         
                        self.IMG = paths
                elif '.dcm' in Files[1]:#identifies the dicom files                                                                                                                                                                  
                        self.DICOM = paths
                elif '.txt' in Files[1]: #identify any text files                                                                                                                                                                    
                        self.Transform_Txt = paths
                elif 'cpp' in Files[1]:#identifies the cpp file                                                                                                                                                                      
                        self.Transform_cpp = paths
                elif 'Def_' not in Files[1]:#find any of the structure files.                                                                                                                                                        
                        Structure_files.append(Files[1])
                        Structures.append(paths)
        self.Structures = Structures
        self.Files = Structure_files



  def __str__(self):

        Print = 'Patient ' + self.Pat_no + ': ' + self.IMG + ' ' + self.DICOM
        for i in range(0,len(self.Structure_files)):
                Print += '_'
                Print += self.Structure_files[i]

        return Print
