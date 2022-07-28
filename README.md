# AutoRegistration

This piece of code is aimed at autimating the process of registering images for the creating of a database to train the CNN.
The main pieces of code that are used are used are the Auto_Registration_V2.py, File_Struct.py, Image_registration.py, metrics.py and Multicore.py

Auto_Registration_V2 is the main piece of code for the code to use, this will bring together all the parts of the code that are used to define functions used in the code and will usde them to perform the refistrations. The code makes use of a few packages like os and glob which allow us to find the correct files to feed into the system. The file paths are given the the File_struct class to allow all the relevent paths for each of the patients to be stored and called when needed by the code in a easy way that means that we do not need to remeber the full path each time we need a specific file. Once the files are storred then the code will check if there is atleast one structure in the list of srtructures in common with the other patient before creating the file for the registration. The code will then calculate the required metrics before creating during and after the registration process. During the registration there is also a log file that is created.

Image_registration contians tghe code that uses nipype to perform the registrations in a way that makes it easy to pass the file paths tot he registration code.

metrics contains a couple of functions that we need to calculate the metrics. 

File_Struct is needed as this is what allows us to store the files that we will use for the registrations in a callabe instance of the class for each of the partients.  

sasdhiuhasdhaidksdjasjdoasdpasop
more changes
sorry for butchering the file
