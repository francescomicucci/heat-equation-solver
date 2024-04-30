.SUFFIXES:
.SUFFIXES: .o .f90 .f77  	



#----------------------------------------
# COMPILERS AND LINKER
#----------------------------------------       
           

#  FFLAGS = -ffree-line-length-none  -O3 -fPIC -std=legacy -march=native -ffast-math -funroll-loops
  FFLAGS = -ffree-line-length-none  -g  -fbounds-check -Wall -ffpe-trap=invalid -static-libgfortran

LNK = mpif90
OPTLNK = $(FFLAGS)
CMPF90 = mpif90 -c -cpp

#----------------------------------------
# DIRECTORIES
#----------------------------------------                       


#----------------------------------------
# LIBRARIES
#----------------------------------------

LIBS    =-L../../2decomp-fft/build/opt/lib -L../../2decomp-fft/build/otp/lib64 -ldecomp2d

#----------------------------------------
# INCLUDES
#----------------------------------------

INCLUDES =-I../../2decomp-fft/build/opt/include


#----------------------------------------
# OBJECTS
#----------------------------------------
  OBJECT_BASIC = 
OBJECTS_DD_D = 
OBJECTS_NS   = 
OBJECTS      = $(OBJECT_BASIC) 

#----------------------------------------
# All executables to be built
#----------------------------------------

EXECUTABLES = TimeDepHeat3D_parallel.exe


ALL:  $(EXECUTABLES)

#----------------------------------------
# Test Parallel
#----------------------------------------



TimeDepHeat3D_parallel.exe:  TimeDepHeat3D_parallel.o $(OBJECTS)
	($(LNK) -o TimeDepHeat3D_parallel.exe $(FFLAGS) TimeDepHeat3D_parallel.o $(OBJECTS) $(LIBS))

TimeDepHeat3D_parallel.o: TimeDepHeat3D_parallel.f90 $(OBJECTS)
	($(CMPF90) $(FFLAGS) $(INCLUDES) TimeDepHeat3D_parallel.f90 )


#----------------------------------------
# Test Schur
#----------------------------------------

#----------------------------------------
# Clean
#----------------------------------------

clean:
	rm $(OBJECTS)
	rm $(EXECUTABLES)
	rm *.mod
	rm TimeDepHeat3D_parallel.o

