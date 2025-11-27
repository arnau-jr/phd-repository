objects = molecule.o force_field.o vibration.o solvent_processor.o
comp = gfortran
OPT = -fbounds-check -O3 -fopenmp


molecule.o: molecule.f90
	$(comp) -c $(OPT) molecule.f90

force_field.o: force_field.f90
	$(comp) -c $(OPT) force_field.f90

vibration.o: vibration.f90 molecule.o force_field.o
	$(comp) -c $(OPT) vibration.f90 molecule.f90 force_field.f90 -llapack

solvent_processor.o: solvent_processor.f90
	$(comp) -c $(OPT) solvent_processor.f90 -llapack

.PHONY: clean_o all

all : molecule.o force_field.o vibration.o solvent_processor.o
	touch molecule.o
	touch force_field.o
	touch vibration.o
	touch solvent_processor.o

clean_o:
	rm -f $(objects)
