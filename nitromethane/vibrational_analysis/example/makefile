objects = test.o
modules = ../molecule.f90 ../force_field.f90 ../vibration.f90
module_objs = ../molecule.o ../force_field.o ../vibration.o
comp = gfortran
OPT = -fbounds-check -O3 

test.x: $(objects) $(module_objs)
	$(comp) -o test.x  $(OPT) $(objects) $(module_objs) -llapack

test.o: test.f90 $(module_objs)
	$(comp) -c $(OPT) test.f90 $(modules) -llapack


.PHONY: clean clean_o

clean:
	rm -f $(objects)
	rm -f *.o
	rm -f *.mod

clean_o:
	rm -f $(objects)
