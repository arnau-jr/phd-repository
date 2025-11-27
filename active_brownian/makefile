.SUFFIXES:

FC = gfortran
FFLAGS = -fbounds-check -O3 -fopenmp 
FLIBS = -llapack

SRC_DIR  = ./modules/
MOD_DIR = $(SRC_DIR)/modfiles/

COMPILE = $(FC) $(FFLAGS) $(TARGET_ARCH) -I$(MOD_DIR) -J$(MOD_DIR) -c
LINK = $(FC) $(FFLAGS) $(TARGET_ARCH) -I$(MOD_DIR) -J$(MOD_DIR)

MAINSOURCE = absim.f90
MODULESOURCES = $(wildcard $(SRC_DIR)*.f90)
SOURCES = $(MAINSOURCE) $(wildcard $(SRC_DIR)*.f90)


$(subst .f90,.x,$(MAINSOURCE)): $(subst .f90,.o,$(MAINSOURCE)) 
	$(LINK) -o $@ $(SOURCES) $(FLIBS)

$(subst .f90,.o,$(MAINSOURCE)): $(MAINSOURCE) $(subst $(SRC_DIR),$(MOD_DIR),$(subst .f90,.mod,$(MODULESOURCES)))
	$(COMPILE) -o $@ $< $(FLIBS)

$(MOD_DIR)%.o: $(SRC_DIR)%.f90
	$(COMPILE) -o $@ $< $(FLIBS)

$(MOD_DIR)mtfort90.mod: $(MOD_DIR)mtfort90.o

.PHONY: clean

clean:
	rm -f $(MOD_DIR)*.o $(MOD_DIR)*.mod
	rm -f *.o
