# PID2021-124297NB-C31 Repository
This repository contains LAMMPS scripts and Fortran codes used to obtain the results presented in publications `doi.org/10.1063/5.0147459` and `doi.org/10.1088/1742-5468/ad4024`. The codes provided are not ready to be run as-is and will require adaptation to work on a specific machine. This repository does not contain any simulation data, it instead includes all the necessary input files to generate it.

The repository is organized in two main directories.

## `nitromethane` (`doi.org/10.1063/5.0147459`)

This directory contains the LAMMPS scripts and codes for simulating a nitromethane molecule in argon gas and liquid water and perform a vibrational excitation. The version used was `LAMMPS 23 Jun 2022 - Update 4` with, at least `EXTRA-MOLECULE KSPACE MOLECULE PYTHON RIGID`.

- `vibrational_analysis` contains the Fortran codes used to compute the Eckart frame of a molecule and perform the vibrational excitation.

- `universal_excitation` contains the Fortran codes to perform an generic vibrational kinetic eneregy excitation.

- `aux_vib` contains some functionality of `vibrational_analysis` in a Python package compiled with `f2py`.

- `common_files` contains force fields, molecular topology and equilibration runs used by the rest of the directories.

- The rest of the directories are either `aap` for nitromethane simulations using the AAP model or `fullsrt` using the SRT model (note that in both cases the SRT model is used for NM's intramolecular interactions). After the model is specified it is followed by the specific quantities computed (e.g. `force`, `work`, etc.) and followed by `nitro` and either `spce` for water or `argon` for argon. For argon only general simulation were performed. E.g. `aap310vacf_nitro_spce` contain simulations for the VACF in water with the AAP model at 310K.

- `new_force_field` contains the codes for computing the `NEW` parametrization.

## `active_brownian` (`doi.org/10.1088/1742-5468/ad4024`)

This directory contains the code used for simulating single ABP particles with exponetially decaying activity (as well as constant activity and other time dependent profiles) and the analytic formulas for their MSD.

- `absim.f90` is the main Fortran code, it receives its inputs via namelist files (`.txt`). It requires the `MT19937` RNG generator found in `modules`.

- `formulas.py` is a python module with all the analytic formulas presented in `doi.org/10.1088/1742-5468/ad4024` and some other useful formulas.

- `janus` contains all the input scripts for running the simulations organized by their type.
