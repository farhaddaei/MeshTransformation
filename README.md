# MeshTransformation
 MeshTransformation
 A simple class to test the best method
 transfering potential,magnetic, and
  current vector fields to new mesh structure
  conserving physical properties of the system.
  
  Install
  -------
  1. Create a conda envirment with Python >= 3.6
  2. Activate the new env
git  4. run following command in terminal
  
     pip install .
  
  It will install the package and also the dependencies
  
  Usage
  -------
  To see the commands and results look at __"RunMe_MFM_Data.ipynb"__ and also Paraview state file in __example__ directory.
  
  Successfully tested for three cases:
   - extract $\vec{A}$ from $\vec{B}$ (MFM Data), recalculate $\vec{B}$ in the same mesh (No interpolation)
   - extract $\vec{A}$ from $\vec{B}$ (MFM Data), interpolate $\vec{A}$ into another mesh with same resolution
   - extract $\vec{A}$ from $\vec{B}$ (MFM Data), interpolate $\vec{A}$ into another mesh with lower resolution
  
  