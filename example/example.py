# /usr/bin/env python

from DataSet import DataSet as DS

BaseAddress = "/home/local/farhadda/BackUp/div_E_Omega_2011-06-02T05-36-00/"
ds1 = DS(SystemOfCoords="CAR", BaseAddress=BaseAddress,
         Pattern="div_E_Omega_2011-06-02T05-36-00_*.npz",
         NBlocks=10, UseBlock=0)
print("Number of cells in each direction : ", ds1.NCell)
ds1.ToPluto(Address="/home/local/farhadda/", BaseName="HaHa_")
# ds1.Write2HDF5("/home/farhadda/OhOh.h5")