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
    >>> from DataSet import DataSet
    >>> DS1 = DataSet((Ny, Ny, Nz), (X0, Y0, Z0), (X1, Y1, Z1), "CAR")
    >>> # FunctionX, FunctionY, and FunctionZ are functions to evluate each component of
    >>> # vector field in different directions and on Cell center of mesh structure
    >>> DS1.Scalar("XComponent", "Cells", FunctionX)
    >>> DS1.Scalar("YComponent", "Cells", FunctionY)
    >>> DS1.Scalar("ZComponent", "Cells", FunctionZ)
    >>> DS1.DivCell("XComponent", "YComponent", "ZComponent", "NewVar")
    >>> DS1.Write2HDF5("Filename.h5")