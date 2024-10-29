import vtk
import numpy as np

def load_domain(filename):
    print ('Loading', filename)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints()
    print ('n_points of the mesh:' ,n_points)
    x_vtk_mesh = np.zeros((n_points,1))
    y_vtk_mesh = np.zeros((n_points,1))
    VTKpoints = vtk.vtkPoints()
    for i in range(n_points):
    	pt_iso  =  data_vtk.GetPoint(i)
    	x_vtk_mesh[i] = pt_iso[0]	
    	y_vtk_mesh[i] = pt_iso[1]
    	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])

    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)
    x  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
    y  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
    print(x)
    exit(1)
    return x, y

def load_domain_from_vtk(filename):
    print('Loading', filename)
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()

    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints()
    print('n_points of the mesh:', n_points)
    
    if n_points == 0:
        print("Error: No points found in the file.")
        return None

    x_vtk_mesh = np.zeros((n_points, 1))
    y_vtk_mesh = np.zeros((n_points, 1))
    z_vtk_mesh = np.zeros((n_points, 1))

    VTKpoints = vtk.vtkPoints()
    for i in range(n_points):
        pt_iso = data_vtk.GetPoint(i)
        x_vtk_mesh[i] = pt_iso[0]
        y_vtk_mesh[i] = pt_iso[1]
        z_vtk_mesh[i] = pt_iso[2]
        VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])

    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)

    x = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh), 1))
    y = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh), 1))
    z = np.reshape(z_vtk_mesh, (np.size(z_vtk_mesh), 1))

    return x, y, z

def load_inlet_mesh(bc_file_in):
    print ('Loading', bc_file_in)
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(bc_file_in)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints()
    
    print ('n_points of at inlet' ,n_points)
    x_vtk_mesh = np.zeros((n_points,1))
    y_vtk_mesh = np.zeros((n_points,1))
    VTKpoints = vtk.vtkPoints()
    for i in range(n_points):
    	pt_iso  =  data_vtk.GetPoint(i)
    	x_vtk_mesh[i] = pt_iso[0]	
    	y_vtk_mesh[i] = pt_iso[1]
    	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)
    xb_in  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh[:]),1))
    yb_in  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh[:]),1))
    return xb_in, yb_in, n_points

def load_inlet_mesh_from_stl(stl_file_in):
    print('Loading', stl_file_in)
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file_in)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints()
    n_cells = data_vtk.GetNumberOfCells()  # 接続情報を含むセル数（面の数）
    print('n_points of the mesh:', n_points)
    print('n_cells (triangles):', n_cells)
    x_vtk_mesh = np.zeros((n_points, 1))
    y_vtk_mesh = np.zeros((n_points, 1))
    z_vtk_mesh = np.zeros((n_points, 1))
    VTKpoints = vtk.vtkPoints()
    for i in range(n_points):
        pt_iso = data_vtk.GetPoint(i)
        x_vtk_mesh[i] = pt_iso[0]
        y_vtk_mesh[i] = pt_iso[1]
        z_vtk_mesh[i] = pt_iso[2]
        VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)

    # 接続情報（三角形の頂点インデックス）を取得
    connectivity = []
    for i in range(n_cells):
        cell = data_vtk.GetCell(i)  # i番目のセル（三角形要素）
        ids = cell.GetPointIds()    # この三角形の節点インデックス
        connectivity.append([ids.GetId(0), ids.GetId(1), ids.GetId(2)])

    xb_in = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh), 1))
    yb_in = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh), 1))
    zb_in = np.reshape(z_vtk_mesh, (np.size(z_vtk_mesh), 1))
    
    return xb_in, yb_in, zb_in, n_points, connectivity

def load_outlet_mesh_from_stl(stl_file_in):
    print('Loading', stl_file_in)
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file_in)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints()
    n_cells = data_vtk.GetNumberOfCells()  # 接続情報を含むセル数（面の数）
    print('n_points of the mesh:', n_points)
    x_vtk_mesh = np.zeros((n_points, 1))
    y_vtk_mesh = np.zeros((n_points, 1))
    z_vtk_mesh = np.zeros((n_points, 1))
    VTKpoints = vtk.vtkPoints()
    for i in range(n_points):
        pt_iso = data_vtk.GetPoint(i)
        x_vtk_mesh[i] = pt_iso[0]
        y_vtk_mesh[i] = pt_iso[1]
        z_vtk_mesh[i] = pt_iso[2]
        VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)

    # 接続情報（三角形の頂点インデックス）を取得
    connectivity = []
    for i in range(n_cells):
        cell = data_vtk.GetCell(i)  # i番目のセル（三角形要素）
        ids = cell.GetPointIds()    # この三角形の節点インデックス
        connectivity.append([ids.GetId(0), ids.GetId(1), ids.GetId(2)])

    xb_out = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh), 1))
    yb_out = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh), 1))
    zb_out = np.reshape(z_vtk_mesh, (np.size(z_vtk_mesh), 1))
    return xb_out, yb_out, zb_out, n_points, connectivity

def load_wall_mesh(bc_file_wall):
    print ('Loading', bc_file_wall)
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(bc_file_wall)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_pointsw = data_vtk.GetNumberOfPoints()
    print ('n_points of at wall' ,n_pointsw)
    x_vtk_mesh = np.zeros((n_pointsw,1))
    y_vtk_mesh = np.zeros((n_pointsw,1))
    VTKpoints = vtk.vtkPoints()
    for i in range(n_pointsw):
    	pt_iso  =  data_vtk.GetPoint(i)
    	x_vtk_mesh[i] = pt_iso[0]	
    	y_vtk_mesh[i] = pt_iso[1]
    	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)
    xb_wall  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
    yb_wall  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
    return xb_wall, yb_wall, n_pointsw

def load_wall_mesh_from_stl(stl_file_in):
    print('Loading', stl_file_in)
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file_in)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_pointsw = data_vtk.GetNumberOfPoints()
    print('n_points of the mesh:', n_pointsw)
    x_vtk_mesh = np.zeros((n_pointsw, 1))
    y_vtk_mesh = np.zeros((n_pointsw, 1))
    z_vtk_mesh = np.zeros((n_pointsw, 1))
    VTKpoints = vtk.vtkPoints()
    for i in range(n_pointsw):
        pt_iso = data_vtk.GetPoint(i)
        x_vtk_mesh[i] = pt_iso[0]
        y_vtk_mesh[i] = pt_iso[1]
        z_vtk_mesh[i] = pt_iso[2]
        VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)
    xb_in = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh), 1))
    yb_in = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh), 1))
    zb_in = np.reshape(z_vtk_mesh, (np.size(z_vtk_mesh), 1))
    
    return xb_in, yb_in, zb_in, n_pointsw