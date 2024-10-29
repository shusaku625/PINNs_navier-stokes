import pandas as pd
import torch
import numpy as np
import vtk

def output_loss_history(list_loss_eqn,list_loss_bc,list_loss_flowrate):
    # 3つのリストをデータフレームに変換
	df = pd.DataFrame({
    	'loss_eqn': list_loss_eqn,
    	'loss_bc': list_loss_bc,
        'loss_flowrate': list_loss_flowrate,
	})
	df.to_csv('loss_out.csv')

def output_loss_historyTime(list_loss_eqn,list_loss_bc):
    # 3つのリストをデータフレームに変換
	df = pd.DataFrame({
    	'loss_eqn': list_loss_eqn,
    	'loss_bc': list_loss_bc,
	})
	df.to_csv('loss_out.csv')

def output_update_vtk(net2_u, net2_v, x, y, epoch, output_interval, base_mesh_file):
    net_in = torch.cat((x,y),1)
    output_u = net2_u(net_in)  #evaluate model (runs out of memory for large GPU problems!)
    output_v = net2_v(net_in)  #evaluate model
    output_u_tmp = output_u.cpu().data.numpy() #need to convert to cpu before converting to numpy
    output_v_tmp = output_v.cpu().data.numpy()
    x_tmp = x.cpu()
    y_tmp = y.cpu()
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(base_mesh_file)
    reader.Update()
    mesh = reader.GetOutput()
    num_points = mesh.GetNumberOfPoints()
    print(len(x_tmp))
    if len(output_u_tmp) != num_points or len(output_v_tmp) != num_points:
    	print(len(x_tmp))
    	raise ValueError("u, vの配列サイズはメッシュの点数と一致している必要があります。")
    velocity = vtk.vtkDoubleArray()
    velocity.SetNumberOfComponents(3)  # 3次元ベクトル (u, v, 0) のための3成分
    velocity.SetName("Velocity")
    for i in range(num_points):
        velocity.InsertNextTuple([output_u_tmp[i][0], output_v_tmp[i][0], 0.0])  # 2Dメッシュのためz方				
    mesh.GetPointData().AddArray(velocity)
    output_file_name = "Results_with_flow_rate/test_{}.vtu".format(int(epoch/output_interval))
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_file_name)
    writer.SetInputData(mesh)
    writer.Write()

def output_update_vtk3D(net2_u, net2_v, net2_w, net2_p, x, y, z, epoch, output_interval, base_mesh_file):
    net_in = torch.cat((x,y,z),1)
    output_u = net2_u(net_in)  #evaluate model (runs out of memory for large GPU problems!)
    output_v = net2_v(net_in)  #evaluate model
    output_w = net2_w(net_in)  #evaluate model
    output_p = net2_p(net_in)  #evaluate model
    output_u_tmp = output_u.cpu().data.numpy() #need to convert to cpu before converting to numpy
    output_v_tmp = output_v.cpu().data.numpy()
    output_w_tmp = output_w.cpu().data.numpy()
    output_p_tmp = output_p.cpu().data.numpy()
    x_tmp = x.cpu()
    y_tmp = y.cpu()
    z_tmp = z.cpu()
    print('Loading', base_mesh_file)
        
    # ファイル形式に応じたリーダーを選択
    if base_mesh_file.endswith('.vtu'):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif base_mesh_file.endswith('.vtk'):
        reader = vtk.vtkUnstructuredGridReader()
    else:
        raise ValueError("Unsupported file format. Please provide a .vtu or .vtk file.")
    # ファイルを読み込む
    reader.SetFileName(base_mesh_file)
    reader.Update()
    # メッシュデータを取得
    mesh = reader.GetOutput()
    num_points = mesh.GetNumberOfPoints()
    print(f'Number of points in mesh: {num_points}')
    # 読み込んだメッシュの点数とu, v, w配列、圧力配列の長さを確認
    if (len(output_u_tmp) != num_points or 
        len(output_v_tmp) != num_points or 
        len(output_w_tmp) != num_points or 
        len(output_p_tmp) != num_points):
        
        print("Mismatch between number of points in mesh and input arrays.")
        print(f'Expected number of points: {num_points}')
        print(f'u array size: {len(output_u_tmp)}')
        print(f'v array size: {len(output_v_tmp)}')
        print(f'w array size: {len(output_w_tmp)}')
        print(f'pressure array size: {len(output_p_tmp)}')
        raise ValueError("u, v, w, pressure arrays must have the same length as the number of points in the mesh.")
    # 速度ベクトル配列を作成
    velocity = vtk.vtkDoubleArray()
    velocity.SetNumberOfComponents(3)  # 3次元ベクトル (u, v, w)
    velocity.SetName("Velocity")
    # 圧力スカラー配列を作成
    pressure = vtk.vtkDoubleArray()
    pressure.SetNumberOfComponents(1)  # 圧力はスカラー
    pressure.SetName("Pressure")
    # 各点の速度と圧力を設定
    for i in range(num_points):
        velocity.InsertNextTuple([output_u_tmp[i][0], output_v_tmp[i][0], output_w_tmp[i][0]])
        pressure.InsertNextValue(output_p_tmp[i][0])
    # メッシュに速度データと圧力データを追加
    mesh.GetPointData().AddArray(velocity)
    mesh.GetPointData().AddArray(pressure)
    # 結果をファイルに書き込む
    output_file_name = f"Results_with_flow_rate/test_{int(epoch/output_interval)}.vtu"
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_file_name)
    writer.SetInputData(mesh)
    writer.Write()

def output_update_vtk3DTime(net2_u, net2_v, net2_w, net2_p, x, y, z, t, epoch, output_interval, base_mesh_file, nPt, path):
    net_in = torch.cat((x,y,z,t),1)
    batch_size = 1024
    output_u_all, output_v_all, output_w_all, output_p_all = [], [], [], []
    # net_in のデータをバッチごとに処理
    for i in range(0, net_in.size(0), batch_size):
        # バッチを作成
        batch_in = net_in[i:i+batch_size]
        # モデルを評価モードに設定（ドロップアウトやバッチノルムを無効化）
        net2_u.eval()
        net2_v.eval()
        net2_w.eval()
        net2_p.eval()
        with torch.no_grad(): 
            output_u = net2_u(batch_in)  #evaluate model (runs out of memory for large GPU problems!)
            output_v = net2_v(batch_in)  #evaluate model
            output_w = net2_w(batch_in)  #evaluate model
            output_p = net2_p(batch_in)  #evaluate model

            # 結果をリストに追加
            output_u_all.append(output_u)
            output_v_all.append(output_v)
            output_w_all.append(output_w)
            output_p_all.append(output_p)

    # 各時間ステップでの速度・圧力の推論結果を結合
    output_u_all = torch.cat(output_u_all, dim=0)
    output_v_all = torch.cat(output_v_all, dim=0)
    output_w_all = torch.cat(output_w_all, dim=0)
    output_p_all = torch.cat(output_p_all, dim=0)
    output_u_tmp = output_u_all.cpu().data.numpy() #need to convert to cpu before converting to numpy
    output_v_tmp = output_v_all.cpu().data.numpy()
    output_w_tmp = output_w_all.cpu().data.numpy()
    output_p_tmp = output_p_all.cpu().data.numpy()
    x_tmp = x.cpu()
    y_tmp = y.cpu()
    z_tmp = z.cpu()

    output_u_split = np.array_split(output_u_tmp, nPt)
    output_v_split = np.array_split(output_v_tmp, nPt)
    output_w_split = np.array_split(output_w_tmp, nPt)
    output_p_split = np.array_split(output_p_tmp, nPt)

    
    for time in range(0, nPt):
        #print('Loading', base_mesh_file)
        if base_mesh_file.endswith('.vtu'):
            reader = vtk.vtkXMLUnstructuredGridReader()
        elif base_mesh_file.endswith('.vtk'):
            reader = vtk.vtkUnstructuredGridReader()
        else:
            raise ValueError("Unsupported file format. Please provide a .vtu or .vtk file.")
        reader.SetFileName(base_mesh_file)
        reader.Update()
        mesh = reader.GetOutput()
        num_points = mesh.GetNumberOfPoints()
        #print(f'Number of points in mesh: {num_points}')
        velocity = vtk.vtkDoubleArray()
        velocity.SetNumberOfComponents(3)  # 3次元ベクトル (u, v, w)
        velocity.SetName("Velocity")
        pressure = vtk.vtkDoubleArray()
        pressure.SetNumberOfComponents(1)  # 圧力はスカラー
        pressure.SetName("Pressure")
        for j in range(num_points):
            velocity.InsertNextTuple([output_u_split[time][j][0], output_v_split[time][j][0], output_w_split[time][j][0]])
            pressure.InsertNextValue(output_p_split[time][j][0])
        # メッシュに速度データと圧力データを追加
        mesh.GetPointData().AddArray(velocity)
        mesh.GetPointData().AddArray(pressure)
        # 結果をファイルに書き込む
        output_file_name = path+f"test_{time}.vtu"
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(output_file_name)
        writer.SetInputData(mesh)
        writer.Write()