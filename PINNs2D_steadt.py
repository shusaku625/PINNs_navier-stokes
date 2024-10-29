import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time
import vtk
from vtk.util import numpy_support as VN
from network import Swish, MySquared, Net2_p, Net2_u, Net2_v
import pandas as pd
import load_mesh
import compute_loss_function
import output_function

def geo_train(device,x_in,y_in,xb,yb,ub,vb,xd,yd,ud,vd,batchsize,learning_rate,epochs,path,Diff,rho,Lambda_BC,nPt,T,xb_inlet,yb_inlet,ub_inlet,vb_inlet ):
	x = torch.Tensor(x_in).to(device)
	y = torch.Tensor(y_in).to(device)
	xb = torch.Tensor(xb).to(device)
	yb = torch.Tensor(yb).to(device)
	ub = torch.Tensor(ub).to(device)
	vb = torch.Tensor(vb).to(device)
	xd = torch.Tensor(xd).to(device)
	yd = torch.Tensor(yd).to(device)
	ud = torch.Tensor(ud).to(device)
	vd = torch.Tensor(vd).to(device)
	xb_inlet = torch.Tensor(xb_inlet).to(device)
	yb_inlet = torch.Tensor(yb_inlet).to(device)
	ub_inlet = torch.Tensor(ub_inlet).to(device)
	vb_inlet = torch.Tensor(vb_inlet).to(device)
	dataset = TensorDataset(x,y)
	dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0, drop_last = True )

	net2_u = Net2_u().to(device)
	net2_v = Net2_v().to(device)
	net2_p = Net2_p().to(device)
	
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	net2_u.apply(init_normal)
	net2_v.apply(init_normal)
	net2_p.apply(init_normal)

	optimizer_u = optim.Adam(net2_u.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer_v = optim.Adam(net2_v.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer_p = optim.Adam(net2_p.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)

	tic = time.time()

	if(Flag_pretrain):
		print('Reading (pretrain) functions first...')
		net2_u.load_state_dict(torch.load(path+"sten_u" + ".pt"))
		net2_v.load_state_dict(torch.load(path+"sten_v" + ".pt"))
		net2_p.load_state_dict(torch.load(path+"sten_p" + ".pt"))

	if (Flag_schedule):
		scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
		scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
		scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=step_epoch, gamma=decay_rate)

		list_loss_eqn = []
		list_loss_bc = []
		list_loss_data = []

		output_interval = 10
		for epoch in range(epochs):
			loss_eqn_tot = 0.
			loss_bc_tot = 0.
			loss_data_tot = 0.
			n = 0
			for batch_idx, (x_in,y_in) in enumerate(dataloader): 
				net2_u.zero_grad()
				net2_v.zero_grad()
				net2_p.zero_grad()
				loss_eqn = compute_loss_function.criterion(net2_u, net2_v, net2_p, x_in, y_in, X_scale, Y_scale, U_scale, Diff, rho)
				loss_bc = compute_loss_function.Loss_BC(net2_u, net2_v, net2_p,xb,yb,ub,vb,xb_inlet,yb_inlet,ub_inlet,x,y)
				loss_data = compute_loss_function.Loss_data(net2_u, net2_v, net2_p,xd,yd,ud,vd)
				loss = loss_eqn + Lambda_BC* loss_bc + Lambda_data*loss_data
				loss.backward()
				optimizer_u.step() 
				optimizer_v.step()
				optimizer_p.step()  
				loss_eqn_tot += loss_eqn
				loss_bc_tot += loss_bc
				loss_data_tot  += loss_data
				n += 1 
				if batch_idx % 40 ==0:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss eqn {:.10f} Loss BC {:.8f} Loss data {:.8f}'.format(
						epoch, batch_idx * len(x_in), len(dataloader.dataset),
						100. * batch_idx / len(dataloader), loss_eqn.item(), loss_bc.item(),loss_data.item()))
			if (Flag_schedule):
					scheduler_u.step()
					scheduler_v.step()
					scheduler_p.step()
			loss_eqn_tot = loss_eqn_tot / n
			loss_bc_tot = loss_bc_tot / n
			loss_data_tot = loss_data_tot / n

			print('*****Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f} Loss data {:.10f} ****'.format(loss_eqn_tot, loss_bc_tot,loss_data_tot) )
			print('learning rate is ', optimizer_u.param_groups[0]['lr'], optimizer_v.param_groups[0]['lr'])

			if epoch%output_interval==0:
				with torch.no_grad():
					list_loss_eqn.append(loss_eqn_tot.item())
					list_loss_bc.append(loss_bc_tot.item())
					list_loss_data.append(loss_data_tot.item())
					output_function.output_loss_history(list_loss_eqn,list_loss_bc,list_loss_data)
					base_mesh_file = "Data/2D-stenosis/sten_mesh000000.vtu"
					output_function.output_update_vtk(net2_u, net2_v, x, y, epoch, output_interval, base_mesh_file)
					#save network
					torch.save(net2_p.state_dict(),path+"sten_data_p" + ".pt")
					torch.save(net2_u.state_dict(),path+"sten_data_u" + ".pt")
					torch.save(net2_v.state_dict(),path+"sten_data_v" + ".pt")
					print ("Data saved!")
		
	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)
	
device = torch.device("cuda")
Lambda_BC  = 20.
Lambda_data = 1.
Directory = "Data/2D-stenosis/"
mesh_file = Directory + "sten_mesh000000.vtu"
bc_file_in = Directory + "inlet_BC.vtk"
bc_file_wall = Directory + "wall_BC.vtk"
File_data = Directory + "velocity_sten_steady.vtu"
fieldname = 'f_5-0' #The velocity field name in the vtk file (see from ParaView)
batchsize = 256 
learning_rate = 1e-5 
epochs  = 5500 
Flag_pretrain = False # True #If true reads the nets from last run
Diff = 0.001
rho = 1.
T = 0.5 #total duraction
Flag_x_length = True #if True scales the eqn such that the length of the domain is = X_scale
X_scale = 2.0 #The length of the  domain (need longer length for separation region)
Y_scale = 1.0 
U_scale = 1.0
U_BC_in = 0.5
Lambda_div = 1.  #penalty factor for continuity eqn (Makes it worse!?)
Lambda_v = 1.  #penalty factor for y-momentum equation

Flag_schedule = True #If true change the learning rate 
if (Flag_schedule):
	learning_rate = 5e-4 #starting learning rate
	step_epoch = 1200 #1000
	decay_rate = 0.1 # 0.1

if (not Flag_x_length):
	X_scale = 1.
	Y_scale = 1.

x, y = load_mesh.load_domain(mesh_file)
nPt = 130  
xStart = 0.
xEnd = 1.
yStart = 0.
yEnd = 1.0

t = np.linspace(0., T, nPt*nPt)
t=t.reshape(-1, 1)
print('shape of x',x.shape)
print('shape of y',y.shape)

xb_in, yb_in, n_points_inlet = load_mesh.load_inlet_mesh(bc_file_in)
xb_wall, yb_wall, n_points_wall = load_mesh.load_wall_mesh(bc_file_wall)

u_in_BC = (yb_in[:]) * ( 0.3 - yb_in[:] )  / 0.0225 * U_BC_in #parabolic

v_in_BC = np.linspace(0., 0., n_points_inlet)
u_wall_BC = np.linspace(0., 0., n_points_wall)
v_wall_BC = np.linspace(0., 0., n_points_wall)

xb = xb_wall
yb = yb_wall
ub = u_wall_BC
vb = v_wall_BC
xb_inlet = xb_in 
yb_inlet =yb_in 
ub_inlet = u_in_BC
vb_inlet = v_in_BC

xb= xb.reshape(-1, 1) #need to reshape to get 2D array
yb= yb.reshape(-1, 1) #need to reshape to get 2D array
ub= ub.reshape(-1, 1) #need to reshape to get 2D array
vb= vb.reshape(-1, 1) #need to reshape to get 2D array
xb_inlet= xb_inlet.reshape(-1, 1) #need to reshape to get 2D array
yb_inlet= yb_inlet.reshape(-1, 1) #need to reshape to get 2D array
ub_inlet= ub_inlet.reshape(-1, 1) #need to reshape to get 2D array
vb_inlet= vb_inlet.reshape(-1, 1) #need to reshape to get 2D array

print('shape of xb',xb.shape)
print('shape of yb',yb.shape)
print('shape of ub',ub.shape)
path = "Results/"

x_data = [1., 1.2, 1.22, 1.31, 1.39 ] 
y_data =[0.15, 0.07, 0.22, 0.036, 0.26 ]
z_data  = [0.,0.,0.,0.,0. ]

x_data = np.asarray(x_data)  #convert to numpy 
y_data = np.asarray(y_data) #convert to numpy 

print ('Loading', File_data)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(File_data)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print ('n_points of the data file read:' ,n_points)

VTKpoints = vtk.vtkPoints()
for i in range(len(x_data)): 
	VTKpoints.InsertPoint(i, x_data[i] , y_data[i]  , z_data[i])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)

probe = vtk.vtkProbeFilter()
probe.SetInputData(point_data)
probe.SetSourceData(data_vtk)
probe.Update()
array = probe.GetOutput().GetPointData().GetArray(fieldname)
data_vel = VN.vtk_to_numpy(array)

data_vel_u = data_vel[:,0] / U_scale
data_vel_v = data_vel[:,1] / U_scale
x_data = x_data / X_scale
y_data = y_data / Y_scale

print('Using input data pts: pts: ',x_data, y_data)
print('Using input data pts: vel u: ',data_vel_u)
print('Using input data pts: vel v: ',data_vel_v)
xd= x_data.reshape(-1, 1) #need to reshape to get 2D array
yd= y_data.reshape(-1, 1) #need to reshape to get 2D array
ud= data_vel_u.reshape(-1, 1) #need to reshape to get 2D array
vd= data_vel_v.reshape(-1, 1) #need to reshape to get 2D array

geo_train(device,x,y,xb,yb,ub,vb,xd,yd,ud,vd,batchsize,learning_rate,epochs,path,Diff,rho,Lambda_BC,nPt,T,xb_inlet,yb_inlet,ub_inlet,vb_inlet)