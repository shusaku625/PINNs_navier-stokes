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
from network import Swish, MySquared, Net2_pt, Net2_ut, Net2_vt, Net2_wt
from network import Net2_p, Net2_u, Net2_v, Net2_w
import pandas as pd
import load_mesh
import compute_loss_function
import output_function
import hydra
from omegaconf import DictConfig, OmegaConf
import os

def geo_train(device,x_in,y_in,z_in,t_in,xb,yb,zb,ub,vb,wb,batchsize,learning_rate,epochs,path,Diff,rho,Lambda_BC,nPt,T,xb_inlet_in,yb_inlet,zb_inlet,xb_outlet,yb_outlet,zb_outlet,ub_inlet,vb_inlet,wb_inlet,p_outlet,connectivity_inlet, connectivity_outlet, Flag_pretrain, Flag_schedule, step_epoch, decay_rate, X_scale, Y_scale, Z_scale, U_scale):
    x = torch.Tensor(x_in).to(device)
    y = torch.Tensor(y_in).to(device)
    z = torch.Tensor(z_in).to(device)
    t = torch.Tensor(t_in).to(device)
    xb = torch.Tensor(xb).to(device)
    yb = torch.Tensor(yb).to(device)
    zb = torch.Tensor(zb).to(device)
    ub = torch.Tensor(ub).to(device)
    vb = torch.Tensor(vb).to(device)
    wb = torch.Tensor(wb).to(device)
    #xd = torch.Tensor(xd).to(device)
    #yd = torch.Tensor(yd).to(device)
    #zd = torch.Tensor(zd).to(device)
    #ud = torch.Tensor(ud).to(device)
    #vd = torch.Tensor(vd).to(device)
    #wd = torch.Tensor(wd).to(device)
    xb_inlet = torch.Tensor(xb_inlet_in).to(device)
    yb_inlet = torch.Tensor(yb_inlet).to(device)
    zb_inlet = torch.Tensor(zb_inlet).to(device)
    xb_outlet = torch.Tensor(xb_outlet).to(device)
    yb_outlet = torch.Tensor(yb_outlet).to(device)
    zb_outlet = torch.Tensor(zb_outlet).to(device)
    ub_inlet = torch.Tensor(ub_inlet).to(device)
    vb_inlet = torch.Tensor(vb_inlet).to(device)
    wb_inlet = torch.Tensor(wb_inlet).to(device)
    p_outlet = torch.Tensor(p_outlet).to(device)
    #expand
    #domain
    x_expand = x.repeat(t.size(0), 1)  # shape: [24308 * 100, 1]
    y_expand = y.repeat(t.size(0), 1)  # shape: [24308 * 100, 1]
    z_expand = z.repeat(t.size(0), 1)  # shape: [24308 * 100, 1]
    t_expand = t.repeat_interleave(x.size(0), dim=0)  # shape: [24308 * 100, 1]
    #wall
    xb_expand = xb.repeat(t.size(0), 1)
    yb_expand = yb.repeat(t.size(0), 1)
    zb_expand = zb.repeat(t.size(0), 1)
    tb_expand = t.repeat_interleave(xb.size(0), dim=0)
    ub_expand = ub.repeat(t.size(0), 1)
    vb_expand = vb.repeat(t.size(0), 1)
    wb_expand = wb.repeat(t.size(0), 1)
    #inlet
    xb_inlet_expand = xb_inlet.repeat(t.size(0), 1)
    yb_inlet_expand = yb_inlet.repeat(t.size(0), 1)
    zb_inlet_expand = zb_inlet.repeat(t.size(0), 1)
    tb_inlet_expand = t.repeat_interleave(xb_inlet.size(0), dim=0)
    ub_inlet_expand = ub_inlet.repeat(t.size(0), 1)
    vb_inlet_expand = vb_inlet.repeat(t.size(0), 1)
    wb_inlet_expand = wb_inlet.repeat_interleave(xb_inlet.size(0), dim=0)
    #outlet
    xb_outlet_expand = xb_inlet.repeat(t.size(0), 1)
    yb_outlet_expand = yb_inlet.repeat(t.size(0), 1)
    zb_outlet_expand = zb_inlet.repeat(t.size(0), 1)
    p_outlet_expand = p_outlet.repeat(t.size(0), 1)
    tb_outlet_expand = t.repeat_interleave(xb_outlet.size(0), dim=0)

    dataset = TensorDataset(x_expand,y_expand,z_expand,t_expand)
    dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0, drop_last = True )

    net2_ut = Net2_ut().to(device)
    net2_vt = Net2_vt().to(device)
    net2_wt = Net2_wt().to(device)
    net2_pt = Net2_pt().to(device)

    def init_normal(m):
    	if type(m) == nn.Linear:
    		nn.init.kaiming_normal_(m.weight)   
    net2_ut.apply(init_normal)
    net2_vt.apply(init_normal)
    net2_wt.apply(init_normal)
    net2_pt.apply(init_normal)   
    optimizer_u = optim.Adam(net2_ut.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_v = optim.Adam(net2_vt.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_w = optim.Adam(net2_wt.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_p = optim.Adam(net2_pt.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)   
    tic = time.time()   
    if(Flag_pretrain):
    	print('Reading (pretrain) functions first...')
    	net2_u.load_state_dict(torch.load(path+"sten_data_u" + ".pt"))
    	net2_v.load_state_dict(torch.load(path+"sten_data_v" + ".pt"))
    	net2_w.load_state_dict(torch.load(path+"sten_data_w" + ".pt"))
    	net2_p.load_state_dict(torch.load(path+"sten_data_p" + ".pt"))  
    if (Flag_schedule):
        scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
        scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
        scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_w, step_size=step_epoch, gamma=decay_rate)
        scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=step_epoch, gamma=decay_rate)  
        list_loss_eqn = []
        list_loss_bc = []
        list_loss_flowrate = []
        list_loss_data = [] 
        output_interval = 20
        for epoch in range(epochs):
            loss_eqn_tot = 0.
            loss_bc_tot = 0.
            loss_flowrate_tot = 0
            #loss_data_tot = 0.
            n = 0
            for batch_idx, (x_in,y_in,z_in,t_in) in enumerate(dataloader): 
                net2_ut.zero_grad()
                net2_vt.zero_grad()
                net2_wt.zero_grad()
                net2_pt.zero_grad()
                loss_eqn = compute_loss_function.criterion3DTime(net2_ut, net2_vt, net2_wt, net2_pt, x_in, y_in, z_in, t_in, X_scale, Y_scale, Z_scale, U_scale, Diff, rho)
                loss_bc = compute_loss_function.Loss_BC3DTime(net2_ut,net2_vt,net2_wt,net2_pt,\
                xb_expand,yb_expand,zb_expand,tb_expand,ub_expand,vb_expand,wb_expand,\
                xb_inlet_expand,yb_inlet_expand,zb_inlet_expand,tb_inlet_expand,\
                xb_outlet_expand,yb_outlet_expand,zb_outlet_expand,tb_outlet_expand,ub_inlet_expand,vb_inlet_expand,wb_inlet_expand,p_outlet_expand)
                #loss_bc = compute_loss_function.Loss_BC3D(net2_u, net2_v, net2_w, net2_p,xb,yb,zb, ub,vb,wb,xb_inlet,yb_inlet,zb_inlet,xb_outlet,yb_outlet,zb_outlet,ub_inlet,vb_inlet,wb_inlet, p_outlet)
                #loss_flow_rate = compute_loss_function.Loss_flow_rate(net2_u, net2_v, net2_w,xb_inlet,yb_inlet,zb_inlet,tb_xb_outlet,yb_outlet,zb_outlet,connectivity_inlet, connectivity_outlet)
                #loss_data = compute_loss_function.Loss_data(net2_u, net2_v, net2_p,xd,yd,ud,vd)
                #loss = loss_eqn + Lambda_BC* loss_bc + Lambda_data*loss_data
                loss = loss_eqn + Lambda_BC* loss_bc
                loss.backward()
                optimizer_u.step() 
                optimizer_v.step()
                optimizer_w.step()
                optimizer_p.step()  
                loss_eqn_tot += loss_eqn
                loss_bc_tot += loss_bc
                #loss_flowrate_tot += loss_flow_rate
                #loss_data_tot  += loss_data
                n += 1 
                if batch_idx % 40 ==0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss eqn {:.10f} Loss BC {:.8f}'.format(
                    	epoch, batch_idx * len(x_in), len(dataloader.dataset),
                    	100. * batch_idx / len(dataloader), loss_eqn.item(), loss_bc.item()))
            if (Flag_schedule):
                scheduler_u.step()
                scheduler_v.step()
                scheduler_w.step()
                scheduler_p.step()
            loss_eqn_tot = loss_eqn_tot / n
            loss_bc_tot = loss_bc_tot / n
            #loss_flowrate_tot = loss_flowrate_tot / n
            #loss_data_tot = loss_data_tot / n  
            print('*****Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f}****'.format(loss_eqn_tot, loss_bc_tot) )
            print('learning rate is ', optimizer_u.param_groups[0]['lr'], optimizer_v.param_groups[0]['lr'], optimizer_w.param_groups[0]['lr']) 
            if epoch%output_interval==0:
                with torch.no_grad():
                    list_loss_eqn.append(loss_eqn_tot.item())
                    list_loss_bc.append(loss_bc_tot.item())
                    #list_loss_flowrate.append(loss_flowrate_tot.item())
                    #list_loss_data.append(loss_data_tot.item())
                    output_function.output_loss_historyTime(list_loss_eqn,list_loss_bc)
                    base_mesh_file = "bend_mesh/bend_pipe.vtk"
                    output_function.output_update_vtk3DTime(net2_ut, net2_vt, net2_wt, net2_pt, x_expand, y_expand, z_expand, t_expand, epoch, output_interval, base_mesh_file, nPt, path)
                    #save network
                    torch.save(net2_pt.state_dict(),path+"sten_data_p" + ".pt")
                    torch.save(net2_ut.state_dict(),path+"sten_data_u" + ".pt")
                    torch.save(net2_vt.state_dict(),path+"sten_data_v" + ".pt")
                    torch.save(net2_vt.state_dict(),path+"sten_data_w" + ".pt")
                    print ("Data saved!")
    
    toc = time.time()
    elapseTime = toc - tic
    print ("elapse time in parallel = ", elapseTime)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    device = torch.device(cfg.resource.device)
    Lambda_BC  = cfg.weight.Lambda_BC
    Lambda_data = cfg.weight.Lambda_data
    Lambda_div = cfg.weight.Lambda_div  #penalty factor for continuity eqn (Makes it worse!?)
    Directory = cfg.input_mesh.path
    mesh_file = Directory + cfg.input_mesh.domain_mesh
    bc_file_in = Directory + cfg.input_mesh.inlet_mesh
    bc_file_wall = Directory + cfg.input_mesh.wall_mesh
    bc_file_out = Directory + cfg.input_mesh.outlet_mesh
    batchsize = cfg.parameter.batch_size
    learning_rate = cfg.parameter.learning_rate
    epochs  = cfg.parameter.epochs
    Flag_pretrain = cfg.Flag.pretrain
    Diff = cfg.physical_parameter.diffusion_coefficient
    rho = cfg.physical_parameter.rho
    T = cfg.physical_parameter.time_duration #total duraction
    X_scale = cfg.scaling.x_scale #The length of the  domain (need longer length for separation region)
    Y_scale = cfg.scaling.y_scale 
    Z_scale = cfg.scaling.z_scale 
    U_scale = cfg.scaling.U_scale
    outputDir = cfg.output.path
    os.makedirs(outputDir, exist_ok=True) 
    
    nPt = cfg.physical_parameter.time_point
    Flag_schedule = cfg.Flag.schedule #If true change the learning rate 

    if (Flag_schedule):
    	learning_rate = cfg.schedule.start_learning_rate #starting learning rate
    	step_epoch = cfg.schedule.step_epoch #1000
    	decay_rate = cfg.schedule.decay_rate # 0.1

    center_x = cfg.scaling.x_center
    center_y = cfg.scaling.y_center
    center_z = cfg.scaling.z_center
    x, y, z = load_mesh.load_domain_from_vtk(mesh_file)
    x = (x - center_x) / X_scale
    y = (y - center_y) / Y_scale
    z = (z - center_z) / Z_scale

    t = np.linspace(0., T, nPt)

    print('shape of x',x.shape)
    print('shape of y',y.shape)
    print('shape of y',z.shape)

    xb_in, yb_in, zb_in, n_points_inlet, connectivity_inlet = load_mesh.load_inlet_mesh_from_stl(bc_file_in)
    xb_out, yb_out, zb_out, n_points_outlet, connectivity_outlet = load_mesh.load_outlet_mesh_from_stl(bc_file_out)
    xb_wall, yb_wall, zb_wall, n_points_wall = load_mesh.load_wall_mesh_from_stl(bc_file_wall)

    #inlet
    frequency = 10.0        # サイン波の周波数
    amplitude = 1.0        # サイン波の振幅
    u_in_BC = np.linspace(0., 0., n_points_inlet)
    v_in_BC = np.linspace(0., 0., n_points_inlet)
    w_in_BC = amplitude * np.sin(frequency * t) + 1.0       # z方向の速度（サイン波）

    #wall boundary
    u_wall_BC = np.linspace(0., 0., n_points_wall)
    v_wall_BC = np.linspace(0., 0., n_points_wall)
    w_wall_BC = np.linspace(0., 0., n_points_wall)

    #outlet boundary
    p_out_BC = np.linspace(0., 0., n_points_outlet)

    xb = (xb_wall - center_x) / X_scale
    yb = (yb_wall - center_y) / Y_scale
    zb = (zb_wall - center_z) / Z_scale
    ub = u_wall_BC
    vb = v_wall_BC
    wb = w_wall_BC
    xb_inlet = (xb_in - center_x) / X_scale
    yb_inlet = (yb_in - center_y) / Y_scale
    zb_inlet = (zb_in - center_z) / Z_scale
    xb_outlet = (xb_out - center_x) / X_scale
    yb_outlet = (yb_out - center_y) / Y_scale
    zb_outlet = (zb_out - center_z) / Z_scale
    ub_inlet = u_in_BC
    vb_inlet = v_in_BC
    wb_inlet = w_in_BC
    p_outlet = p_out_BC

    xb= xb.reshape(-1, 1) #need to reshape to get 2D array
    yb= yb.reshape(-1, 1) #need to reshape to get 2D array
    zb= zb.reshape(-1, 1) #need to reshape to get 2D array
    ub= ub.reshape(-1, 1) #need to reshape to get 2D array
    vb= vb.reshape(-1, 1) #need to reshape to get 2D array
    wb= wb.reshape(-1, 1) #need to reshape to get 2D array
    t = t.reshape(-1,1)
    xb_inlet= xb_inlet.reshape(-1, 1) #need to reshape to get 2D array
    yb_inlet= yb_inlet.reshape(-1, 1) #need to reshape to get 2D array
    zb_inlet= zb_inlet.reshape(-1, 1) #need to reshape to get 2D array
    xb_outlet= xb_outlet.reshape(-1, 1) #need to reshape to get 2D array
    yb_outlet= yb_outlet.reshape(-1, 1) #need to reshape to get 2D array
    zb_outlet= zb_outlet.reshape(-1, 1) #need to reshape to get 2D array
    ub_inlet= ub_inlet.reshape(-1, 1) #need to reshape to get 2D array
    vb_inlet= vb_inlet.reshape(-1, 1) #need to reshape to get 2D array
    wb_inlet= wb_inlet.reshape(-1, 1) #need to reshape to get 2D array
    p_outlet= p_outlet.reshape(-1, 1) #need to reshape to get 2D array
    print(ub_inlet.shape)
    print(wb_inlet.shape)
    print('shape of xb',x.shape)
    print('shape of yb',y.shape)
    print('shape of yb',z.shape)
    print('shape of t',t.shape)
    path = cfg.output.path

    geo_train(device,x,y,z,t,xb,yb,zb,ub,vb,wb,batchsize,learning_rate,epochs,path,Diff,rho,Lambda_BC,nPt,T,xb_inlet,yb_inlet,zb_inlet,xb_outlet,yb_outlet,zb_outlet,ub_inlet,vb_inlet,wb_inlet,p_outlet, connectivity_inlet, connectivity_outlet, Flag_pretrain, Flag_schedule, step_epoch, decay_rate, X_scale, Y_scale, Z_scale, U_scale)

if __name__ == "__main__":
    my_app()