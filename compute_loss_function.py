import torch
import torch.nn as nn

def criterion(net2_u, net2_v, net2_p, x, y, X_scale, Y_scale, U_scale, Diff, rho):
    x.requires_grad = True
    y.requires_grad = True
    net_in = torch.cat((x,y),1)
    u = net2_u(net_in)
    u = u.view(len(u),-1)
    v = net2_v(net_in)
    v = v.view(len(v),-1)
    P = net2_p(net_in)
    P = P.view(len(P),-1)
    u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    P_x = torch.autograd.grad(P,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    P_y = torch.autograd.grad(P,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    XX_scale = U_scale * (X_scale**2)
    YY_scale = U_scale * (Y_scale**2)
    UU_scale  = U_scale **2
    loss_2 = u*u_x / X_scale + v*u_y / Y_scale - Diff*( u_xx/XX_scale  + u_yy /YY_scale  )+ 1/rho* (P_x / (X_scale*UU_scale)   )  #X-dir
    loss_1 = u*v_x / X_scale + v*v_y / Y_scale - Diff*( v_xx/ XX_scale + v_yy / YY_scale )+ 1/rho*(P_y / (Y_scale*UU_scale)   ) #Y-dir
    loss_3 = (u_x / X_scale + v_y / Y_scale) #continuity
    loss_f = nn.MSELoss()
    loss = loss_f(loss_1,torch.zeros_like(loss_1))+  loss_f(loss_2,torch.zeros_like(loss_2))+  loss_f(loss_3,torch.zeros_like(loss_3))
    return loss

def criterion3D(net2_u, net2_v, net2_w, net2_p, x, y, z, X_scale, Y_scale, Z_scale, U_scale, Diff, rho):
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    net_in = torch.cat((x,y,z),1)
    u = net2_u(net_in)
    u = u.view(len(u),-1)
    v = net2_v(net_in)
    v = v.view(len(v),-1)
    w = net2_w(net_in)
    w = w.view(len(w),-1)
    P = net2_p(net_in)
    P = P.view(len(P),-1)
    u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    u_z = torch.autograd.grad(u,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    u_zz = torch.autograd.grad(u_z,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    v_z = torch.autograd.grad(v,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    v_zz = torch.autograd.grad(v_z,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    w_x = torch.autograd.grad(w,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    w_xx = torch.autograd.grad(w_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    w_y = torch.autograd.grad(w,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    w_yy = torch.autograd.grad(w_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    w_z = torch.autograd.grad(w,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    w_zz = torch.autograd.grad(w_z,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    P_x = torch.autograd.grad(P,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    P_y = torch.autograd.grad(P,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    P_z = torch.autograd.grad(P,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    XX_scale = U_scale * (X_scale**2)
    YY_scale = U_scale * (Y_scale**2)
    ZZ_scale = U_scale * (Z_scale**2)
    UU_scale  = U_scale **2
    loss_1 = u*u_x / X_scale + v*u_y / Y_scale + w*u_z / Z_scale - Diff*(u_xx/XX_scale + u_yy / YY_scale + u_zz / ZZ_scale)+ 1/rho* (P_x / (X_scale*UU_scale)   )  #X-dir
    loss_2 = u*v_x / X_scale + v*v_y / Y_scale + w*v_z / Z_scale - Diff*(v_xx/XX_scale + v_yy / YY_scale + v_zz / ZZ_scale)+ 1/rho*(P_y / (Y_scale*UU_scale)   ) #Y-dir
    loss_3 = u*w_x / X_scale + v*w_y / Y_scale + w*w_z / Z_scale - Diff*(w_xx/XX_scale + w_yy / YY_scale + w_zz / ZZ_scale)+ 1/rho*(P_z / (Z_scale*UU_scale)   ) #Y-dir
    loss_4 = (u_x / X_scale + v_y / Y_scale + w_z / Z_scale) #continuity
    loss_f = nn.MSELoss()
    loss = loss_f(loss_1,torch.zeros_like(loss_1))+  loss_f(loss_2,torch.zeros_like(loss_2))+  loss_f(loss_3,torch.zeros_like(loss_3))+  100*loss_f(loss_4,torch.zeros_like(loss_4))
    return loss

def criterion3DTime(net2_ut, net2_vt, net2_wt, net2_pt, x, y, z, t, X_scale, Y_scale, Z_scale, U_scale, Diff, rho):
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    t.requires_grad = True
    net_in = torch.cat((x,y,z,t),1)
    u = net2_ut(net_in)
    u = u.view(len(u),-1)
    v = net2_vt(net_in)
    v = v.view(len(v),-1)
    w = net2_wt(net_in)
    w = w.view(len(w),-1)
    P = net2_pt(net_in)
    P = P.view(len(P),-1)
    u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    u_z = torch.autograd.grad(u,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    u_zz = torch.autograd.grad(u_z,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    v_t = torch.autograd.grad(v,t,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    v_z = torch.autograd.grad(v,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    v_zz = torch.autograd.grad(v_z,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    w_t = torch.autograd.grad(w,t,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    w_x = torch.autograd.grad(w,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    w_xx = torch.autograd.grad(w_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    w_y = torch.autograd.grad(w,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    w_yy = torch.autograd.grad(w_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    w_z = torch.autograd.grad(w,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    w_zz = torch.autograd.grad(w_z,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    P_x = torch.autograd.grad(P,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    P_y = torch.autograd.grad(P,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    P_z = torch.autograd.grad(P,z,grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]
    XX_scale = U_scale * (X_scale**2)
    YY_scale = U_scale * (Y_scale**2)
    ZZ_scale = U_scale * (Z_scale**2)
    UU_scale  = U_scale **2
    loss_1 = u_t + u*u_x / X_scale + v*u_y / Y_scale + w*u_z / Z_scale - Diff*(u_xx/XX_scale + u_yy / YY_scale + u_zz / ZZ_scale)+ 1/rho* (P_x / (X_scale*UU_scale)   )  #X-dir
    loss_2 = v_t + u*v_x / X_scale + v*v_y / Y_scale + w*v_z / Z_scale - Diff*(v_xx/XX_scale + v_yy / YY_scale + v_zz / ZZ_scale)+ 1/rho*(P_y / (Y_scale*UU_scale)   ) #Y-dir
    loss_3 = w_t + u*w_x / X_scale + v*w_y / Y_scale + w*w_z / Z_scale - Diff*(w_xx/XX_scale + w_yy / YY_scale + w_zz / ZZ_scale)+ 1/rho*(P_z / (Z_scale*UU_scale)   ) #Y-dir
    loss_4 = (u_x / X_scale + v_y / Y_scale + w_z / Z_scale) #continuity
    loss_f = nn.MSELoss()
    loss = loss_f(loss_1,torch.zeros_like(loss_1))+  loss_f(loss_2,torch.zeros_like(loss_2))+  loss_f(loss_3,torch.zeros_like(loss_3))+  100*loss_f(loss_4,torch.zeros_like(loss_4))
    return loss

def Loss_BC(net2_u, net2_v, net2_p, xb,yb,ub,vb, xb_inlet, yb_inlet, ub_inlet, x, y ):
	net_in1 = torch.cat((xb, yb), 1)
	out1_u = net2_u(net_in1)
	out1_v = net2_v(net_in1)
	out1_u = out1_u.view(len(out1_u), -1)
	out1_v = out1_v.view(len(out1_v), -1)
	loss_f = nn.MSELoss()
	loss_noslip = loss_f(out1_u, torch.zeros_like(out1_u)) + loss_f(out1_v, torch.zeros_like(out1_v)) 
	return loss_noslip

def Loss_BC3D(net2_u, net2_v, net2_w, net2_p, xb,yb,zb,ub,vb,wb, xb_inlet, yb_inlet,zb_inlet, xb_outlet,yb_outlet,zb_outlet,ub_inlet,vb_inlet,wb_inlet,p_outlet):
    net_in1 = torch.cat((xb, yb, zb), 1)
    out1_u = net2_u(net_in1)
    out1_v = net2_v(net_in1)
    out1_w = net2_w(net_in1)
    out1_u = out1_u.view(len(out1_u), -1)
    out1_v = out1_v.view(len(out1_v), -1)
    out1_w = out1_w.view(len(out1_w), -1)
    loss_f = nn.MSELoss()
    loss_noslip = loss_f(out1_u, torch.zeros_like(out1_u)) + loss_f(out1_v, torch.zeros_like(out1_v)) + loss_f(out1_w, torch.zeros_like(out1_w)) 

    net_in2 = torch.cat((xb_inlet, yb_inlet, zb_inlet), 1)
    out2_u = net2_u(net_in2)
    out2_v = net2_v(net_in2)
    out2_w = net2_w(net_in2)
    out2_u = out2_u.view(len(out2_u), -1)
    out2_v = out2_v.view(len(out2_v), -1)
    out2_w = out2_w.view(len(out2_w), -1)
    loss_inlet = loss_f(out2_u, ub_inlet) + loss_f(out2_v, vb_inlet) + loss_f(out2_w, wb_inlet) 

    net_in3 = torch.cat((xb_outlet, yb_outlet, zb_outlet), 1)
    out3_p = net2_p(net_in3)
    out3_p = out3_p.view(len(out3_p), -1)
    loss_outlet = loss_f(out3_p, p_outlet)

    return loss_noslip+loss_inlet+loss_outlet

def Loss_BC3DTime(net2_ut,net2_vt,net2_wt,net2_pt,xb,yb,zb,t,ub,vb,wb,xb_inlet,yb_inlet,zb_inlet,tb_inlet,xb_outlet,yb_outlet,zb_outlet,tb_outlet,ub_inlet,vb_inlet,wb_inlet,p_outlet):
    #no-slip
    net_in1 = torch.cat((xb, yb, zb, t), 1)
    out1_u = net2_ut(net_in1)
    out1_v = net2_vt(net_in1)
    out1_w = net2_wt(net_in1)
    out1_u = out1_u.view(len(out1_u), -1)
    out1_v = out1_v.view(len(out1_v), -1)
    out1_w = out1_w.view(len(out1_w), -1)
    loss_f = nn.MSELoss()
    loss_noslip = loss_f(out1_u, torch.zeros_like(out1_u)) + loss_f(out1_v, torch.zeros_like(out1_v)) + loss_f(out1_w, torch.zeros_like(out1_w)) 
    #inlet
    net_in2 = torch.cat((xb_inlet, yb_inlet, zb_inlet, tb_inlet), 1)
    out2_u = net2_ut(net_in2)
    out2_v = net2_vt(net_in2)
    out2_w = net2_wt(net_in2)
    out2_u = out2_u.view(len(out2_u), -1)
    out2_v = out2_v.view(len(out2_v), -1)
    out2_w = out2_w.view(len(out2_w), -1)
    loss_inlet = loss_f(out2_u, ub_inlet) + loss_f(out2_v, vb_inlet) + loss_f(out2_w, wb_inlet) 
    #outlet
    net_in3 = torch.cat((xb_outlet, yb_outlet, zb_outlet, tb_outlet), 1)
    out3_p = net2_pt(net_in3)
    out3_p = out3_p.view(len(out3_p), -1)
    loss_outlet = loss_f(out3_p, p_outlet)

    return loss_noslip+loss_inlet+loss_outlet

def Loss_data(net2_u, net2_v, net2_p, xd, yd, ud, vd):
	net_in1 = torch.cat((xd, yd), 1)
	out1_u = net2_u(net_in1)
	out1_v = net2_v(net_in1)
	out1_u = out1_u.view(len(out1_u), -1)
	out1_v = out1_v.view(len(out1_v), -1)
	loss_f = nn.MSELoss()
	loss_d = loss_f(out1_u, ud) + loss_f(out1_v, vd) 
	return loss_d

def Loss_flow_rate(net2_u, net2_v, net2_w,xb_inlet,yb_inlet,zb_inlet,xb_outlet,yb_outlet,zb_outlet,connectivity_inlet, connectivity_outlet):
    net_in1 = torch.cat((xb_inlet, yb_inlet, zb_inlet), 1)
    out1_u = net2_u(net_in1)
    out1_v = net2_v(net_in1)
    out1_w = net2_w(net_in1)
    out1_u = out1_u.view(len(out1_u), -1)
    out1_v = out1_v.view(len(out1_v), -1)
    out1_w = out1_w.view(len(out1_w), -1)

    net_in2 = torch.cat((xb_outlet, yb_outlet, zb_outlet), 1)
    out2_u = net2_u(net_in2)
    out2_v = net2_v(net_in2)
    out2_w = net2_w(net_in2)
    out2_u = out2_u.view(len(out2_u), -1)
    out2_v = out2_v.view(len(out2_v), -1)
    out2_w = out2_w.view(len(out2_w), -1)

    total_flow_rate_inlet = 0.0
    total_flow_rate_outlet = 0.0
    for triangle in connectivity_inlet:
        # 三角形の3つの頂点を取得
        p1 = torch.tensor([xb_inlet[triangle[0]], yb_inlet[triangle[0]], zb_inlet[triangle[0]]])
        p2 = torch.tensor([xb_inlet[triangle[1]], yb_inlet[triangle[1]], zb_inlet[triangle[1]]])
        p3 = torch.tensor([xb_inlet[triangle[2]], yb_inlet[triangle[2]], zb_inlet[triangle[2]]])
        # 三角形の2辺のベクトル
        v1 = p2 - p1
        v2 = p3 - p1
        # 法線ベクトルを外積で計算
        normal_vector = torch.cross(v1, v2)
        normal_vector_length = torch.norm(normal_vector)
        # 三角形の面積
        area = 0.5 * normal_vector_length
        # 法線ベクトルを正規化
        normal_unit_vector = normal_vector / normal_vector_length
        # 各頂点での速度ベクトル
        velocity_p1 = torch.tensor([out1_u[triangle[0]], out1_v[triangle[0]], out1_w[triangle[0]]])
        velocity_p2 = torch.tensor([out1_u[triangle[1]], out1_v[triangle[1]], out1_w[triangle[1]]])
        velocity_p3 = torch.tensor([out1_u[triangle[2]], out1_v[triangle[2]], out1_w[triangle[2]]])
        # 各頂点の速度の平均
        velocity_avg = (velocity_p1 + velocity_p2 + velocity_p3) / 3.0
        # 法線方向の速度成分を計算
        normal_velocity = torch.dot(velocity_avg, normal_unit_vector)
        #print(normal_velocity)
        #exit(1)
        # 流量 = 面積 * 法線方向の速度成分
        flow_rate = area * normal_velocity
        total_flow_rate_inlet += flow_rate

    for triangle in connectivity_outlet:
        # 三角形の3つの頂点を取得
        p1 = torch.tensor([xb_outlet[triangle[0]], yb_outlet[triangle[0]], zb_outlet[triangle[0]]])
        p2 = torch.tensor([xb_outlet[triangle[1]], yb_outlet[triangle[1]], zb_outlet[triangle[1]]])
        p3 = torch.tensor([xb_outlet[triangle[2]], yb_outlet[triangle[2]], zb_outlet[triangle[2]]])
        # 三角形の2辺のベクトル
        v1 = p2 - p1
        v2 = p3 - p1
        # 法線ベクトルを外積で計算
        normal_vector = torch.cross(v1, v2)
        normal_vector_length = torch.norm(normal_vector)
        # 三角形の面積
        area = 0.5 * normal_vector_length
        # 法線ベクトルを正規化
        normal_unit_vector = normal_vector / normal_vector_length
        # 各頂点での速度ベクトル
        velocity_p1 = torch.tensor([out2_u[triangle[0]], out2_v[triangle[0]], out2_w[triangle[0]]])
        velocity_p2 = torch.tensor([out2_u[triangle[1]], out2_v[triangle[1]], out2_w[triangle[1]]])
        velocity_p3 = torch.tensor([out2_u[triangle[2]], out2_v[triangle[2]], out2_w[triangle[2]]])
        # 各頂点の速度の平均
        velocity_avg = (velocity_p1 + velocity_p2 + velocity_p3) / 3.0
        # 法線方向の速度成分を計算
        normal_velocity = torch.dot(velocity_avg, normal_unit_vector)
        #exit(1)
        # 流量 = 面積 * 法線方向の速度成分
        flow_rate = area * normal_velocity
        total_flow_rate_outlet += flow_rate
    print("inlet : {}".format(total_flow_rate_inlet))
    print("outlet: {}".format(total_flow_rate_outlet))
    mse_loss = torch.mean((total_flow_rate_inlet + total_flow_rate_outlet) ** 2)
    return mse_loss * 1000