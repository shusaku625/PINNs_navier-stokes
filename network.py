import torch
import torch.nn as nn

input_n = 3
input_nt = 4
h_n = 128

class Swish(nn.Module):
	def __init__(self, inplace=True):
		super(Swish, self).__init__()
		self.inplace = inplace
	def forward(self, x):
		if self.inplace:
			x.mul_(torch.sigmoid(x))
			return x
		else:
			return x * torch.sigmoid(x)

class MySquared(nn.Module):
	def __init__(self, inplace=True):
		super(MySquared, self).__init__()
		self.inplace = inplace
	def forward(self, x):
		return torch.square(x)

class Net2_u(nn.Module):
	def __init__(self):
		super(Net2_u, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(input_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,1),
		)
	def forward(self,x):	
		output = self.main(x)
		return output

class Net2_v(nn.Module):
	def __init__(self):
		super(Net2_v, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(input_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,1),
		)
	def forward(self,x):	
		output = self.main(x)
		return output

class Net2_w(nn.Module):
	def __init__(self):
		super(Net2_w, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(input_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,1),
		)
	def forward(self,x):	
		output = self.main(x)
		return output

class Net2_p(nn.Module):
	def __init__(self):
		super(Net2_p, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(input_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,1),
		)
	def forward(self,x):
		output = self.main(x)
		return  output

class Net2_ut(nn.Module):
	def __init__(self):
		super(Net2_ut, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(input_nt,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,1),
		)
	def forward(self,x):	
		output = self.main(x)
		return output

class Net2_vt(nn.Module):
	def __init__(self):
		super(Net2_vt, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(input_nt,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,1),
		)
	def forward(self,x):	
		output = self.main(x)
		return output

class Net2_wt(nn.Module):
	def __init__(self):
		super(Net2_wt, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(input_nt,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,1),
		)
	def forward(self,x):	
		output = self.main(x)
		return output

class Net2_pt(nn.Module):
	def __init__(self):
		super(Net2_pt, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(input_nt,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,h_n),
			Swish(),
			nn.Linear(h_n,1),
		)
	def forward(self,x):
		output = self.main(x)
		return  output