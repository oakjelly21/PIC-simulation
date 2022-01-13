import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import sys
import os



#x = 0
def handle_close(evt):
	global flag
	flag = True
def getAcc( pos, Nx, boxsize, n0, Gmtx, Lmtx ):

	# Calculate Electron Number Density on the Mesh by
	# placing particles into the 2 nearest bins (j & j+1, with proper weights)
	# and normalizing
	N          = pos.shape[0]
	dx         = boxsize / Nx
	j          = np.floor(pos/dx).astype(int)
	jp1        = j+1
	weight_j   = ( jp1*dx - pos  )/dx
	weight_jp1 = ( pos    - j*dx )/dx
	jp1        = np.mod(jp1, Nx)   # periodic BC
	#print(j)
	#print(j+1)

	#print(weight_j.shape)
	#print(Nx)
	n  = np.bincount(j[:,0],   weights=weight_j[:,0],   minlength=Nx);
	n += np.bincount(jp1[:,0], weights=weight_jp1[:,0], minlength=Nx);
	n *= n0 * boxsize / N / dx
	#print((np.transpose(n)).shape)
	#print(np.matrix(n).shape)
	#print(n0)
	# Solve Poisson's Equation: laplacian(phi) = n-n0
	#phi_grid = spsolve(Lmtx, n-n0)
	#print(Lmtx.todense().shape)
	#print(np.matrix(n-n0).shape)
	#print((Lmtx.todense())@(np.matrix(n-n0).transpose()))
	#print(((np.matrix(n-n0))@(Lmtx.todense())).shape)
	#print((np.linalg.pinv(Lmtx.todense())).shape)
	#print((sp.linalg.inv(Lmtx)))
	phi_grid = spsolve(Lmtx,(n-n0))#@(np.matrix(n-n0).transpose())
	#print(phi_grid)
	#print(phi_grid.shape)
	#print(Lmtx.shape)
	#print(Gmtx.shape)

	# Apply Derivative to get the Electric field
	E_grid = - Gmtx @ phi_grid
	global plotE
	plotE = E_grid
	#print(plotE.shape)
	#print((weight_j@E_grid[j].transpose()).shape)
	# Interpolate grid value onto particle locations
	#print(E_grid[j].shape)
	#print((E_grid*weight_j.transpose()).shape)
	#E =  np.diagonal(weight_j @ np.squeeze(E_grid[j])) +np.diagonal(weight_jp1 @ np.squeeze(E_grid[jp1]))
	E =  weight_j*E_grid[j] +weight_jp1*E_grid[jp1]
	#print(E.shape)
	#print(weight_j.shape)
	#print(np.diagonal(weight_j*np.squeeze(E_grid[j])).shape)
	'''
	e = []
	for i in range(0,Nx):
		if(i<Nx-1):
			e.append((weight_j[i]*E_grid[i])+(weight_jp1[i]*E_grid[i+1]))
		else:
			e.append(weight_j[i]*E_grid[i])
	#print(np.array(e).shape)
	#print(weight_j[0])
	#print(np.squeeze(weight_j[0]))
	#print(np.array(e))
	a = -1*np.matrix(np.squeeze(e)).transpose()
	print(a)
	print(a.shape)
	'''
	a = -1*E
	return a


""" Plasma PIC simulation """

# Simulation parameters
N         = 40000   # Number of particles
Nx        = 600     # Number of mesh cells
t         = 0       # current time of the simulation
tEnd      = 100      # time at which simulation ends
dt        = 1       # timestep
boxsize   = 75      # periodic domain [0,boxsize]
n0        = 1       # electron number density
vb        = 5       # beam velocity
vth       = 1.5       # beam width
A         = 0.1     # perturbation
plotRealTime = True # switch on for plotting as the simulation goes along

# Generate Initial Conditions
np.random.seed(42)            # set the random number generator seed
# construct 2 opposite-moving Guassian beams
pos  = np.random.rand(N,1) * boxsize
vel  = vth * np.random.randn(N,1) + vb
Nh = int(N/2)
vel[Nh:] *= -1
# add perturbation
vel *= (1 + A*np.sin(2*np.pi*pos/boxsize))

# Construct matrix G to computer Gradient  (1st derivative)
dx = boxsize/Nx
e = np.ones(Nx)
diags = np.array([-1,1])
vals  = np.vstack((-e,e))
Gmtx = sp.spdiags(vals, diags, Nx, Nx);
Gmtx = sp.lil_matrix(Gmtx)
Gmtx[0,Nx-1] = -1
Gmtx[Nx-1,0] = 1
Gmtx /= (2*dx)
Gmtx = sp.csr_matrix(Gmtx)

# Construct matrix L to computer Laplacian (2nd derivative)
diags = np.array([-1,0,1])
vals  = np.vstack((e,-2*e,e))
Lmtx = sp.spdiags(vals, diags, Nx, Nx);
Lmtx = sp.lil_matrix(Lmtx)
Lmtx[0,Nx-1] = 1
Lmtx[Nx-1,0] = 1
Lmtx /= dx**2
Lmtx = sp.csr_matrix(Lmtx)

# calculate initial gravitational accelerations
acc = getAcc( pos, Nx, boxsize, n0, Gmtx, Lmtx )

# number of timesteps
#print("hey1")
Nt = int(np.ceil(tEnd/dt))

# prep figure
fig, axs = plt.subplots(1)
plt.show(block=False)
#plt.tight_layout()
plt.subplots_adjust(left=0.1, bottom=0.093, right=0.971, top=0.926, wspace=0.205, hspace=0.42)
flag = False
fig.canvas.mpl_connect('close_event', handle_close)
x = np.arange(0,boxsize,dx)
print(x.shape)
plotE = x.shape[0]
#print(plotE.shape)
#print("hey2")
# Simulation Main Loop
plt.pause(5)
for i in range(Nt):
	# (1/2) kick
	if flag:
		break;
	vel += acc * dt/2.0

	# drift (and apply periodic boundary conditions)
	pos += vel * dt
	pos = np.mod(pos, boxsize)

	# update accelerations
	acc = getAcc( pos, Nx, boxsize, n0, Gmtx, Lmtx )

	# (1/2) kick
	vel += (acc * dt/2.0)

	# update time
	t += dt

	# plot in real time - color 1/2 particles blue, other half red
	if plotRealTime or (i == Nt-1):
		plt.cla()
		#axs[1].cla()
		plt.scatter(pos[0:Nh],vel[0:Nh],s=.4,color='blue', alpha=0.5)
		plt.scatter(pos[Nh:], vel[Nh:], s=.4,color='red',  alpha=0.5)

		plt.axis([-1*(boxsize*.2),boxsize+(boxsize*0.2),-20,20])
		#axs[1].bar(t,0.5*(np.linalg.norm(vel)**2),color='b',width = 0.8)
		#axs[1].bar(x,plotE,color ='r' )
		#axs[0].set(xlabel='X(m)', ylabel='V(m/s)')
		plt.title('Velocity vs Position')
		plt.xlabel('x(m)')
		plt.ylabel('v(m/s)')
		#axs[0].set(xlabel='x-label', ylabel='y-label')
		#axs[1].set(xlabel='X(m)', ylabel='Electric Field(N/C)')
		#axs[1].set_title('Electric Field vs Position')
		#axs[1].set(xlabel='T(s)', ylabel='K.E(J)')
		#axs[1].set_title('Kinetic Energy vs Time')
		#axs[1].set(xlabel='x-label', ylabel='y-label')

		plt.pause(0.001)

		#print(x)
	#if(x==-1):
	#	break;

# Save figure
#plt.xlabel('x')
#plt.ylabel('v')
#plt.savefig('pic.png',dpi=240)
plt.show()
