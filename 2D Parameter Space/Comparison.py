import numpy as np
import dedalus.public as d3
import dedalus.core as d4
import logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pathlib
import subprocess
import h5py
import glob
import re

QDs=[0, 28e-4, 14e-3]
Ds=[1e-1, 2e-1, 5e-1, 1.0, 2.0]
Names1=['0','28e-4','14e-3']
Names2=['1e-1','2e-1','5e-1','1','2']
Lx, Lz = 128, 1
Nx, Nz = 64*128, 64
Md = 3
Rayleigh = 2e6
Prandtl = 0.7
dealias = 3/2
stop_sim_time = 200
timestepper = d3.RK222
max_timestep = 0.125
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
d = dist.Field(name='d', bases=(xbasis,zbasis))
m = dist.Field(name='m', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_d1 = dist.Field(name='tau_d1', bases=xbasis)
tau_d2 = dist.Field(name='tau_d2', bases=xbasis)
tau_m1 = dist.Field(name='tau_m1', bases=xbasis)
tau_m2 = dist.Field(name='tau_m2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# Substitutions
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_d = d3.grad(d) + ez*lift(tau_d1) # First-order reduction
grad_m = d3.grad(m) + ez*lift(tau_m1) # First-order reduction
ncc = dist.Field(name='ncc', bases=zbasis)
ncc['g'] = z
ncc.change_scales(3/2)
u_x = u @ ex
u_z = u @ ez
dz = lambda A: d3.Differentiate(A, coords['z'])
integ = lambda A: d3.Integrate(A, coords)
integ1= lambda A: d3.Integrate(A, coords)



Vaisala= 3+1
nu = (Lz**3*Md)/(2*10**6 / Prandtl)**(1/2)
files = sorted(glob.glob(f'snapshots 128, 2e6/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
with h5py.File(files[0], mode='r') as file:
    moist_buoyancy1 = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy1.dims[1][0][:]
    zgrid=moist_buoyancy1.dims[2][0][:]
    moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
    buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*zgrid)
    saturation=moist_buoyancy-dry_buoyancy+Vaisala*zgrid
    extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*zgrid)
    extra_buoyancy=np.sum(extra_buoyancy,axis=2)
for k in range(1,len(files)):
    with h5py.File(files[k], mode='r') as file:
        moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
        dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
        buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*zgrid)
        extra_buoyancy1=buoyancy-(dry_buoyancy-Vaisala*zgrid)
        extra_buoyancy1=np.sum(extra_buoyancy1,axis=2)
        extra_buoyancy=np.append(extra_buoyancy,extra_buoyancy1,axis=0)
        
tgrid=np.arange(len(extra_buoyancy[:,0]))*nu/4
# Plotting extra buoyancy
plt.figure()
plt.contourf(xgrid, tgrid, extra_buoyancy[:,:])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title(f'Spectrum Comparison')
plt.savefig(f'Spectrum 128, 2e6 Comparison')
plt.close()
        
