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
import os
import re

QDs=[0, 28e-4, 14e-3]
Ds=[3e-1, 4e-1, 6e-1, 7e-1, 8e-1, 9e-1]
Names1=['0 ','28e-4','14e-3']
Names2=['3e-1', '4e-1', '6e-1', '7e-1', '8e-1', '9e-1']
Names3=['0','28e-4','14e-3']
Lx, Lz = 128, 1
Nx, Nz = 4096, 32
Vaisala= 3
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

i=2

"""
for j in range(len(Ds)):
    Md = 3*Ds[j]
    nu = (Lz**3*Md)/(2*10**6 / Prandtl)**(1/2)
    kappa = (Lz**3*Md)/(Rayleigh * Prandtl)**(1/2)
    files = sorted(glob.glob(f'snapshots {Names1[i]}, DH={Names2[j]}/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
    with h5py.File(files[-10], mode='r') as file:
        moist_buoyancy1 = file['tasks']['moist buoyancy']
        xgrid=moist_buoyancy1.dims[1][0][:]
        zgrid=moist_buoyancy1.dims[2][0][:]
        dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
    for k in range(-10,0):
        with h5py.File(files[k], mode='r') as file:
            dry_buoyancy1 = file['tasks']['dry buoyancy'][:, :, :]
            dry_buoyancy=np.append(dry_buoyancy,dry_buoyancy1,axis=0)
    
    Lt = dry_buoyancy.shape[0]  # Assuming dry_buoyancy is of shape (time, x, z)
    # Variable Initializations
    Dboxes = np.zeros((Lx, Nz))
    D_grad_boxes = np.zeros(Lx)
    D_thickness = np.zeros(Lx)
    D_thickness1 = np.zeros(Lx)
    D_thickness2 = np.zeros(Lx)
    local_Rayleigh = np.zeros(Lx)
    
    # Main Computation Loop
    for l in range(Lx):
        Dboxes[l,: ] = np.sum(dry_buoyancy[:, l*Nz:(l+1)*Nz, :], axis=(0, 1)) / (Lt * Nz)
    
        # Gradient of Dboxes along the z-axis
        D_grad_boxes[l] = (Dboxes[l, 1] - Dboxes[l, 0]) / (zgrid[1] - zgrid[0])
    
        # Thickness calculation
        D_thickness1[l] = np.min(Dboxes[l, :]) * D_grad_boxes[l]
        D_thickness2[l] = np.max(Dboxes[l, :]) * D_grad_boxes[l]
        
        if D_thickness1[l] == 0:
            D_thickness1[l] = 1
        
        D_thickness[l] = min(D_thickness1[l], D_thickness2[l])
        
        if D_thickness[l] > 1:
            D_thickness[l] = 1
        
        # Local Rayleigh number calculation
        local_Rayleigh[l] = D_thickness[l] * D_grad_boxes[l] / (nu * kappa)
    
    # Plotting
    plt.hist(local_Rayleigh, bins=128, color='skyblue', edgecolor='black', 
             weights=np.ones_like(local_Rayleigh) / len(local_Rayleigh) * 100)
    plt.title(f'Histogram of Local Rayleigh Numbers QD={Names3[i]}, DH={Names2[j]}')
    plt.xlabel('Local Rayleigh Number')
    plt.ylabel('Frequency (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'Local_RA 128, 2e6, QD={Names3[i]}, DH={Names2[j]}')
    plt.close()
    


for j in range(len(Ds)):
    Md = 3*Ds[j]
    nu = (Lz**3*Md)/(2*10**6 / Prandtl)**(1/2)
    files = sorted(glob.glob(f'snapshots {Names1[i]}, DH={Names2[j]}/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
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
            
    tgrid=np.arange(len(extra_buoyancy[:,0]))*nu/2
    # Plotting extra buoyancy
    plt.figure()
    plt.contourf(xgrid, tgrid, extra_buoyancy[:,:])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Spectrum QD={Names3[i]}, DH={Names2[j]}')
    plt.savefig(f'Spectrum 128, 2e6, QD={Names3[i]}, DH={Names2[j]}')
    plt.close()


for j in range(len(Ds)):
    Md = 3*Ds[j]
    nu = (Lz**3*Md)/(2*10**6 / Prandtl)**(1/2)
    files = sorted(glob.glob(f'snapshots {Names1[i]}, DH={Names2[j]}/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
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
            
    tgrid=np.arange(len(extra_buoyancy[:,0]))*nu*QDs[i]/(2*Ds[j])
    # Plotting extra buoyancy
    plt.figure()
    plt.contourf(xgrid, tgrid, extra_buoyancy[:,:])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Time Scale QD={Names3[i]}, DH={Names2[j]}')
    plt.savefig(f'Time Scale 128, 2e6, QD={Names3[i]}, DH={Names2[j]}')
    plt.close()

for j in range(len(Ds)):
    folder_name = f"Local Rayleigh QD={Names3[i]}, DH={Names2[j]}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    Md = 3*Ds[j]
    nu = (Lz**3*Md)/(2*10**6 / Prandtl)**(1/2)
    kappa = (Lz**3*Md)/(Rayleigh * Prandtl)**(1/2)
    files = sorted(glob.glob(f'snapshots {Names1[i]}, DH={Names2[j]}/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
    for k in range(len(files)):
        with h5py.File(files[k], mode='r') as file:
            moist_buoyancy1 = file['tasks']['moist buoyancy']
            xgrid=moist_buoyancy1.dims[1][0][:]
            zgrid=moist_buoyancy1.dims[2][0][:]
            dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
        
        Lt = dry_buoyancy.shape[0]  # Assuming dry_buoyancy is of shape (time, x, z)
        # Variable Initializations
        Dboxes = np.zeros((Lx, Nz))
        D_grad_boxes = np.zeros(Lx)
        D_thickness = np.zeros(Lx)
        D_thickness1 = np.zeros(Lx)
        D_thickness2 = np.zeros(Lx)
        local_Rayleigh = np.zeros(Lx)
        
        # Main Computation Loop
        for l in range(Lx):
            Dboxes[l,: ] = np.sum(dry_buoyancy[:, l*Nz:(l+1)*Nz, :], axis=(0, 1)) / (Lt * Nz)
        
            # Gradient of Dboxes along the z-axis
            D_grad_boxes[l] = (Dboxes[l, 1] - Dboxes[l, 0]) / (zgrid[1] - zgrid[0])
        
            # Thickness calculation
            D_thickness1[l] = np.min(Dboxes[l, :]) * D_grad_boxes[l]
            D_thickness2[l] = np.max(Dboxes[l, :]) * D_grad_boxes[l]
            
            if D_thickness1[l] == 0:
                D_thickness1[l] = 1
            
            D_thickness[l] = min(D_thickness1[l], D_thickness2[l])
            
            if D_thickness[l] > 1:
                D_thickness[l] = 1
            
            # Local Rayleigh number calculation
            local_Rayleigh[l] = D_thickness[l] * D_grad_boxes[l] / (nu * kappa)
        
        # Plotting
        plt.hist(local_Rayleigh, bins=128, color='skyblue', edgecolor='black', 
                 weights=np.ones_like(local_Rayleigh) / len(local_Rayleigh) * 100)
        plt.title(f'Histogram of Local Rayleigh Numbers QD={Names3[i]}, DH={Names2[j]}')
        plt.xlabel('Local Rayleigh Number')
        plt.ylabel('Frequency (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(folder_name, f'Local_RA, Index={k}'))
        plt.close()


for j in range(len(Ds)):
    Md = 3*Ds[j]
    nu = (Lz**3*Md)/(2*10**6 / Prandtl)**(1/2)
    files = sorted(glob.glob(f'snapshots {Names1[i]}, DH={Names2[j]}/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
    fig, ax = plt.subplots()

    with h5py.File(files[0], mode='r') as file:
        KEs = file['tasks']['total KE'][:,:,:]

    for k in range(1,len(files)):
        with h5py.File(files[k], mode='r') as file:
            KE = file['tasks']['total KE'][:,:,:]
            KEs=np.append(KEs,KE,axis=0)
   
    ax.plot(np.arange(len(KEs[:,0,0]))*nu/2, KEs[:,0,0])

    ax.set_title('Total KE vs Time')
    ax.grid(True)
    ax.set_xlabel(r"Normalized Time $\nu t/H^2$")
    ax.set_ylabel(r"Total KE  $\log_{10}E_k(t)$")
    ax.set_yscale('log')
    plt.savefig(f'Total_KE_vs_Time 128, 2e6, QD={Names3[i]}, DH={Names2[j]}')
    plt.close()
    
for j in range(len(Ds)):
    files = sorted(glob.glob(f'snapshots {Names1[i]}, DH={Names2[j]}/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
    with h5py.File(files[-1], mode='r') as file:
        moist_buoyancy1 = file['tasks']['moist buoyancy']
        xgrid=moist_buoyancy1.dims[1][0][:]
        zgrid=moist_buoyancy1.dims[2][0][:]
        moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
        dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
        buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*zgrid)
        saturation=moist_buoyancy-dry_buoyancy+Vaisala*zgrid
        extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*zgrid)


        # Plotting extra buoyancy
        plt.figure()
        plt.contourf(xgrid, zgrid, extra_buoyancy[-1,:,:].T)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(f'Final Clouds QD={Names3[i]}, DH={Names2[j]}')
        plt.savefig(f'Final Clouds 128, 2e6, QD={Names3[i]}, DH={Names2[j]}')
        plt.close()


for j in range(len(Ds)):
    Md = 3*Ds[j]
    nu = (Lz**3*Md)/(2*10**6 / Prandtl)**(1/2)
    files = sorted(glob.glob(f'snapshots {Names1[i]}, DH={Names2[j]}/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
    fig, ax = plt.subplots()

    with h5py.File(files[0], mode='r') as file:
        moist_buoyancy1 = file['tasks']['moist buoyancy']
        xgrid=moist_buoyancy1.dims[1][0][:]
        zgrid=moist_buoyancy1.dims[2][0][:]
        moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
        dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
        water = 2*moist_buoyancy-dry_buoyancy
        total_water=water.sum(axis=(1,2))
        variances = np.var(water, axis=1)

    for k in range(1,len(files)):
        with h5py.File(files[k], mode='r') as file:
            moist_buoyancy1 = file['tasks']['moist buoyancy']
            xgrid=moist_buoyancy1.dims[1][0][:]
            zgrid=moist_buoyancy1.dims[2][0][:]
            moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
            dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
            water = 2*moist_buoyancy-dry_buoyancy
            total_water1=water.sum(axis=(1,2))
            total_water=np.append(total_water,total_water1,axis=0)
            variance_water = np.var(water, axis=1)
            variances=np.append(variances,variance_water,axis=0)
    
    variances=np.mean(variances[:,:],axis=1)
    fig1, ax1 = plt.subplots()  
    ax1.plot(np.arange(len(variances))*nu/2, variances[:])

    ax1.set_title('Horizontal Water Variance')
    ax1.grid(True)
    ax1.set_xlabel(r"Normalized Time $\nu t/H^2$")
    ax1.set_ylabel(r"$\chi M-D$")
    plt.savefig(f'Hotizontal Water Variance 128, 2e6, QD={Names3[i]}, DH={Names2[j]}')
    plt.close()
    
    fig2, ax2 = plt.subplots()
    ax2.plot(np.arange(len(total_water))*nu/2, total_water[:])

    ax2.set_title('Total Water')
    ax2.grid(True)
    ax2.set_xlabel(r"Normalized Time $\nu t/H^2$")
    ax2.set_ylabel(r"$\langle (\chi M-D) \rangle$")
    plt.savefig(f'Total Water 128, 2e6, QD={Names3[i]}, DH={Names2[j]}')
    plt.close()

 

for j in range(len(Ds)):
    Lx, Lz = 128, 1
    Nx, Nz = 32*128, 32
    Md = 3*Ds[j]
    nu = (Lz**3*Md)/(2*10**6 / Prandtl)**(1/2)
    files = sorted(glob.glob(f'snapshots {Names1[i]}, DH={Names2[j]}/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
    
    cloud_covers=[]
    condensed_waters=[]
    with h5py.File(files[0], mode='r') as file:
        moist_buoyancy1 = file['tasks']['moist buoyancy']
        xgrid=moist_buoyancy1.dims[1][0][:]
        zgrid=moist_buoyancy1.dims[2][0][:]
        #moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
        #dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
        #buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*zgrid)
        #extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*zgrid)
    
    for k in range(len(files)):
        with h5py.File(files[k], mode='r') as file:
            moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
            dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
            buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*zgrid)
            extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*zgrid)
            for l in range(len(extra_buoyancy[:,0,0])):
                cloud_cover=len(np.where(np.sum(extra_buoyancy[l,:,:],axis=1)>0)[0])/Nx
                condensed_water=np.sum(extra_buoyancy[l,:,:])
                cloud_covers.append(cloud_cover)
                condensed_waters.append(condensed_water)
                
    fig1, ax1 = plt.subplots()  
    ax1.plot(np.arange(len(cloud_covers))/2*nu, cloud_covers)
    ax1.set_title('Cloud Cover vs Time')
    ax1.grid(True)
    ax1.set_xlabel(r"Normalized Time $\nu t/H^2$")
    ax1.set_ylabel(r"Cloud Cover")
    plt.savefig(f'Cloud_Cover_vs_Time 128, 2e6, QD={Names3[i]}, DH={Names2[j]}')
    plt.close()
    
    fig2, ax2 = plt.subplots()
    ax2.plot(np.arange(len(condensed_waters))/2*nu, condensed_waters)
    ax2.set_title('Condensed Water vs Time')
    ax2.grid(True)
    ax2.set_xlabel(r"Normalized Time $\nu t/H^2$")
    ax2.set_ylabel(r"Condensed Water")
    plt.savefig(f'Condensed_Water_vs_Time 128, 2e6, QD={Names3[i]}, DH={Names2[j]}')
    plt.close()
"""

for j in range(len(Ds)):
    folder_name = f"Vertical Profiles QD={Names3[i]}, DH={Names2[j]}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    Md = 3*Ds[j]
    nu = (Lz**3*Md)/(2*10**6 / Prandtl)**(1/2)
    files = sorted(glob.glob(f'snapshots {Names1[i]}, DH={Names2[j]}/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
    for k in range(len(files)):
        with h5py.File(files[k], mode='r') as file:
            moist_buoyancy = file['tasks']['moist buoyancy']
            xgrid=moist_buoyancy.dims[1][0][:]
            zgrid=moist_buoyancy.dims[2][0][:]
            dry_buoyancy = file['tasks']['dry buoyancy'][:,:,:]
            moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
            buoyancy = file['tasks']['buoyancy'][:,:,:]
            avgmbs=np.mean(moist_buoyancy,axis=1)
            avgdbs=np.mean(dry_buoyancy,axis=1)
            avgbs=np.mean(buoyancy,axis=1)
            avgmbs1=np.mean(avgmbs,axis=0)
            avgdbs1=np.mean(avgdbs,axis=0)
            avgbs1=np.mean(avgbs,axis=0)
            
        fig, ax = plt.subplots(3, figsize=(6,12))
                
        ax[0].plot(zgrid, avgmbs1[:],linestyle='dashdot',label='Q=0.07')
        ax[1].plot(zgrid, avgdbs1[:],linestyle='dashdot',label='Q=0.07')
        ax[2].plot(zgrid, avgbs1[:],linestyle='dashdot',label='Q=0.07')
        
        ax[0].set_title('Vertical M Profile')
        ax[0].grid(True)
        ax[0].set_xlabel(r"$z$")
        ax[0].set_ylabel(r"$M$")
        ax[0].legend()
        
        ax[1].set_title('Vertical D Profile')
        ax[1].grid(True)
        ax[1].set_xlabel(r"$z$")
        ax[1].set_ylabel(r"$D$")
        ax[1].legend()
        
        ax[2].set_title('Vertical B Profile')
        ax[2].grid(True)
        ax[2].set_xlabel(r"$z$")
        ax[2].set_ylabel(r"$B$")
        ax[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, f'Vertical Profile Time Average, Index={k}.png'))
        plt.close()
   
   
for j in range(len(Ds)): 
    Md = 3*Ds[j]
    nu = (Lz**3*Md)/(2*10**6 / Prandtl)**(1/2)
    kappa = (Lz**3*Md)/(Rayleigh * Prandtl)**(1/2)
    files = sorted(glob.glob(f'snapshots {Names1[i]}, DH={Names2[j]}/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
    with h5py.File(files[0], mode='r') as file:
            moist_buoyancy1 = file['tasks']['moist buoyancy']
            xgrid=moist_buoyancy1.dims[1][0][:]
            zgrid=moist_buoyancy1.dims[2][0][:]
            dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
            moist_buoyancy = file['tasks']['moist buoyancy'][:, :, :]
            Lt = dry_buoyancy.shape[0]
            Dboxes=np.mean(dry_buoyancy, axis=1)
            Mboxes=np.mean(dry_buoyancy, axis=1)
            D_grad_boxes = (Dboxes[:, 1] - Dboxes[:, 0]) / (zgrid[1] - zgrid[0])
            M_grad_boxes = (Mboxes[:, 1] - Mboxes[:, 0]) / (zgrid[1] - zgrid[0])
            Nusselt=1/(7*Ds[j])*D_grad_boxes-2/(7*Ds[j])*M_grad_boxes
    for k in range(1,len(files)):
        with h5py.File(files[k], mode='r') as file:
            moist_buoyancy1 = file['tasks']['moist buoyancy']
            xgrid=moist_buoyancy1.dims[1][0][:]
            zgrid=moist_buoyancy1.dims[2][0][:]
            dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
            moist_buoyancy = file['tasks']['moist buoyancy'][:, :, :]
            Lt = dry_buoyancy.shape[0]
            Dboxes=np.mean(dry_buoyancy, axis=1)
            Mboxes=np.mean(dry_buoyancy, axis=1)
            D_grad_boxes = (Dboxes[:, 1] - Dboxes[:, 0]) / (zgrid[1] - zgrid[0])
            M_grad_boxes = (Mboxes[:, 1] - Mboxes[:, 0]) / (zgrid[1] - zgrid[0])
            Nusselt1=1/(7*Ds[j])*D_grad_boxes-2/(7*Ds[j])*M_grad_boxes
            Nusselt=np.append(Nusselt, Nusselt1, axis=0)
            
    ax.plot(np.arange(len(Nusselt))*nu/2, Nusselt[:])

    ax.set_title('Nusselt vs Time')
    ax.grid(True)
    ax.set_xlabel(r"Normalized Time $\nu t/H^2$")
    ax.set_ylabel(r"Nusselt Number")
    plt.savefig(f'Nusselt_vs_Time 128, 2e6, QD={Names3[i]}, DH={Names2[j]}')
    plt.close()
"""
print("start Animating")

for qd in Names1:
    for dh in Names2:
        Vaisala= 3+Ds[j]
        files = sorted(glob.glob(f'snapshots {Names1[i]}, DH={Names2[j]}/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
        with h5py.File(files[0], mode='r') as file:
            extra_buoyancy = file['tasks']['additional buoyancy'][:, :, :]
            xgrid=file['tasks']['additional buoyancy'].dims[1][0][:]
            zgrid=file['tasks']['additional buoyancy'].dims[2][0][:]
            clouds = np.where(extra_buoyancy < 0, 0, extra_buoyancy)
        for k in range(1,len(files)):
            with h5py.File(files[k], mode='r') as file:
                extra_buoyancy = file['tasks']['additional buoyancy'][:, :, :] 
                clouds=np.append(clouds,extra_buoyancy,axis=0)
    
        clouds = np.where(clouds < 0, 0, clouds)     
        global_min=np.min(clouds) 
        global_max=np.max(clouds)
        condition = (clouds == global_max)
        max_pos = np.where(condition)
        print("finished processing")
        fig, ax = plt.subplots()
        cax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # Define colorbar axes position
        img = ax.contourf(xgrid, zgrid, clouds[int(max_pos[0]), :, :].T, vmin=global_min, vmax=global_max,cmap='Blues_r', levels=15)
        cb = fig.colorbar(img, cax=cax)
        cb.set_label('Extra Buoyancy')
        def animate(frame):
            ax.clear()
            img = ax.contourf(xgrid, zgrid, clouds[frame, :, : ].T,vmin=global_min, vmax=global_max,cmap='Blues_r')
            ax.set_title('Extra_Buoyancy, Frame: {}'.format(frame))
            ax.set_xlabel('x')  # Add x-axis label
            ax.set_ylabel('z')  # Add y-axis label
    
        # Call animate method
        animation = FuncAnimation(fig, animate, frames=len(clouds), interval=100, blit=False)
        animation.save(f'clouds 128, 2e6, QD={Names1[i]}, DH={Names2[j]}.mp4', writer='ffmpeg', dpi=200)
        # Display the plot
        plt.show()
        print(f'clouds 128, 2e6, QD={Names1[i]}, DH={Names2[j]} finished')
"""
