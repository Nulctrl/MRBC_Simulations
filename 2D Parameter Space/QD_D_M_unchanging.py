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
Nx, Nz = 4096, 32
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

"""
for i in range(len(QDs)):
    for j in range(len(Ds)):
        Vaisala= 3+Ds[j]
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
        plt.title(f'Spectrum QD={Names1[i]}, DH={Names2[j]}')
        plt.savefig(f'Spectrum 128, 2e6, QD={Names1[i]}, DH={Names2[j]}')
        plt.close()
        

for i in range(len(QDs)):
    for j in range(len(Ds)):
        Vaisala= 3+Ds[j]
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
        plt.savefig(f'Total_KE_vs_Time 128, 2e6, QD={Names1[i]}, DH={Names2[j]}')
        plt.close()
        
for i in range(len(QDs)):
    for j in range(len(Ds)):
        Vaisala= 3+Ds[j]
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
            plt.title(f'Final Clouds QD={Names1[i]}, DH={Names2[j]}')
            plt.savefig(f'Final Clouds 128, 2e6, QD={Names1[i]}, DH={Names2[j]}')
            plt.close()

for i in range(len(QDs)):
    for j in range(len(Ds)):
        Vaisala= 3+Ds[j]
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
            variances = np.var(water, axis=1)
    
        for k in range(1,len(files)):
            with h5py.File(files[k], mode='r') as file:
                moist_buoyancy1 = file['tasks']['moist buoyancy']
                xgrid=moist_buoyancy1.dims[1][0][:]
                zgrid=moist_buoyancy1.dims[2][0][:]
                moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
                dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
                water = 2*moist_buoyancy-dry_buoyancy
                variance_water = np.var(water, axis=1)
                variances=np.append(variances,variance_water,axis=0)
        
        variances=np.mean(variances[:,:],axis=1)
       
        ax.plot(np.arange(len(variances))*nu, variances[:])
    
        ax.set_title('Horizontal Water Variance')
        ax.grid(True)
        ax.set_xlabel(r"Normalized Time $\nu t/H^2$")
        ax.set_ylabel(r"$\langle Var(\chi M-D) \rangle$")
        plt.savefig(f'Hotizontal Water Variance 128, 2e6, QD={Names1[i]}, DH={Names2[j]}')
        plt.close()

"""     
for i in range(len(QDs)):
    for j in range(len(Ds)):
        Vaisala= 3+Ds[j]
        Lx, Lz = 128, 1
        Nx, Nz = 32*128, 32
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
        ax1.plot(np.arange(len(cloud_covers))/4*nu, cloud_covers)
        ax1.set_title('Cloud Cover vs Time')
        ax1.grid(True)
        ax1.set_xlabel(r"Normalized Time $\nu t/H^2$")
        ax1.set_ylabel(r"Cloud Cover")
        plt.savefig(f'Cloud_Cover_vs_Time 128, 2e6, QD={Names1[i]}, DH={Names2[j]}, DH={dh}')
        plt.close()
        
        fig2, ax2 = plt.subplots()
        ax2.plot(np.arange(len(condensed_waters))/4*nu, condensed_waters)
        ax2.set_title('Condensed Water vs Time')
        ax2.grid(True)
        ax2.set_xlabel(r"Normalized Time $\nu t/H^2$")
        ax2.set_ylabel(r"Condensed Water")
        plt.savefig(f'Condensed_Water_vs_Time 128, 2e6, QD={Names1[i]}, DH={Names2[j]}, DH={dh}')
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
