Vaisala= 3+0.5
nu = (Lz**3*Md)/(2*10**6 / Prandtl)**(1/2)
files = sorted(glob.glob('snapshots/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
with h5py.File(files[-2], mode='r') as file:
    moist_buoyancy1 = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy1.dims[1][0][:]
    zgrid=moist_buoyancy1.dims[2][0][:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
for k in range(-2,0):
    with h5py.File(files[k], mode='r') as file:
        dry_buoyancy1 = file['tasks']['dry buoyancy'][:, :, :]
        dry_buoyancy=np.append(dry_buoyancy,dry_buoyancy1,axis=0)

Lt = dry_buoyancy.shape[0]

# Initializations
Dboxes = np.zeros((Lx, Nz))
D_grad_boxes = np.zeros(Lx)
D_thickness = np.zeros(Lx)
local_Rayleigh = np.zeros(Lx)

# Main Iteration
for l in range(Lx):
    Dboxes[l,: ] = np.sum(dry_buoyancy[:, l*Nz:(l+1)*Nz, :], axis=(0, 1)) / (Lt * Nz)

    # Gradient of Dboxes
    D_grad_boxes[l] = (Dboxes[l, 1] - Dboxes[l, 0]) / (zgrid[1] - zgrid[0])

    # Thickness
    D_thickness[l] = np.min(Dboxes[l, :]) * D_grad_boxes[l]

    if D_thickness[l] == 0:
        D_thickness[l] = 1

    # Local RA
    local_Rayleigh[l] = D_thickness[l] * D_grad_boxes[l] / (nu * kappa)

plt.hist(local_Rayleigh, bins=30, color='skyblue', edgecolor='black', 
         weights=np.ones_like(local_Rayleigh) / len(local_Rayleigh) * 100)
plt.title('Histogram of Local Rayleigh Numbers')
plt.xlabel('Local Rayleigh Number')
plt.ylabel('Frequency (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

