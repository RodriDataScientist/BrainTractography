# %%
import nibabel as nib
from get_seeds import get_seeds
from utils import visualize_streamlines

# 1. Cargar datos necesarios
# --------------------------------------------
# B: máscara binaria (FA > umbral)
# P: volumen de picos direccionales (4D)
# nii_path: opcional, archivo .nii con posiciones semilla

# Ruta de los archivos
B_path = "data/mask_FA_0p2.nii"
P_path = "data/all_peaks_norm_ISMRM_NNLS.nii"
#P_path = "data/one_peak.nii"
nii_path = "data/seed_cc.nii" 

B = nib.load(B_path).get_fdata()
P = nib.load(P_path).get_fdata()

# 2. Definir parámetros del modelo
# --------------------------------------------
params = {
    "B": B,
    "n": 1,                   # porcentaje de voxels usados como semillas
    "P": P,
    "num_peaks": 5,      
    "orientaciones": None,    # se calcularán automáticamente si es None
    "epsilon_1": 1e-1,
    "m": 7,                   # subvoxelizaciones por semilla
    "ratio": 0.3,
    "alpha": 1.0,
    "w": 1,
    "z": 1,
    "sigma": 1,
    "epsilon_2": 1e-10,
    "G": 0.1,
    "nii_path": nii_path
}

# 3. Crear objeto tractografía
# --------------------------------------------
seeds_obj = get_seeds(**params)

# 4. Ejecutar la tractografía
# --------------------------------------------
num_iter = 100  # número de pasos
gamma1 = 0.1
gamma2 = 0.9
gamma3 = 0

print("Ejecutando tractografía...")
streamlines = seeds_obj.run_tractography(num_iter, gamma1, gamma2, gamma3)
print(f"Se generaron {len(streamlines)} streamlines.")

# 5. Guardar resultados
# --------------------------------------------
from nibabel.streamlines import Tractogram, TckFile

# Cargar referencia para affine
ref_nii = nib.load(P_path)

# Crear tractograma
tractogram = Tractogram(streamlines, affine_to_rasmm=ref_nii.affine)

# Guardar en formato .tck
tck = TckFile(tractogram, header={})
tck_path = "output/streamlines_one_peak.tck"
tck.save(tck_path)

print(f"Streamlines guardadas en: {tck_path}")

# Preguntar para visualizar
input("Pulsa enter para visualizar resultados...")
visualize_streamlines(tck_path, B_path)
# %%