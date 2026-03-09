import random
import numpy as np
import nibabel as nib
from particle import Particle
from family import Family

from utils import compute_cdf

class get_seeds:
    def __init__(self, B, n, P, num_peaks, orientaciones,
                 epsilon_1, m, ratio, alpha, w, z, sigma, epsilon_2,
                 G, nii_path=None):

        # 1.- Guardar parámetros:
        self.B = B
        self.n = n
        self.P = P
        self.num_peaks = num_peaks
        self.orientaciones = (orientaciones if orientaciones is not None
                              else np.arange(start=0, stop=3*(num_peaks-1)+1, step=3, dtype=int))
        self.epsilon_1 = epsilon_1
        self.m = m
        self.ratio = ratio
        self.alpha = alpha
        self.w = w
        self.z = z
        self.sigma = sigma
        self.epsilon_2 = epsilon_2
        self.G = G
        self.nii_path = nii_path

        # 2.- Generar semillas enteras + subvoxels
        self.Seeds, self.n1, self.n2, self.n3, self.N = self.seeds(self.nii_path)
        self.Seeds_update, self.N_update = self.update_seeds_voxel_position()
        self.SeedsReal = self.sub_voxel_position() 
        self.BAG_SeedsReal, self.N_Real = self.bag_seeds_real()

        # 3.- Calcular d0 en el primer voxel semilla
        if self.Seeds.shape[0] > 0:
            seed_voxel = self.Seeds[0].astype(int)
            directions_entero, norms_entero, count_entero = self.peaks_initial_seeds_voxel_position(seed_voxel.reshape(1,3))
            if count_entero > 0:
                # Seleccionamos el índice del pico con norma máxima
                idx_max = np.argmax(norms_entero)
                d0_max = directions_entero[idx_max].copy()
            else:
                # Si no hay picos, se toma el vector nulo
                d0_max = np.zeros(3, dtype=float)
        else:
            d0_max = np.zeros(3, dtype=float)

        # 4.- Convertir cada bloque de coordenadas en partículas y familias:
        self.familias = []
        for bloque in self.BAG_SeedsReal:
            particulas = [Particle(pos, active=True) for pos in bloque]
            fam = Family(particulas, G=self.G)
            # Asignar la máscara B y sus dimensiones a cada familia
            fam.B = self.B
            fam.n1 = self.n1
            fam.n2 = self.n2
            fam.n3 = self.n3
            self.familias.append(fam)

        # 5.- Asignar d0_max a cada partícula recién creada
        for fam in self.familias:
            for part in fam.particles:
                part.direction = d0_max.copy()

        # 6.- Calcular picos iniciales (d1) para cada familia, si hace falta:
        self.BAG_initial_peak_right, self.BAG_initial_peak_left = self.peaks_initial_seeds_real()

    def seeds(self, nii_path=None):
        """
        Obtiene semillas de dos maneras:
        1. Si `nii_path` es None: Genera semillas aleatorias desde B.
        2. Si `nii_path` tiene una ruta válida: Lee semillas desde el archivo .nii.

        Retorna:
        --------
        - Seeds: Matriz de semillas seleccionadas (N x 3).
        - n1, n2, n3: Dimensiones del volumen.
        - N: Número total de semillas.
        """
        
        # Obtener dimensiones de B
        n1 = self.B.shape[0]
        n2 = self.B.shape[1]
        n3 = self.B.shape[2]

        # Si se especifica un archivo .nii, leer semillas desde ahí
        if nii_path is not None:
            print(f"Cargando semillas desde {nii_path}...")
            
            # Cargar archivo .nii
            nii_data = nib.load(nii_path)
            Seeds_nii = nii_data.get_fdata()
            
            # Obtener coordenadas donde el valor es 1
            indices = np.argwhere(Seeds_nii == 1)
            
            if indices.shape[0] == 0:
                raise ValueError(f"No se encontraron semillas válidas en {nii_path}")

            Seeds = indices
            N = Seeds.shape[0]
            print(f"{N} semillas cargadas desde {nii_path}.")
        
        else:
            # Si no se pasa un .nii, generar semillas aleatorias desde B
            print("Generando semillas aleatorias desde B...")

            # Recorrer B para encontrar coordenadas donde B == 1
            FA_array = np.argwhere(self.B == 1)

            # Seleccionar un porcentaje de semillas aleatorias
            Indices = np.random.choice(
                a=np.arange(start=0, stop=FA_array.shape[0], step=1, dtype=int),
                size=round(FA_array.shape[0] * self.n),
                replace=False,
            )
            Seeds = FA_array[Indices]
            N = Seeds.shape[0]
            print(f"{N} semillas generadas aleatoriamente.")

        return Seeds, n1, n2, n3, N
    
    def update_seeds_voxel_position(self):
        """
        Devuelve una bolsa de Seeds actualizada 
        (aquellas con al menos un peak sin ruido)
        """
        Seeds_update            = []  # Bolsa de seeds con al menos un peak
        for i in range(0, self.N):
            SeedTest = self.Seeds[i]
            SeedTest = SeedTest.reshape((1,3))
            counter_peaks        = self.peaks_initial_seeds_voxel_position(SeedTest)[2]  # Contador de peaks (no ruido)
            if (counter_peaks > 0):
                Seeds_update.append(SeedTest)                   # Bolsa de Seeds con al menos un peak
        
        if len(Seeds_update) > 0:
            Seeds_update = np.array(Seeds_update)  # Convierte la lista a un array
            Seeds_update = np.squeeze(Seeds_update, axis=1)  # Elimina dimensiones extra si es necesario
            N_update = Seeds_update.shape[0]
        else:
            Seeds_update = np.empty((0, 3))  # Array vacío si no hay seeds válidas
            N_update = 0

        return Seeds_update, N_update
    
    def sub_voxel_position(self):
        """
        Devuelve una bolsa de m coordenadas reales por cada coordenada 
        real + la cordenada entera asociada
        """
        np.random.seed(random.randint(a = 0, b = 100))   # Experimento reproducible
        SeedsReal = []
        for i in range (0, self.N_update):
            seeds_real = []
            SeedAux = self.Seeds_update[i]
            SeedAux = SeedAux.reshape((1,3))
            seeds_real.append(SeedAux)
            x = SeedAux[0,0]
            y = SeedAux[0,1]
            z = SeedAux[0,2]
            RealCounter = 0

            while (RealCounter < self.m):
                SubVoxel = []
                x_real = np.random.uniform(low = x - self.ratio, high = x + self.ratio, size = 1)
                y_real = np.random.uniform(low = y - self.ratio, high = y + self.ratio, size = 1)
                z_real = np.random.uniform(low = z - self.ratio, high = z + self.ratio, size = 1)
                
                if ( (x_real - x)**2 + (y_real - y)**2 + (z_real - z)**2 <= self.ratio**2 ):
                    SubVoxel.append([x_real, y_real, z_real])
                    SubVoxel = np.asarray(SubVoxel).reshape((1,3))
                    seeds_real.append(SubVoxel)
                    RealCounter = RealCounter + 1
            SeedsReal.append(seeds_real)
            
        return SeedsReal
    
    def peaks_initial_seeds_voxel_position(self, SeedAux):
        """
        Devuelve una bolsa de peaks (sin ruido), una bolsa de sus contribuciones y 
        un contador de peaks de una coordenada entera
        """
        SeedAux = SeedAux.reshape((1,3))
        bag_direction_peaks   = []                                    # Guardar peak          (final)
        bag_norm_peaks        = []                                    # Guardar norma de peak (final)

        for o in self.orientaciones:                                  # Recorrer todos los peaks
            peak_voxel = self.P[int(SeedAux[0][0]), int(SeedAux[0][1]), int(SeedAux[0][2]), o:(o+3)]
            norm_peak  = np.linalg.norm(peak_voxel)                   # Obtener la norma del peak
                            
            if (norm_peak > self.epsilon_1):                          # CONDICION: El pico no sea ruido
                peak_voxel_unit = (peak_voxel)/(norm_peak)            # Convertir a unitario el peak   
                bag_direction_peaks.append(peak_voxel_unit)           # Agregar el peak unit
                bag_norm_peaks.append(norm_peak)                      # Agregar su norma
                
        counter_peaks = len(bag_norm_peaks)
            
        return bag_direction_peaks, bag_norm_peaks, counter_peaks
    
    def bag_seeds_real(self):
        """ Recibe SeedsReal y devuelve un arreglo sin dimensiones extra """
        BAG_SeedsReal   = []
        array_SeedsReal = np.array(self.SeedsReal)
    
        for seed in array_SeedsReal:
            modified_seed = np.squeeze(np.array(seed), axis=1)
            BAG_SeedsReal.append(modified_seed)
        
        N_real    = len(BAG_SeedsReal)    # Numero de bloques (familias)
        return BAG_SeedsReal, N_real
    
    def peaks_initial_seeds_real(self):
        """
        Devuelve dos bolsas de los peaks iniciales (right y left) 
        seleccionados para cada coordenada entera y real
        """
        BAG_initial_peak_right = [] # Bolsa de picos iniciales seleccionados por cada seed (final)
        BAG_initial_peak_left  = []
        
        for i in range(0, self.N_Real):            # Recorre todos los bloques
            SeedAux = self.BAG_SeedsReal[i][0]     # Elemeto 0 del bloque i (lider)
            SeedAux = SeedAux.reshape((1,3))
            bag_initial_peak_right  = []      # Bolsa de picos iniciales (por bloque)
            bag_initial_peak_left   = []
            # Bolsa de peaks (no ruido), bolsa de normas, contador de peaks
            bag_direction_peaks, bag_norm_peaks, counter_peaks = self.peaks_initial_seeds_voxel_position(SeedAux)
            max_direction = bag_direction_peaks[np.argmax(bag_norm_peaks)]   # Pico seleccionado
            for j in range(0, self.m + 1):
                bag_initial_peak_right.append(max_direction)                      
                bag_initial_peak_left.append(-1*max_direction)
                
            BAG_initial_peak_right.append(bag_initial_peak_right)
            BAG_initial_peak_left.append(bag_initial_peak_left)
                
        return BAG_initial_peak_right, BAG_initial_peak_left
    
    def quadrant(self, SeedAux):
        """
        Funcion que recibe una coordenada y devuelve una matriz 'vecinos' (8 x 3) 
        de los vertices en un cubo de la coordenada ingresada.
        """
        SeedAux = SeedAux.reshape((1,3))
        combinaciones = [ 
                    [np.ceil(SeedAux[0][0]),    np.ceil(SeedAux[0][1]),  np.ceil(SeedAux[0][2])],
                    [np.floor(SeedAux[0][0]),   np.floor(SeedAux[0][1]), np.floor(SeedAux[0][2])],
                    [np.ceil(SeedAux[0][0]),    np.ceil(SeedAux[0][1]),  np.floor(SeedAux[0][2])],
                    [np.ceil(SeedAux[0][0]),    np.floor(SeedAux[0][1]), np.floor(SeedAux[0][2])],
                    [np.ceil(SeedAux[0][0]),    np.floor(SeedAux[0][1]), np.ceil(SeedAux[0][2])],
                    [np.floor(SeedAux[0][0]),   np.floor(SeedAux[0][1]), np.ceil(SeedAux[0][2])],
                    [np.floor(SeedAux[0][0]),   np.ceil(SeedAux[0][1]),  np.ceil(SeedAux[0][2])],
                    [np.floor(SeedAux[0][0]),   np.ceil(SeedAux[0][1]),  np.floor(SeedAux[0][2])] 
                    ]
        fellows = np.asarray(combinaciones, dtype = int )
        return fellows
    
    def condition(self, fellows):
        """
        Devuelve los vecinos aptos
        """
        counter_fellows  = 0           # Contador de vecinos libres ('1')
        good_fellows     = []          # Almacenar coordenadas de vecinos libres 
        
        for k in range (0, fellows.shape[0]):                            # Recorrer vertices del cubo
            if (0 < fellows[k][0] < self.n1 and 0 < fellows[k][1] < self.n2 and 0 < fellows[k][2] < self.n3 and self.B[fellows[k][0], fellows[k][1], fellows[k][2]] == 1):    # Si ese vertice (coordenada) es '1'...
                good_fellows.append([fellows[k][0], fellows[k][1], fellows[k][2]])                       # ... la agregamos a la bolsa
                counter_fellows += 1                                                      # Contar el numero de vecinos libres                            
        good_fellows = np.asarray(good_fellows, dtype = int)
        return good_fellows, counter_fellows
    
    def peaks(self, SeedAux):
        """Obtiene los peaks por bloques de una semilla (coord. real)"""
        fellows                       = self.quadrant(SeedAux)  # Calcular fellows de una Coordenada real
        good_fellows, counter_fellows = self.condition(fellows) # Calcular good fellows de los fellows
        
        BAG_direction_peaks = []
        BAG_norm_peaks      = []
        BAG_VoxelPosition   = []
        
        for j in range(0, counter_fellows):                     # Por cada good fellow inicializar un arreglo vacio de peaks y norms
            bag_direction_peaks = []
            bag_norm_peaks      = []
            
            yint_1 = good_fellows[j][0]                     
            yint_2 = good_fellows[j][1]
            yint_3 = good_fellows[j][2]
            
            for o in self.orientaciones:
                peak_voxel = self.P[yint_1, yint_2, yint_3, o:(o+3)]
                norm_peak  = np.linalg.norm(peak_voxel)          # Obtener la norma del peak
                peak_voxel_unit = (peak_voxel)/(norm_peak) if norm_peak != 0 else 0      # Convertir a unitario el peak
                
                if (norm_peak > self.epsilon_1):
                    bag_direction_peaks.append(peak_voxel_unit)           # Agregar el peak unit
                    bag_norm_peaks.append(norm_peak)                      # Agregar su norma
            
            if (len(bag_norm_peaks) > 0):
                BAG_direction_peaks.append(bag_direction_peaks)               # Guardar los peaks por bloques 
                BAG_norm_peaks.append(bag_norm_peaks)                         # Guardar las normas por bloques
                BAG_VoxelPosition.append([yint_1, yint_2, yint_3])            # Guardar las coordenadas int
            
        BAG_VoxelPosition    = np.asarray([BAG_VoxelPosition])
        BAG_VoxelPosition    = np.squeeze(BAG_VoxelPosition, axis=0)
        CounterPeaks         = len(BAG_norm_peaks)
        CounterVoxelPosition = len(BAG_VoxelPosition)
        
        return BAG_direction_peaks, BAG_norm_peaks, CounterPeaks, BAG_VoxelPosition, CounterVoxelPosition

    def compute_peaks_for_family(self, fam, streamlines_fam):
        """
        Para cada partícula activa en la familia:
          1. Llamar a get_seeds.peaks(pos) → (BAG_direction_peaks, BAG_norm_peaks, counter_peaks, BAG_VoxelPosition, CounterVoxelPosition)
          2. “Entregar” esos arrays a la partícula: part.collect_neighbor_peaks(...)
          3. part.compute_probabilities(self.w, self.z, self.sigma, self.epsilon_2)
          4. Construir all_peaks = [p for block in BAG_direction_peaks for p in block]
          5. cdf = compute_cdf(part.peak_weights)
          6. (idx_sel, _) = part.choose_peak(all_peaks, cdf)
          7. part.direction = all_peaks[idx_sel]
        Si la partícula está inactiva o no hay picos, se le asigna dirección cero.
        """
        bag_direction_peaks_family = []

        for i, part in enumerate(fam.particles):
            if not part.active:
                # Partícula inactiva → dirección = (0,0,0)
                bag_direction_peaks_family.append(np.zeros(3, dtype=float))
                continue

            pos = part.position
            streamline_actual = streamlines_fam[i]

            # 1. Extraer picos vecinos del voxel en 'pos'
            BAG_direction_peaks, BAG_norm_peaks, counter_peaks, BAG_VoxelPosition, CounterVoxelPosition = self.peaks(pos)

            if counter_peaks > 0:
                # 2. Almacenar datos filtrados en la propia partícula
                part.collect_neighbor_peaks(BAG_direction_peaks, BAG_norm_peaks, BAG_VoxelPosition)

                # 3. Calcular φ, ψ₁, ψ₂ y poblar part.peak_weights
                part.compute_probabilities(self.w, self.z, self.sigma, self.epsilon_2)

                # 4. Aplanar la lista de picos en all_peaks:  
                all_peaks = [p for block in BAG_direction_peaks for p in block]

                # 5. Generar la CDF a partir de part.peak_weights
                cdf = compute_cdf(part.peak_weights)

                # 6. Muestrear un índice usando el método de la partícula
                idx_sel, _ = part.choose_peak(cdf)

                # 7. Asignar la dirección seleccionada a part.direction
                part.direction = all_peaks[idx_sel].copy()
                d1 = part.direction

            else:
                # No hay picos válidos → dirección nula y la partícula se puede desactivar si quieres
                d1 = np.zeros(3, dtype=float)

            bag_direction_peaks_family.append(d1)

        return bag_direction_peaks_family
    
    def run_tractography(self, num_iteraciones, gamma1, gamma2, gamma3):
        self.streamlines = []
        for fam in self.familias:
            for part in fam.particles:
                self.streamlines.append([part.position.copy()])

        for it in range(num_iteraciones):
            # 1. Chequear si queda alguna partícula activa
            alguna_activa = any(part.active for fam in self.familias for part in fam.particles)
            if not alguna_activa:
                print(f"No quedan partículas activas en la iteración {it}. Terminando.")
                break

            cursor_global = 0
            for fam in self.familias:
                n_part = len(fam.particles)
                streamlines_fam = self.streamlines[cursor_global : cursor_global + n_part]

                # 2. Calcular d1 para cada partícula activa en la familia
                bag_direction_peaks_family = self.compute_peaks_for_family(fam, streamlines_fam)

                # 3. Actualizar partículas
                fam.update_all_particles(
                    alpha=self.alpha,
                    gamma1=gamma1,
                    gamma2=gamma2,
                    gamma3=gamma3,
                    bag_direction_peaks=bag_direction_peaks_family,
                    streamlines=streamlines_fam
                )

                # 4. Guardar nuevas posiciones en las streamlines
                for i, part in enumerate(fam.particles):
                    if part.active:
                        nueva_pos = part.position.copy()
                        self.streamlines[cursor_global + i].append(nueva_pos)

                cursor_global += n_part 

        return self.streamlines