import numpy as np
from utils import phi, psi_1, psi_2, compute_pdf, compute_cdf

class Particle:
    def __init__(self, position, active=True):
        self.position = np.array(position, dtype=float) # array (shape (3,)) que indica la posición (x,y,z) de la partícula en espacio real.
        self.active = active                            # booleano que indica si la partícula aún participa en el cálculo.
        self.direction = np.zeros(3, dtype=float)       # dirección actual de la partícula (dₜ,ₘ)
        self.streamline = [self.position.copy()]        # Streamline de la partícula: lista de posiciones anteriores
        self.good_fellows = []                          # array de coordenadas enteras de voxeles vecinos
        self.peak_directions = []                       # lista de vectores unitarios qᵣⱼ
        self.peak_norms = []                            # lista de normas bᵣⱼ
        self.peak_weights = []                          # φᵣⱼ·ψ₁ᵣ·ψ₂ᵣ para cada pico

    def deactivate(self):
        """Marca la partícula como inactiva."""
        self.active = False

    def update_particle(self, d1, d2, d3, alpha, gamma1, gamma2, gamma3):
        """
        Combina d1, d2 y d3 con sus pesos, normaliza y mueve la partícula:
        Luego actualiza posición, guarda la nueva dirección en self.direction y agrega a streamline.
        """
        if not self.active:
            return

        # 1.- Combinar los tres componentes
        d_total = gamma1 * d1 + gamma2 * d2 + gamma3 * d3
        norm = np.linalg.norm(d_total)

        # 2.- Si el vector combinado es (casi) cero, desactivamos
        if norm <= 1e-8:
            self.deactivate()
            return

        d_total = d_total / norm

        # 3.- Guardar la dirección resultante
        self.direction = d_total.copy()

        # 4.- Actualizar posición y streamline
        new_pos = self.position + alpha * d_total
        self.position = new_pos
        self.streamline.append(new_pos.copy())    

    def choose_peak(self, cumulative_density):
        """
        Crea un r ∼ U(0,1) y devuelve el índice del pico conforme a cumulative_density.
        """
        I = len(cumulative_density)  # debe ser len(self.peak_weights)+1
        r = np.random.uniform(0, 1)
        for k in range(1, I):
            if cumulative_density[k - 1] <= r < cumulative_density[k]:
                return k - 1, r
            
        # Si debido a imprecisiones de punto flotante `r` no cae en ningún intervalo de la CDF,
        # devolvemos I-2 que equivale al último índice válido de `peaks` (len(peaks)-1).    
        return I - 2, r
    
    def collect_neighbor_peaks(self, BAG_direction_peaks, BAG_norm_peaks, BAG_VoxelPosition):
        """
        Guarda internamente los resultados ya filtrados de get_seeds.peaks(...):
          - voxeles vecinos (coordenadas enteras) en self.good_fellows
          - vectores unit qᵣⱼ en self.peak_directions
          - normas bᵣⱼ en self.peak_norms
        """
        self.good_fellows = BAG_VoxelPosition.copy()
        self.peak_directions = [block.copy() for block in BAG_direction_peaks]
        self.peak_norms = [block.copy() for block in BAG_norm_peaks]
        self.peak_weights = []  # se llenará en compute_probabilities()

    def compute_probabilities(self, w, z, sigma, epsilon_2):
        """
        - φ = phi(w, BAG_direction_peaks, BAG_norm_peaks, CounterPeaks, streamline_actual)
        - ψ₁ = psi_1(epsilon_2, z, streamline_actual, BAG_VoxelPosition, CounterVoxelPosition)
        - ψ₂ = psi_2(sigma, streamline_actual, BAG_VoxelPosition, CounterVoxelPosition)
        - density = compute_pdf(φ_blocks, ψ1_list, ψ2_list, CounterPeaks)
        - cdf = compute_cdf(density)
        - (idx_sel, r) = choose_peak(all_peaks, cdf)

        Almacena la densidad en self.peak_weights y devuelve el índice del pico seleccionado.
        """
        # 1.- Prepara los argumentos
        streamline_actual = self.streamline
        BAG_direction_peaks = self.peak_directions   # lista de bloques
        BAG_norm_peaks = self.peak_norms             # lista de normas
        CounterPeaks = len(BAG_norm_peaks)
        BAG_VoxelPosition = self.good_fellows        # lista de coordenadas enteras
        CounterVoxelPosition = len(BAG_VoxelPosition)

        # 2.- Calcular paralelismo
        phi_blocks = phi(
            w,
            BAG_direction_peaks,
            BAG_norm_peaks,
            CounterPeaks,
            streamline_actual
        )

        # 3.- Calcular colinearidad
        psi1_list = psi_1(
            epsilon_2,
            z,
            streamline_actual,
            BAG_VoxelPosition,
            CounterVoxelPosition
        )

        # 4.- Calcular proximidad
        psi2_list = psi_2(
            sigma,
            streamline_actual,
            BAG_VoxelPosition,
            CounterVoxelPosition
        )

        # 5.- Aplanar todo en una única lista de densidades
        all_density = compute_pdf(
            phi_blocks,
            psi1_list,
            psi2_list,
            CounterPeaks
        )
        self.peak_weights = all_density.copy()

        # 6.- Construir la lista de todos los vectores pico en el mismo orden
        all_peaks = [p for block in BAG_direction_peaks for p in block]

        # 7.- Convertir densidad en CDF y muestrear
        cdf = compute_cdf(all_density)
        idx_sel, r = self.choose_peak(cdf)

        return idx_sel, r