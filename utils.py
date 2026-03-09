from dipy.io.streamline import load_tractogram
from fury import window, actor
import numpy as np

def compute_unit_tangent(streamline):
    """
    Calcula el vector tangente unitario a la trayectoria más reciente de una partícula.
    Si la trayectoria tiene menos de 2 puntos, retorna un vector nulo.

    Parámetro:
    - streamline: lista de coordenadas [[x0, y0, z0], ..., [xk, yk, zk]]

    Retorna:
    - vector unitario de dirección (shape (3,))
    """
    if len(streamline) < 2:
        return np.zeros(3, dtype=float)
    
    p1 = np.array(streamline[-2])
    p2 = np.array(streamline[-1])
    vec = p2 - p1
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else np.zeros(3, dtype=float)

def phi(w, BAG_direction_peaks, BAG_norm_peaks, CounterPeaks, streamline_actual):
        """ Calcula el grado de paralelismo entre el la direcccion actual (d) y los peaks (q)"""
        PARALLELISM           = []
        tamanio               = len(streamline_actual)
        current_position      = streamline_actual[tamanio-2] - streamline_actual[tamanio-1]
        current_position_norm = np.linalg.norm(current_position)
        if current_position_norm == 0:
            current_position_unit = np.zeros_like(current_position)
        else:
            current_position_unit = current_position / current_position_norm
        
        for i in range(0, CounterPeaks):                            # No. de bloque
            parallelism = []
            for j in range(0, len(BAG_direction_peaks[i])):         # No. de peak en el bloque
                peak = BAG_direction_peaks[i][j]
                peak = peak.reshape((1,3))
                product = BAG_norm_peaks[i][j]*( ( np.abs( np.vdot(peak.T, current_position_unit) ) )**w )
                parallelism.append(product)
            PARALLELISM.append(parallelism)
        
        return PARALLELISM

def psi_1(epsilon_2, z, streamline_actual, BAG_VoxelPosition, CounterVoxelPosition):
        """Mide el nivel de colinearidad entre los siguientes vectores: 
        La direccion actual (d) y [x - y]."""
        COLLINEARITY          = []
        tamanio               = len(streamline_actual)
        
        current_position      = streamline_actual[tamanio-1] - streamline_actual[tamanio-2]
        current_position_norm = np.linalg.norm(current_position)
        if current_position_norm == 0:
            current_position_unit = np.zeros_like(current_position)
        else:
            current_position_unit = current_position / current_position_norm
    
        x = streamline_actual[tamanio-1]
        
        for i in range(0, CounterVoxelPosition):
            v            = x - BAG_VoxelPosition[i]
            v            = v.reshape((1,3))
            numerator    = np.abs( np.vdot(current_position.T, v) )
            denominator  = np.linalg.norm(v)
            if (denominator > epsilon_2):
                quotient     = (numerator/denominator)**z
                COLLINEARITY.append(quotient)
            else:
                COLLINEARITY.append(0)
        return COLLINEARITY

def psi_2(sigma, streamline_actual, BAG_VoxelPosition, CounterVoxelPosition):
        """Pesa la proximidad Euclideana de la posicion actual (x) al 
        centro del voxel vecino (y)"""
        PROXIMITY             = []
        tamanio               = len(streamline_actual)
        x                     = streamline_actual[tamanio-1]
        cte                   = (-1)/(2*sigma**2)
        
        for i in range(0, CounterVoxelPosition):
            value       = cte*((np.linalg.norm(x - BAG_VoxelPosition[i]))**2)
            exponential = np.exp(value)
            PROXIMITY.append(exponential)
        
        return PROXIMITY

def compute_pdf(PARALLELISM, COLLINEARITY, PROXIMITY, CounterPeaks):
    """
    Recibe:
      - PARALLELISM: lista de listas φ_blocks[r][j]
      - COLLINEARITY: lista de ψ1[r] para cada bloque r
      - PROXIMITY: lista de ψ2[r] para cada bloque r
      - CounterPeaks: número de bloques (r)

    Retorna:
      - Una lista plana 1D con todas las densidades φᵣⱼ * ψ1ᵣ * ψ2ᵣ,
        en el mismo orden en que se verían los picos aplanados.
    """
    all_density = []
    for r in range(CounterPeaks):
        for j in range(len(PARALLELISM[r])):
            prob  = PARALLELISM[r][j] * COLLINEARITY[r] * PROXIMITY[r]
            all_density.append(prob)
    return all_density

def compute_cdf(density):
    """
    Convierte una lista de densidades en su correspondiente función de distribución acumulada (CDF).
    """
    density = np.array(density, dtype=float)
    total = np.sum(density)

    if total == 0:
        list_density_unit = np.zeros_like(density)
    else:
        list_density_unit = density / total

    cumulative_density = np.cumsum(list_density_unit)
    return np.insert(cumulative_density, 0, 0.0)

def visualize_streamlines(ruta_streamlines, ruta_mask):
    # Cargar el tractograma
    sft = load_tractogram(ruta_streamlines, ruta_mask, bbox_valid_check=False)

    # Crear actor de fibras
    streamlines_actor = actor.line(sft.streamlines)

    # Crear escena
    scene = window.Scene()
    scene.add(streamlines_actor)

    # Mostrar ventana interactiva
    window.show(scene)