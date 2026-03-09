import numpy as np
from utils import compute_unit_tangent

class Family:
    def __init__(self, particles, G):
        """
        particles: lista de objetos Particle.
        G: constante de atracción utilizada en el cálculo de flocking (fuerza d3).
        B: máscara binaria (FA > umbral)
        n1, n2, n3: dimensiones del volumen.
        """
        self.particles = particles
        self.G = G
        self.B = None
        self.n1 = self.n2 = self.n3 = None

    def compute_flocking_force(self, index):
        """
        Calcula la fuerza de flocking (d3) para la partícula en la posición 'index'.
        Retorna un vector unitario (3,), o vector nulo si no hay interacción.
        """
        pi = self.particles[index]
        if not pi.active:
            return np.zeros(3, dtype=float)

        F = np.zeros(3, dtype=float)
        for j, pj in enumerate(self.particles):
            if j == index or not pj.active:
                continue
            diff = pj.position - pi.position
            dist = np.linalg.norm(diff)
            if dist > 0:
                F += self.G * (diff / dist)

        norm = np.linalg.norm(F)
        return (F / norm) if norm > 0 else F
    
    def update_all_particles(self, alpha, gamma1, gamma2, gamma3, bag_direction_peaks, streamlines):
        """
        Actualiza la posición de todas las partículas activas en esta familia:
          1) combina d1, d2, d3 y llama a part.update_particle(...)
          2) chequea FA / límites y, si está fuera, desactiva a la partícula
             o, en caso contrario, deja la posición ya agregada en la streamline.
        """
        for i, part in enumerate(self.particles):
            if not part.active:
                continue

            # 1) d1 desde get_seeds
            d1 = bag_direction_peaks[i]

            # 2) d2: tangente de la trayectoria
            d2 = compute_unit_tangent(streamlines[i])
            if np.dot(d1, d2) < 0:
                d1 = -d1

            # 3) d3: fuerza de flocking
            d3 = self.compute_flocking_force(i)

            # 4) Guardar posición anterior (por si hay que deshacer)
            old_pos = part.position.copy()

            # 5) Mover la partícula usando update_particle (combina d1, d2, d3)
            part.update_particle(
                d1=d1,
                d2=d2,
                d3=d3,
                alpha=alpha,
                gamma1=gamma1,
                gamma2=gamma2,
                gamma3=gamma3
            )

            # 6) Si update_particle la desactivó (por norm≈0), saltamos el chequeo de FA
            if not part.active:
                continue

            # 7) Ahora part.position es ya la nueva posición tras alpha·d_total.
            new_vox = np.round(part.position).astype(int)
            x, y, z = new_vox

            # 8) Verificar límites y máscara B
            if (self.B is None or self.n1 is None):
                # No hay máscara FA: dejamos la posición tal cual
                continue
            else:
                # Chequeo de límites
                if not (0 <= x < self.n1 and 0 <= y < self.n2 and 0 <= z < self.n3):
                    part.deactivate()
                    # Revertir la posición y streamline: 
                    #  - sacamos el último paso (ya agregado en update_particle)
                    part.streamline.pop()
                    part.position = old_pos
                    continue

                # Chequeo de FA (máscara B): si sale fuera de sustancia blanca, desactivar
                if self.B[x, y, z] == 0:
                    part.deactivate()
                    # Revertir la posición y streamline
                    part.streamline.pop()
                    part.position = old_pos
                    continue