# Tractografía Cerebral Basada en Flocking

Implementación en Python de un algoritmo de tractografía por resonancia magnética de difusión (dMRI) basado en el modelo de **flocking de partículas**. El sistema propaga trayectorias (streamlines) combinando la dirección local de los picos de difusión, la inercia de la trayectoria y una fuerza de cohesión entre partículas vecinas.

---

## Estructura del repositorio

```
tractografia-flocking/
│
├── main.py            # Punto de entrada: configuración de parámetros y ejecución
├── get_seeds.py       # Generación de semillas, subvoxelización, selección de picos y bucle principal
├── family.py          # Clase Family: agrupa partículas y calcula la fuerza de flocking
├── particle.py        # Clase Particle: estado, actualización de posición y muestreo de picos
├── utils.py           # Funciones auxiliares: tangente, φ, ψ₁, ψ₂, PDF, CDF y visualización
│
├── data/              # (no incluida) Archivos .nii de entrada
└── output/            # (no incluida) Streamlines generadas en formato .tck
```

> Las carpetas `data/` y `output/` **no se incluyen en el repositorio** por el tamaño de los archivos. Ver sección [Datos](#datos) para instrucciones.

---

## Descripción de los módulos

### `main.py`
Carga los archivos `.nii` de entrada, define los parámetros del modelo, instancia el objeto `get_seeds` y ejecuta la tractografía. Guarda el resultado en formato `.tck` y lanza la visualización interactiva.

### `get_seeds.py`
Contiene la clase `get_seeds`, que centraliza:
- Lectura o generación aleatoria de semillas desde la máscara FA.
- Filtrado de semillas sin picos válidos.
- Subvoxelización: generación de `m` coordenadas reales por semilla dentro de una esfera de radio `ratio`.
- Cálculo de los picos direccionales en voxeles vecinos (función `peaks`).
- Selección probabilística del pico activo por partícula (`compute_peaks_for_family`).
- Bucle principal de tractografía (`run_tractography`).

### `family.py`
Contiene la clase `Family`, que agrupa un conjunto de partículas (una semilla entera + `m` subvoxélicas). Responsable de:
- Calcular la fuerza de flocking `d3` entre partículas activas del grupo.
- Coordinar la actualización de todas las partículas llamando a `Particle.update_particle`.
- Aplicar los criterios de parada (límites del volumen y máscara FA).

### `particle.py`
Contiene la clase `Particle`, que representa una sola trayectoria. Cada partícula mantiene:
- `position`: coordenada real actual `(x, y, z)`.
- `direction`: vector de dirección actual.
- `streamline`: lista de posiciones visitadas.
- `active`: estado booleano.

Métodos principales:
- `update_particle`: combina `d1`, `d2` y `d3` con sus pesos, normaliza y actualiza posición.
- `compute_probabilities`: calcula `φ`, `ψ₁`, `ψ₂` y muestrea el pico activo mediante la CDF.
- `choose_peak`: muestrea un índice desde la CDF con `r ~ U(0,1)`.

### `utils.py`
Funciones matemáticas y de visualización:
- `compute_unit_tangent`: vector tangente unitario entre los dos últimos puntos de la streamline.
- `phi`: medida de paralelismo entre la dirección actual y los picos vecinos.
- `psi_1`: medida de colinealidad entre la dirección y el vector hacia el centro del voxel vecino.
- `psi_2`: proximidad euclidiana gaussiana al centro del voxel vecino.
- `compute_pdf`: producto de `φ`, `ψ₁` y `ψ₂` para cada pico candidato.
- `compute_cdf`: conversión de la PDF en CDF acumulada.
- `visualize_streamlines`: carga un `.tck` y lo muestra con FURY/Dipy.

---

## Instalación

Se recomienda usar un entorno virtual:

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install numpy nibabel dipy fury
```

---

## Datos

Los archivos de entrada deben colocarse en la carpeta `data/`:

| Archivo                          | Descripción                                      |
|----------------------------------|--------------------------------------------------|
| `mask_FA_0p2.nii`                | Máscara binaria (FA > 0.2)                       |
| `all_peaks_norm_ISMRM_NNLS.nii`  | Volumen 4D de picos direccionales normalizados   |
| `seed_cc.nii`                    | (Opcional) Máscara de semillas del cuerpo calloso |

---

## Uso

Editar los parámetros en `main.py` según sea necesario y ejecutar:

```bash
python main.py
```

Las streamlines se guardan automáticamente en `output/streamlines_one_peak.tck`. Al finalizar, se lanza la visualización interactiva con FURY.

Para visualizar un resultado previo sin re-ejecutar la tractografía:

```bash
python visualize.py
```

---

## Parámetros principales

| Parámetro   | Descripción                                              | Valor por defecto |
|-------------|----------------------------------------------------------|-------------------|
| `alpha`     | Tamaño de paso por iteración                             | `1.0`             |
| `m`         | Número de subvoxelizaciones por semilla                  | `7`               |
| `ratio`     | Radio de la esfera de subvoxelización                    | `0.3`             |
| `gamma1`    | Peso de la dirección local `d1`                          | `0.1`             |
| `gamma2`    | Peso de la inercia `d2`                                  | `0.9`             |
| `gamma3`    | Peso de la fuerza de flocking `d3`                       | `0.0`             |
| `G`         | Constante de atracción entre partículas                  | `0.1`             |
| `num_peaks` | Número máximo de picos por voxel                         | `5`               |
| `epsilon_1` | Umbral de norma mínima para considerar un pico válido    | `0.1`             |

---

## Formato de salida

Las streamlines se exportan en formato **`.tck`** (MRtrix), compatible con herramientas como MRtrix3, TrackVis y Dipy. Cada streamline es una lista de coordenadas `(x, y, z)` en espacio de voxel, transformadas al espacio RAS mediante el `affine` del volumen de referencia.
