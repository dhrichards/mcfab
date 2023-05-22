# mcfab: Monte-Carlo Fabric Evolution Model

A Monte-Carlo based solver for fabric evolution. It is coded in JAX and can be run on GPU, giving ~50x speedup compared to a NumPy implementation. It solves ice fabric evolution equations with:
- a contribution to lattice rotation from D, the deformation tensor
- a contribution to lattice rotation from S, the deviatoric stress tensor subject to some flow law
- A browninan motion term, corresponding to a diffusional process
- Migration recrystallization like Placidi et al. (2010)

A number of rheologies are incoprated, such as Rathmann et al. (2021) and the GOLF rheology of Gillet-Chaulet et al. (2005). A parameter set which reproduces Richards et al. (2021) is also included.




## Installation

The package requires JAX and JAXlib to be installed. Instructions can be found [here](https://jax.readthedocs.io/en/latest/). The package also depends on [jaxopt](https://github.com/google/jaxopt) if a fully non-linear flow law is used.

The package can be installed by:
```bash
git clone https://github.com/dhrichards/mcfab.git
cd mcfab
pip install .
```
## Example usage

```python
import jax.numpy as jnp
TODO

```

## References

- Gillet-Chaulet et al. (2005): A user-friendly anisotropic flow law for ice-sheet modeling https://doi.org/10.3189/172756505781829584

- Placidi et al. (2010): Continuum-mechanical, Anisotropic Flow model for polar ice masses, based on an anisotropic Flow Enhancement factor https://doi.org/10.1007/s00161-009-0126-0

- Rathmann et al. (2021): Effect of an orientation-dependent non-linear grain fluidity on bulk directional enhancement factors https://doi.org/10.1017/jog.2020.117

- Richards et al. (2021): The evolution of ice fabrics: A continuum modelling approach validated against laboratory experiments https://doi.org/10.1016/j.epsl.2020.116718


