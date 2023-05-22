# mcfab: Monte-Carlo Fabric Evolution Model

A Monte-Carlo based solver for fabric evolution. It is coded in JAX and can be run on GPU, giving ~50x speedup compared to a NumPy implementation. It solves fabric evolution equations of the form:

$$ \frac{\partial f}{\partial t} = - \nabla_i(f v_i) + \lambda {\nabla}^2(f) + f\Gamma $$
which can be expressed as a stochastic differential equation for a large number of grains with $\vec{c}$-axis towards $c_i$ and mass fraction $m$:
$$ \dot{c}_i = v_i + \sqrt{2\lambda} W^Q_i $$
$$ \dot{m} = m\Gamma $$
where
$$ v_i = W_{ij}c_j - \iota_D(D_{ij}c_j - D_{jk}c_kc_jc_i) - \iota_S(\hat{S}_{ij}c_j - \hat{S}_{jk}c_kc_jc_i)  $$

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

- Rathmann et al. (2021): Effect of an orientation-dependent non-linear grain fluidity on bulk directional enhancement factors https://doi.org/10.1017/jog.2020.117

- Richards et al. (2021): The evolution of ice fabrics: A continuum modelling approach validated against laboratory experiments https://doi.org/10.1016/j.epsl.2020.116718


