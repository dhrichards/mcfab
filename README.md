# MesoFabMC: Mesoscopic Monte-Carlo Fabric Evolution Model

A fabric evolution model for ice based on the theory of mixtures of continuous diversity (Faria, 2006; Placidi et al. 2010; Richards et al. 2021), but solved by re-expressing the partial differential equation as a stochastic differential equation and using a Monte-Carlo method. This the following advantages over spherical harmonics: 

- Can represent very strong fabrics 
- Stable over much longer timesteps
- Simple to understand


The model incorporates the effects of rigid-body rotation, basal-slip deformation, migration and rotational recrystallization.

## Installation

`pip install meso_fab_mc`

## References

Faria, S.H. (2006). Creep and recrystallization of large polycrystalline masses. I. General continuum theory. https://doi.org/10.1098/rspa.2005.1610

Placidi, L., Greve, R., Seddik, H., Faria S.H., (2010) Continuum-mechanical, Anisotropic Flow model for polar ice masses, based on an anisotropic Flow Enhancement factor. https://doi.org/10.1007/s00161-009-0126-0

Richards, D.H.M,  Pegler, S.S, Piazolo, S., Harlen, O.G., (2021)
The evolution of ice fabrics: A continuum modelling approach validated against laboratory experiments https://doi.org/10.1016/j.epsl.2020.116718

