# JopSlantStack

The slant stack is implemented in one of two modes (time or depth).  In depth mode, the implementation is the wave-number domain with a change of variables from the vertical wavenumber ``k_z`` and the horizontal subsurface half-offset wavenumber ``k_h`` to the angle of incidence (half of opening angle) ``\theta`` measured with respect to the normal to a planar reflector. An application of this mode would be the conversion of subsurface offset gathers (SSOG) into angle gathers (SSAG). In time mode, the implementation is in the frequency/wavenumber domain with a change of variables from frequency ``\omega`` and surface offset wavenumber ``k_h`` to the ray parameter ``p``. An application of this mode would be the tau-p transform of shot gathers. The change of variables is defined via a mapping from ``k_z`` and ``\theta`` (or ``p``) to ``k_h``. 

We start by noting the definition of the ray parameter ``p`` with takeoff angle ``\gamma`` (with respect to the vertical)

```math
p = \frac{\sin\gamma}{c}.
```

Note that the ray parameter is preserved in the system for each wave. The relation between the takoff angle and the plane wave can be derived from the dispersion relation and is given by

```math
\sin(\gamma) = \frac{k_x}{\omega/c},
```

so that the mapping between ``k_h=k_x`` and ``p`` is

```math
p = \frac{k_x}{\omega}.
```

In time mode, this provides the mapping between ``k_h`` and ``p`` via the relation
```math
d(\omega,p) = \delta(k_h - p\omega)d(\omega,k_h).
```

In depth mode, we start from the extended imaging condition between source and receiver plane waves

```math
I(x,z,h) = e^{i(k_{sx}(x+h)+k_{sz}z)} e^{-i(k_{rx}(x-h)+k_{rz}z)}=e^{i((k_{sx}-k_{rx})x + (k_{sz}-k_{rz})z + (k_{sx}+k_{rx})h)}.
```

The image vertical wavenumber and subsurface half-offset wavenumber at a reflection point are given by ``(k_z,k_h) = (k_{sz}-k_{rz}, k_{sx}+k_{rx})``. Assuming a horizontal reflector, we have ``k_{sx} = k_{rx}``, ``k_{sz} = -k_{rz}``, and ``\gamma_s = 2\pi - \gamma_r = \theta - \pi``. Given that ``\frac{k_{sx}}{k_{sz}} = -\tan(\gamma_s)`` we deduce

```math
k_h = 2k_{sx} = -2k_{sz}\tan(\gamma_s) = -k_z \tan(\theta).
```

With some trigonometric manipulations, it can be shown that this relation still holds for non-horizontal reflections.

Thus, the mapping between ``k_h`` and ``\theta`` is given by

```math
d(k_z,\theta) = \delta(k_h + k_z \tan(\theta))d(k_z,k_h).
```

In 3D, the mapping from ``k_{hx},k_{hy}`` to reflection angle ``\theta`` at a given azimuth ``\phi`` depends on the geologic dip defined by the dip ``\alpha`` and azimuth ``\beta``. Without further derivations, it can be shown that this mapping is given by

```math
d(k_z,\theta,\phi) = \delta(k_{hx} + \xi\, k_z \tan(\theta)\cos(\phi))\delta(k_{hy} + \xi\,k_z \tan(\theta)\sin(\phi))d(k_z,k_{hx},k_{hy}),
```
where ``\xi`` is a correction factor given by

```math
\xi = \sqrt(\frac{1 + \tan^2(\alpha)}{1 + \tan^2(\alpha)\cos^2(\beta - \phi)}).
```