# JopSlantStack

The slant stack is implemented in the wave-number domain using a change of variables from the vertical wavenumber ``k_z`` and the horizontal offset wavenumber ``k_h`` to the ray parameter ``p_h``.  The change of variables is defined via a mappting from ``k_z`` and ``p`` to ``k_h`` that is derived from dispersion relations for the downward propagating wave 

``\frac{\omega^2}{c^2} = k_{gx}^2 + k_{gz}^2``

and the reflected wave

``\frac{\omega^2}{c^2} = k_{sx}^2 + k_{sz}^2``.

Next we note the definition of the ray parameter ``p`` with incidence angle ``\theta_s`` and reflected angle ``\theta_g``, and assume that ``\theta=\theta_s=\theta_g``:

``
p = \frac{1}{c}(\sin\theta_g + \sin\theta_s) = \frac{2}{c}(\sin\theta) = \frac{2}{c}(\frac{ck_x}{\omega})
``

The relation between the incidence angle and the plane wave is given by:

``
sin(\theta) = \frac{k_x}{\omega/c}
``

So that,

``
p = 2\frac{k_x}{\omega}
``

The offset wavenumber is ``k_h=k_{sx}+k_{gx}`` and we make the assumption that ``k_{sx}=k_{gx}`` so that ``k_{sx}=k_{gx}=k_x``, ``k_h=2k_x`` and ``k_{gz}=k_{sz}=k_z``.  Now, we can find the mapping from ``p_h`` and ``k_z`` to ``k_h`` starting from the above dispersion relation we find:

``
k_h^2 = \frac{\omega^2}{c^2} - k_z^2 \\
k_h = \left(\frac{\omega^2}{c^2} - k_z^2\right)
``



