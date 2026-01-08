# JopSlantStack

The slant stack is implemented in the wave-number domain using a change of variables from the vertical wavenumber ``k_z`` and the horizontal offset wavenumber ``k_h`` to the ray parameter ``p_h``.  The change of variables is defined via a mappting from ``k_z`` and ``p`` to ``k_h`` that is derived from dispersion relations for the downward propagating wave 

```math
\frac{\omega^2}{c^2} = k_{gx}^2 + k_{gz}^2
```

and the reflected wave

```math
\frac{\omega^2}{c^2} = k_{sx}^2 + k_{sz}^2.
```

Next we note the definition of the ray parameter ``p`` with incidence angle ``\theta_s`` and reflected angle ``\theta_g``, and assume that ``\theta=\theta_s=\theta_g``:

```math
p = \frac{1}{c}(\sin\theta_g + \sin\theta_s) = \frac{2}{c}(\sin\theta)
```

The relation between the incidence angle and the plane wave is given by:

```math
\sin(\theta) = \frac{k_x}{\omega/c}
```

So that,

```math
p = 2\frac{k_x}{\omega}
```

The offset wavenumber is ``k_h=k_{sx}+k_{gx}`` and we make the assumption that ``k_{sx}=k_{gx}`` so that ``k_{sx}=k_{gx}=k_x``, ``k_h=2k_x`` and ``k_{gz}=k_{sz}=k_z``.  Now, we can find the mapping from ``p_h`` and ``k_z`` to ``k_h`` starting from the above dispersion relation we find:

```math
\begin{aligned}
k_h^2 &= \frac{\omega^2}{c^2} - k_z^2 \\
k_h   &= \sqrt{\frac{\omega^2}{c^2} - k_z^2} \tag{1}
\end{aligned}
```

dividing both sides by ``\omega/c``,

```math
\frac{ck_h}{\omega} = \sqrt{1 - \left(\frac{ck_z}{\omega}\right)^2}
```

note that ``p=k_h/\omega`` so that,

```math
\begin{aligned}
cp                              &= \sqrt{1 - \left(\frac{ck_z}{\omega}\right)^2} \\
(cp)^2                          &= 1 - \left(\frac{ck_z}{\omega}\right)^2 \\
\left(\frac{c}{\omega}\right)^2 &= \frac{1}{k_z^2}\left(1 - (cp)^2\right) \tag{2}
\end{aligned}
```

Substituting equation 2 into equation 1 gives,

```math
\begin{aligned}
k_h^2 &= k_z^2\left(1 - (cp)^2\right)^{-1} - k_z^2 \\
k_h   &= k_z\left[\left(1 - (cp)^2\right)^{-1} - 1\right] \tag{3}
\end{aligned}
```

We use equation 3 to map between ``k_h`` and ``p``.

