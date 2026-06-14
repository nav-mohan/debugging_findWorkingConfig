## Local Edge Detection algorithm

> A. Gelb and E. Tadmor, "Local edge detection for non-linear signals," Journal of Scientific Computing, 28:279-306, 2006.


- assume you have a uniformly sampled curve 
- $f_0, f_1, f_2, f_3, ... f_N$ at points $x_0,x_1,x_2,...x_N$ where $x_i - x_{i-1} = h$
- Now consider a point $f_{j+1/2}$ in between $(x_j,f_j)$ and $(x_{j+1},f_{j+1})$
- Now do a taylor series expansion of $(f_{i},x_{i})$ about this point $(f_{j-1/2},x_{j-1/2})$ for $ i \in [j-1,j+2]$ so the $\delta x = \pm n\frac{h}{2}$
    - $f_j = f(x) - \frac{h}{2}f'(x) + \frac{h^2}{8}f''(x) - \frac{h^3}{48}f'''(x) \rbrack_{x_{j+1/2}} $
    - $f_{j+1} = f(x) + \frac{h}{2}f'(x) + \frac{h^2}{8}f''(x) + \frac{h^3}{48}f'''(x) \rbrack_{x_{j+1/2}} $
    - $f_{j-1} = f(x) - 3 \frac{h}{2}f'(x) + \frac{1}{2}(\frac{3h}{2})^2f''(x) - \frac{1}{6}(\frac{3h}{2})^3f'''(x) \rbrack_{x_{j+1/2}} $

    - $f_{j+2} = f(x) + 3 \frac{h}{2}f'(x) + \frac{1}{2}(\frac{3h}{2})^2f''(x) + \frac{1}{6}(\frac{3h}{2})^3f'''(x) \rbrack_{x_{j+1/2}} $

- Now take linear combinations of the above 4 expressions such that the first derivative term $f'(x) \rbrack_{x_{j+1/2}}$ vanishes. 

- That gives you the order-3 approximation for the jump between $f_j$ and $f_{j+1}$
    - $\Delta^{(3)}f_{j+1/2} = - f_{j+2} + 3 f_{j+1} - 3 f_j + f_{j-1} $
- An anologue of this in the first-derivative would be just the difference between the two neighboring values. 
    - $\Delta^{(1)}f_{j+1/2} = f_{j+1} - f_j$
- and similarly, for order-5, 
    - $\Delta^{(5)}f_{j+1/2} = f_{j+3} - 5f_{j+2} + 10 f_{j+1} - 10 f_j + 5 f_{j-1} - f{j-2} $

- NOTE: this is only a discrete approximation. The Taylore series expansion for $\Delta^{(3)} f_{j+1/2}$ will have remaining terms of order $h^3, h^5$ and higher i.e $\mathcal{O}(h^3)$

## why do we pick coefficients such that certain terms cancel out? 
- because we want our $\Delta^{(i)}$ jump-measure to be "blind" to terms of order less than $i$ in the taylor-expansion (assuming the original function can be expressed as $\sum c_i (\delta x)^i$ ). 
- This is also called a "high-pass" filter i.e we are only interested in the contributions from the higher-order derivatives of the funtion. 
- for example, consider $\Delta^{(3)}$ detector on a quadratic polynomial $x^2$, sampled at $x_i = 1,2,3,4$ and evaluated at $x_{j+1/2} = 2.5$ 
    $$\Delta^{(3)}f_{j+1/2} = -f_{j+2} + 3 f_{j+1} - 3f_j + f_{j-1} \\
    = -(4^2) + 3(3^2) - 3(2^2) + (1) \\ 
    = 0
     $$
- so the $\Delta^{(3)}$ jump detector vanishes for quadratic polynomials. 
- but it doesnt vanish for polynomials of order $\geq 3$
- for example, consider $\Delta^{(3)}$ detector on a cubic polynomial $x^3$, sampled at $x_i = 1,2,3,4$ and evaluated at $x_{j+1/2} = 2.5$ 
    $$\Delta^{(3)}f_{j+1/2} = -f_{j+2} + 3 f_{j+1} - 3f_j + f_{j-1} \\
    = -(4^3) + 3(3^3) - 3(2^3) + (1) \\ 
    = -6
     $$

- for example, consider $\Delta^{(3)}$ detector on a quartic polynomial $x^4$, sampled at $x_i = 1,2,3,4$ and evaluated at $x_{j+1/2} = 2.5$ 
    $$\Delta^{(3)}f_{j+1/2} = -f_{j+2} + 3 f_{j+1} - 3f_j + f_{j-1} \\
    = -(4^4) + 3(3^4) - 3(2^4) + (1) \\ 
    = -60
     $$

## what is the maximum value of the $\Delta^{(p)}$ jump detector? 
- for order 3, the maximum value is twice the difference in neighboring points. 
    - $\Delta^{(3)} = -f_{j+2} + 3 f_{j+1} - 3 f_{j} + f_{j-1} $
    - now evaluate the $\Delta^{(3)}$ between $x_j$ and $x_{j+1}$ 
    - assuming continuity on the left side of $x_{j+1/2}$ and on the right side of $x_{j+1/2}$ 
        - $f_{j+2} \approx f(x_+)$ (on the right side of $x_{j+1/2})
        - $f_{j+1} \approx f(x_+)$ (on the right side of $x_{j+1/2})
        - $f_{j-1} \approx f(x_-)$ (on the left side of $x_{j+1/2})
        - $f_{j} \approx f(x_-)$ (on the left side of $x_{j+1/2})
    - if the function was indeed continuos, the maximum possible value $\Delta^{(3)}$ would be 
        - $\Delta^{(3)} \approx -f(x_+) + 3 f(x_+) - 3 f(x_-) + f(x_-)  = 2\lbrack f(x_+) - f(x_-)\rbrack $
    - so the $\Delta^{(3)}$ can be atmost twice the difference between neighboring values.
    - and similarly, $\Delta^{(5)}$ can be atmost 6 times the difference between neighboring values.

## how does the sampling-rsolution affect $\Delta${(p)}$ jump detection?
- The Taylor series expansion for $\Delta^{(3)} f_{j+1/2} \sim \mathcal{O}(h^3)$ 
- So if $[j,j+1]$ is a continuous section of the curve, then $\lim_{h \to 0} \Delta^{(3)}f_{j+1/2} \to 0 $ 
- but if $[j,j+1]$ has a discontinuity then the value of $\Delta^{(3)}f_{j+1/2}$ will blow up. 
- if $j = x_{j+1} - x_j$ is large then the blowing up of $\Delta$ is smeared over a large window. Moreover if the window is too wide compared to the sharpness of the discontinuity, then the points $f_j$ and $f_{j+1}$ might end up almost equal 