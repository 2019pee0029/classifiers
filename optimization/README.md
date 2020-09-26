## Optimization

Python implementation of first and second order methods used in neural networks back propagation. 

### Methods available
1. Gradient Descent
1. Bisection
1. Newton
1. Modified Newton
1. Levenberg-Marquardt
1. Quasi-Newton Methods
1. One Step Secant
1. Conjugate Gradient Methods

## Files description
- optimization.py:      Python code
- optimization.pdf:     Full description of the methods available, deployment and test results.
- optimization.Rmd:     R Markdown code to create the report.

## Usage
Include the desired objective function, derivative and hessian in the ```gradient``` method, then just call one of the available methods to get the local or global minimum.

```
gradient(x, equation = 1)
    Compute the gradient and hessian of each equation.
    
    Input:
        x:              vector with initial values x1 and x2
        equation:       1: f(x1, x2) = x1^2 + 2.x2^2 - 2.x1.x2 - 2.x2
                        2: f(x1, x2) = r1(x)**2 + r2(x)**2 (method Levenberg-Marquardt)
                        3: f(x1, x2) = (x1 - 2)^4 + (x1 - 2.x2)^2
    Output
        f, g, h:        function, gradient and hessian
```

## Reference

M. S. Bazaraa, H. D. Sherali, C. M. Shetty. (2006) - "Nonlinear Programming - Theory and Algorithms" - 3rd edition

