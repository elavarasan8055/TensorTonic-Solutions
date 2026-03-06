def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    xmin=x0
    for _ in range(steps):
        xmin=xmin-(lr*(2*a*xmin+b))
    return xmin