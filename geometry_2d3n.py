import sympy as sym


def get_element_geometry_2d3n(number_of_gauss_points):
    number_of_nodes = 3
    dimension = 2

    shape_functions = []
    shape_function_derivatives = []

    dimensional_symbols = []
    for i in range(dimension):
        dimensional_symbols.append(sym.Symbol("x_{" + str(i) + "}"))

    # since they are linear elements, jacobian matrix is constant
    inv_det_j = sym.Symbol(r"|J|^{-1}")

    for g in range(number_of_gauss_points):
        gauss_shape_functions = []
        # gauss_shape_function_derivatives = []
        for a in range(number_of_nodes):
            # current_shape_function = sym.Function(r"N^{" + str(a) + "}")
            current_shape_function = sym.Symbol(r"N^{" + str(a+1) + "}")
            gauss_shape_functions.append(1.0/3.0)
            # gauss_shape_function_derivatives_row = []
            # for i in range(dimension):
                # gauss_shape_function_derivatives_row.append(
                #     current_shape_function(dimensional_symbols[i]).diff(
                #         dimensional_symbols[i]))
                # gauss_shape_function_derivatives_row.append(sym.Symbol(r"\frac{dN^{" + str(a+1) + "}}{dx_{" + str(i+1)  + "}}"))
            
        gauss_shape_function_derivatives = [[inv_det_j, 0.0], [0.0, inv_det_j], [-inv_det_j, -inv_det_j]]
        shape_functions.append(gauss_shape_functions)
        shape_function_derivatives.append(gauss_shape_function_derivatives)

    return shape_functions, shape_function_derivatives, dimensional_symbols

if __name__=="__main__":
    shape_functions, shape_function_derivatives, dimensional_symbols = get_element_geometry_2d3n(1)
    print(shape_functions)
    print(shape_function_derivatives)
    print(dimensional_symbols)
