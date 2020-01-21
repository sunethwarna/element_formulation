import numpy as np
import scipy as sp
import sympy as sym

from geometry_2d3n import get_element_geometry_2d3n as element_geometry


def create_vector_symbol(symbol, dimension):
    symbol_vector = []
    for i in range(dimension):
        symbol_vector.append(sym.Symbol(symbol + "_{" + str(i) + "}"))

    return symbol_vector


def get_convection_operator(vector, gauss_shape_function_derivatives):
    convective_vector = []
    number_of_nodes = len(gauss_shape_function_derivatives)
    dimension = len(vector)

    for a in range(number_of_nodes):
        current_value = 0.0
        for i in range(dimension):
            current_value += vector[i] * gauss_shape_function_derivatives[a][i]
        convective_vector.append(current_value)

    return convective_vector


def steady_cdr_element_weak_formulation(gauss_point_index):
    shape_functions, shape_function_derivatives, _ = element_geometry(1)
    
    gauss_point_shape_functions = shape_functions[gauss_point_index]
    gauss_point_shape_function_derivatives = shape_function_derivatives[
        gauss_point_index]
    
    number_of_nodes = len(gauss_point_shape_functions)
    dimension = len(gauss_point_shape_function_derivatives[0])

    # creating gauss point symbol quantities
    velocity = create_vector_symbol("u", dimension)
    effective_viscosity = sym.Symbol(r"\tilde{\nu}_\phi")
    effective_reaction = sym.Symbol(r"\tilde{s}_\phi")
    tau = sym.Symbol(r"\tau")

    # get operators
    velocity_convection_operator = get_convection_operator(
        velocity, gauss_point_shape_function_derivatives)

    lhs_matrix = []
    for a in range(number_of_nodes):
        lhs_matrix_row = []
        for b in range(number_of_nodes):
            lhs_matrix_ab = 0.0

            dNa_dNb = 0.0
            for i in range(dimension):
                dNa_dNb += gauss_point_shape_function_derivatives[a][
                    i] * gauss_point_shape_function_derivatives[b][i]

            # adding main terms from steady cdr weak formulation
            lhs_matrix_ab += gauss_point_shape_functions[a] * velocity_convection_operator[b]
            lhs_matrix_ab += effective_viscosity * dNa_dNb
            lhs_matrix_ab += effective_reaction * gauss_point_shape_functions[a] * gauss_point_shape_functions[b]

            # # adding SUPG stabilization term
            lhs_matrix_ab += tau * (
                velocity_convection_operator[a] +
                effective_reaction * gauss_point_shape_functions[a]
            ) * velocity_convection_operator[b]
            lhs_matrix_ab += tau * (
                velocity_convection_operator[a] +
                effective_reaction * gauss_point_shape_functions[a]
            ) * effective_reaction * gauss_point_shape_functions[b]

            lhs_matrix_row.append(lhs_matrix_ab)

        lhs_matrix.append(lhs_matrix_row)
    print(lhs_matrix)
    return lhs_matrix

def calculate_matrix_row_sum(input_matrix):
    number_of_rows = len(input_matrix)
    number_of_columns = len(input_matrix[0])

    row_sum = []
    for i in range(number_of_rows):
        current_row_sum = 0.0
        for j in range(number_of_columns):
            current_row_sum += input_matrix[i][j]
        row_sum.append(current_row_sum)

    return row_sum

def main():
    lhs_matrix = steady_cdr_element_weak_formulation(0)
    row_sum_vector = calculate_matrix_row_sum(lhs_matrix)
    number_of_nodes = len(row_sum_vector)

    print("------- Row sums -------")
    for a in range(number_of_nodes):
        print("Row " + str(a+1) + ": ")
        print(sym.latex(sym.simplify(row_sum_vector[a])))


if __name__ == "__main__":
    main()