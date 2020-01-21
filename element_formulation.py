import numpy as np
import scipy as sp
import sympy as sym

from geometry_2d3n import get_element_geometry_2d3n as element_geometry


def create_zero_matrix(a, b):
    output_matrix = []
    for i in range(a):
        output_matrix_row = []
        for j in range(b):
            output_matrix_row.append(0.0)
        output_matrix.append(output_matrix_row)

    return output_matrix


def add_matrices(matrix_a, matrix_b):
    rows = len(matrix_a)
    output_matrix = create_zero_matrix(rows, rows)
    for a in range(rows):
        for b in range(rows):
            output_matrix[a][b] = matrix_a[a][b] + matrix_b[a][b]

    return output_matrix


def create_vector_symbol(symbol, dimension):
    symbol_vector = []
    for i in range(dimension):
        symbol_vector.append(sym.Symbol(symbol + "_{" + str(i + 1) + "}"))

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


def steady_cdr_element_weak_formulation(gauss_point_index,
                                        number_of_gauss_points):
    shape_functions, shape_function_derivatives, _ = element_geometry(
        number_of_gauss_points)

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
            lhs_matrix_ab += gauss_point_shape_functions[
                a] * velocity_convection_operator[b]
            lhs_matrix_ab += effective_viscosity * dNa_dNb
            lhs_matrix_ab += effective_reaction * gauss_point_shape_functions[
                a] * gauss_point_shape_functions[b]

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
    return lhs_matrix, velocity, tau, effective_viscosity, effective_reaction, gauss_point_shape_functions, gauss_point_shape_function_derivatives, velocity_convection_operator


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


def calculate_discrete_upwind_operator_elements(input_matrix):
    number_of_nodes = len(input_matrix)
    discrete_upwind_operator = []
    for a in range(number_of_nodes):
        for b in range(a + 1, number_of_nodes):
            discrete_upwind_operator.append([
                input_matrix[a][b] - input_matrix[b][a],
                "(" + str(a + 1) + "," + str(b + 1) + ")"
            ])

    return discrete_upwind_operator


def print_matrix(input_matrix, heading, symplification_list):
    print(r"-------- " + heading + r" ---------- \\")

    number_of_nodes = len(input_matrix)
    for a in range(number_of_nodes):
        for b in range(number_of_nodes):
            s_str = sym.latex(sym.simplify(input_matrix[a][b], rational=True))
            # for pair in symplification_list:
            #     s_str = s_str.replace(sym.latex(pair[0]), pair[1])

            print("Matrix (" + str(a + 1) + ", " + str(b + 1) + ") :")
            print(r"\begin{dmath}")
            print(s_str)
            print(r"\end{dmath}")


def print_vector(input_vector, heading, symplification_list):
    print(r"-------- " + heading + r" ---------- \\")

    number_of_nodes = len(input_vector)

    for a in range(number_of_nodes):
        s_str = sym.latex(sym.simplify(input_vector[a], rational=True))
        # for pair in symplification_list:
        #     s_str = s_str.replace(sym.latex(pair[0]), pair[1])
        print("Vector (" + str(a + 1) + ") :")
        print(r"\begin{dmath}")
        print(s_str)
        print(r"\end{dmath}")


def calculate_symmetric_matrix(input_matrix):
    number_of_nodes = len(input_matrix)

    output_matrix = create_zero_matrix(number_of_nodes, number_of_nodes)
    for a in range(number_of_nodes):
        for b in range(a, number_of_nodes):
            output_matrix[a][b] = 0.5 * (input_matrix[a][b] +
                                         input_matrix[b][a])
            output_matrix[b][a] = output_matrix[a][b]

    return output_matrix


def calculate_antisymmetric_matrix(input_matrix):
    number_of_nodes = len(input_matrix)

    output_matrix = create_zero_matrix(number_of_nodes, number_of_nodes)
    for a in range(number_of_nodes):
        for b in range(number_of_nodes):
            output_matrix[a][b] = 0.5 * (input_matrix[a][b] -
                                         input_matrix[b][a])

    return output_matrix


def construct_discrete_upwind_matrix(input_matrix, tau, velocity,
                                     effective_viscosity, effective_reaction,
                                     gauss_shape_functions,
                                     gausss_shape_function_derivatives,
                                     velocity_convection_operator):
    number_of_nodes = len(gauss_shape_functions)
    dimension = len(velocity)

    output_matrix = create_zero_matrix(number_of_nodes, number_of_nodes)

    # gamma = sym.Symbol(r"\gamma")
    gamma = 1.0

    for a in range(number_of_nodes):
        for b in range(a + 1, number_of_nodes):
            value = 0.0

            dNadNb = 0.0
            for i in range(dimension):
                dNadNb += gausss_shape_function_derivatives[a][
                    i] * gausss_shape_function_derivatives[b][i]

            value += effective_viscosity * dNadNb
            value += tau * (gauss_shape_functions[a] * effective_reaction +
                            velocity_convection_operator[a]) * (
                                gauss_shape_functions[b] * effective_reaction +
                                velocity_convection_operator[b])
            value += gauss_shape_functions[a] * gauss_shape_functions[
                b] * effective_reaction
            value += sym.Max(
                gauss_shape_functions[a] * velocity_convection_operator[a],
                gauss_shape_functions[b] * velocity_convection_operator[b])

            output_matrix[a][b] = -1.0 * gamma * value
            output_matrix[b][a] = output_matrix[a][b]

    for a in range(number_of_nodes):
        dii = 0.0
        for b in range(number_of_nodes):
            if (a != b):
                dii += output_matrix[a][b]
                # dii += output_matrix[a][b] + input_matrix[a][b]
        # output_matrix[a][a] = -1.0 * dii - sym.Abs(input_matrix[a][a])
        output_matrix[a][a] = -1.0 * dii

    return output_matrix


def main():
    gauss_point_index = 0
    number_of_gauss_points = 1

    lhs_matrix, velocity, tau, effective_viscosity, effective_reaction, gauss_point_shape_functions, gauss_point_shape_function_derivatives, velocity_convection_operator = steady_cdr_element_weak_formulation(
        gauss_point_index, number_of_gauss_points)
    row_sum_vector = calculate_matrix_row_sum(lhs_matrix)

    symmetric_lhs_matrix = calculate_symmetric_matrix(lhs_matrix)
    anti_symmetric_lhs_matrix = calculate_antisymmetric_matrix(lhs_matrix)

    discrete_upwind_elements = calculate_discrete_upwind_operator_elements(
        lhs_matrix)
    number_of_nodes = len(row_sum_vector)
    dimension = len(velocity)

    symplification_list = []
    for a in range(number_of_nodes):
        current_value = 0.0
        for i in range(dimension):
            current_value += gauss_point_shape_function_derivatives[a][
                i] * velocity[i]
        symplification_list.append([
            current_value, r"\underline{u}\cdot\nabla N^{" + str(a + 1) + "}"
        ])

    # print_vector(row_sum_vector, "Row sums", symplification_list)
    # print_matrix(lhs_matrix, "LHS Matrix", symplification_list)

    # print("--------- Discrete Upwind Operator --------")
    # for element in discrete_upwind_elements:
    #     print(element[1])
    #     print(sym.latex(sym.simplify(element[0])))

    # print_matrix(symmetric_lhs_matrix, "Symmetric LHS Matrix", symplification_list)
    # print_matrix(anti_symmetric_lhs_matrix, "Anti-symmetric LHS Matrix", symplification_list)

    discrete_diffusion_matrix = construct_discrete_upwind_matrix(
        lhs_matrix, tau, velocity, effective_viscosity, effective_reaction,
        gauss_point_shape_functions, gauss_point_shape_function_derivatives,
        velocity_convection_operator)

    print_matrix(discrete_diffusion_matrix, "Discrete diffusion matrix",
                 symplification_list)

    a_tilde = create_zero_matrix(number_of_nodes, number_of_nodes)
    for a in range(number_of_nodes):
        for b in range(number_of_nodes):
            a_tilde[a][b] = lhs_matrix[a][b] + discrete_diffusion_matrix[a][b]

    velocity_magnitude = sym.Symbol(r"|\underline{u}|")

    # calculate stream line diffusion matrix
    k_s = sym.Symbol("k_s")
    matrix_diffusion_stream_line = create_zero_matrix(number_of_nodes,
                                                      number_of_nodes)
    for a in range(number_of_nodes):
        for b in range(number_of_nodes):
            matrix_diffusion_stream_line[a][
                b] = k_s * velocity_convection_operator[
                    a] * velocity_convection_operator[b] / (
                        velocity_magnitude * velocity_magnitude)

    # calculate cross wind diffusion matrix
    k_c = sym.Symbol("k_c")
    matrix_diffusion_crosswind = create_zero_matrix(number_of_nodes,
                                                    number_of_nodes)
    for a in range(number_of_nodes):
        for b in range(number_of_nodes):
            dNa_dNb = 0.0
            for i in range(dimension):
                dNa_dNb += gauss_point_shape_function_derivatives[a][
                    i] * gauss_point_shape_function_derivatives[b][i]

            matrix_diffusion_crosswind[a][b] = k_c * (
                dNa_dNb - velocity_convection_operator[a] *
                velocity_convection_operator[b] /
                (velocity_magnitude * velocity_magnitude))

    print_matrix(matrix_diffusion_stream_line, "stream line diffusion matrix",
                 symplification_list)
    print_matrix(matrix_diffusion_crosswind, "cross wind diffusion matrix",
                 symplification_list)
    print_matrix(
        add_matrices(matrix_diffusion_stream_line, matrix_diffusion_crosswind),
        "total diffusion matrix", symplification_list)

    print_vector(calculate_matrix_row_sum(matrix_diffusion_crosswind), "cross wind row sum", symplification_list)
    print_vector(calculate_matrix_row_sum(matrix_diffusion_stream_line), "stream line row sum", symplification_list)

    # a_tilde_row_sum = calculate_matrix_row_sum(a_tilde)
    # print_vector(a_tilde_row_sum, "a_tilde row sum", symplification_list)
    # print_matrix(a_tilde, "a_tilde matrix", symplification_list)


if __name__ == "__main__":
    main()