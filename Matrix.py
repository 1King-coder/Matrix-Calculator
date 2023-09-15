from random import randint
from typing import Union
from math import ceil
from Vectors import Vector
from copy import deepcopy
"""
In this module i aimed to
create some Matrix calculations
to learn more about Matrices and
how i can write a code without using
boilerplates.
"""


class Matrix:
    """
    Matrix Object supporting
    most of basic matrix operations.
    """

    def __init__(self, arg: Union[list, tuple]) -> None:
        """
        Can create a Matrix Object with
        a sent matrix array or create one
        with random numbers with the size sent
        in a tuple (rows, columns).
        """

        self.matrix: list = arg
        self.rows_num: int = len(self.matrix)
        self.cols_num: int = len(self.matrix[0])

        self.size: str = f'{self.rows_num}x{self.cols_num}'

    @property
    def matrix (self) -> list:
        return self.__matrix
    
    @matrix.setter
    def matrix (self, arg):
        if isinstance(arg, list):
            self.__matrix: list = arg
            return

        self.__matrix: list = Matrix.gen_matrix(arg[0], arg[1]).matrix



    @property
    def det (self) -> Union[int, float]:
        return Determinant.det(self.matrix)

    @staticmethod
    def isQuadratic (matrix: list) -> bool:
        return len(matrix) == len(matrix[0])

    @staticmethod
    def map_matrix(func, rows_num: int, cols_num: int) -> 'Matrix':
        """
        Receive a function and return
        a new result Matrix.
        """

        new_matrix: list = []
        
        for i in range(rows_num):
            new_matrix.append([])

            for j in range(cols_num):
                new_matrix[i].append(func(i, j))

        # new_matrix = list(filter(lambda x: x != [], new_matrix))

        return Matrix(new_matrix)
    
    @staticmethod
    def gen_matrix(rows: int, columns: int) -> list:
        """
        Generate a matrix with
        the sent size and random numbers
        between 0 and 100.
        """
        return Matrix.map_matrix(
            lambda i, j: randint(0, 100), 
            rows, 
            columns
        )

    def __repr__(self) -> str:
        """
        Pretty print a Matrix.
        """

        print()
        matrix_str = ''
        
        for i in range(self.rows_num):
            matrix_str += '|'

            for j in range(self.cols_num):
                if j == self.cols_num-1:
                    matrix_str += f'{self.matrix[i][j]:3}'
                    continue

                matrix_str += f'{self.matrix[i][j]:3} '
            matrix_str += '|\n'

        return matrix_str
    
    def __add__(self, other_matrix: 'Matrix') -> 'Matrix':

        """
        Allow to add Matrices with operator.
        """

        if len(other_matrix.matrix) != self.rows_num or len(other_matrix.matrix[0]) != self.cols_num:
            raise IndexError('Matrices must be of the same size.')

        
        return Matrix.map_matrix(
            lambda i, j: self.matrix[i][j] + other_matrix.matrix[i][j],
            self.rows_num,
            self.cols_num
        )
    
    def __sub__(self, other_matrix: 'Matrix') -> 'Matrix':

        """
        Allow to subtract Matrices with operator.
        """

        if len(other_matrix.matrix) != self.rows_num or len(other_matrix.matrix[0]) != self.cols_num:
            raise IndexError('Matrices must be of the same size.')

        new_matrix = []

        return Matrix.map_matrix(
            lambda i, j: self.matrix[i][j] - other_matrix.matrix[i][j],
            self.rows_num,
            self.cols_num
        )
    
    def __mul__(self, param: Union['Matrix', int, float]) -> 'Matrix':
        
        """
        Allow to multiply Matrices or
        a Matrix by a real number with operator.
        """

        matrix_1 = Matrix(self.matrix)
        
        if type(param) == Matrix:

            if self.cols_num != param.rows_num:
                raise IndexError("The number of columns of a matrix must be equal to the other's number of rows.")

            return Matrix.mul_matrices(matrix_1, param)
        
        return Matrix.mul_matrix_by_number(matrix_1, param)

    # Need revision
    def __truediv__(self, param):
        """
        Allow to divide Matrices or
        a Matrix by a real number with operator.
        """
        

        matrix_1 = Matrix(self.matrix)
        
        # Here
        """
        if type(param) == Matrix:

            if self.cols_num != param.rows_num:
                raise IndexError("The number of columns of a matrix must be equal to the other's number of rows.")

            return Matrix.div_matrices(matrix_1, param)
        """
        
        return Matrix.div_matrix_by_number(matrix_1, param)       
    

    @staticmethod
    def mul_matrices(matrix_1: 'Matrix', matrix_2: 'Matrix') -> 'Matrix':
        """
        Multiply a Matrix by other.
        """

        def multiply_matrices (i, j, num = 0):
            for k in range(matrix_1.cols_num):
                num += matrix_1.matrix[i][k] * matrix_2.matrix[k][j]
            return round(num)
        
        return Matrix.map_matrix(
            multiply_matrices,
            matrix_1.rows_num,
            matrix_2.cols_num
        )

    # Need revision
    """
    @staticmethod
    def div_matrices (matrix_1: 'Matrix', matrix_2: 'Matrix') -> 'Matrix':
        
        # Divide a Matrix by other.
        
        return matrix_1 * Matrix.invert(matrix_2)
    """
        

    @staticmethod
    def mul_matrix_by_number(matrix: 'Matrix', num: Union[int, float]) -> 'Matrix':
        """
        Multiply a Matrix by a real number.
        """
        
        return Matrix.map_matrix(
            lambda i, j: matrix.matrix[i][j] * num,
            matrix.rows_num,
            matrix.cols_num
        )
    
    
    @staticmethod
    def div_matrix_by_number(matrix: 'Matrix', num: Union[int, float]) -> 'Matrix':
        """
        Divide a Matrix by a real number.
        """
        
        return Matrix.map_matrix(
            lambda i, j: round(matrix.matrix[i][j] / num, 3),
            matrix.rows_num,
            matrix.cols_num
        )
    

    @staticmethod
    def transpose(matrix: 'Matrix') -> 'Matrix':
        """
        Transpose a Matrix.
        """

        """
        return Matrix.map_matrix(
            lambda i, j: matrix.matrix[j][i],
            matrix.cols_num,
            matrix.rows_num
        )
        """

        return Matrix(list(map(list, zip(*matrix.matrix))))

    @staticmethod
    def minor_coplementary (matrix: list, i: int, j: int) -> Union[int, float]:
        minor_comp = [row[:j] + row[j+1:] for row in (matrix[:i]+matrix[i+1:])]

        return Matrix(minor_comp).det

    @staticmethod
    def cofactor (matrix: list, row: int, col: int()) -> Union[int, float]:
        cofactor = matrix[row][col] * Matrix.minor_coplementary(matrix, row, col) * (-1)**((row+1) + (col+1))

        return cofactor

    @staticmethod
    def order_lines (matrix: list, to_order_matrices: list = [] , index: int = 0):
        sorted_matrix = sorted(
            deepcopy(matrix[index:]),
            key = lambda element: abs(element[index]),
            reverse=True
        )

        to_order_matrices.append(matrix)
        max_module_index = matrix.index(sorted_matrix[0])

        for to_order in to_order_matrices:
            to_order[index], to_order[max_module_index] = to_order[max_module_index], to_order[index]
        
        to_order_matrices = []
        print(Matrix(matrix))

    @staticmethod
    def gaussian_elimination (matrix: 'Matrix') -> 'Matrix':
        escalonated_matrix = deepcopy(matrix.matrix)
        
        for i in range(matrix.rows_num):
            Matrix.order_lines(escalonated_matrix, [], i)

            for j in range(i + 1, matrix.cols_num):

                multiplier = escalonated_matrix[j][i]/escalonated_matrix[i][i]
                escalonated_matrix[j] = [
                    escalonated_matrix[j][k] - escalonated_matrix[i][k]*multiplier
                    for k in range(matrix.rows_num)
                ]

        return Matrix(escalonated_matrix) 
                

    @staticmethod
    def invert(matrix: 'Matrix') -> 'Matrix':
        """
        Verify if the Matrix is invertible
        and if it is, invert it.
        """
        det = matrix.det

        # Base for 2x2 matrices

        if matrix.size == '2x2':
            return Matrix([
                [(matrix.matrix[1][1]/det), (-matrix.matrix[0][1]/det)],
                [(-matrix.matrix[1][0]/det), (matrix.matrix[0][0]/det)]
            ])

        if det != 0:

            minors_matrix = Matrix.map_matrix(
                lambda i, j: Matrix.minor_coplementary(matrix.matrix, i, j), 
                matrix.rows_num, 
                matrix.cols_num
            )

            minors_matrix = Matrix.transpose(minors_matrix)
            inverted_matrix = minors_matrix / det
            
            return inverted_matrix

        raise TypeError('Matrix not invertible.')
        

class Determinant:
    """
    Determinant class 
    containing methods to
    calculate a Matrix determinant.
    """
    @staticmethod
    def det(matrix: list) -> Union[int, float]:
            
        size = f"{len(matrix)}x{len(matrix[0])}"

        if not Matrix.isQuadratic(matrix):
            raise TypeError('Only quadratic matrices can have determinant.')

        if size == '2x2':
            return Determinant.det2x2(matrix)

        if size == '3x3':
            return Determinant.det3x3(matrix)

        return Determinant.detNxN(matrix)
    
    # Base for 2x2 Matrix
    @staticmethod
    def det2x2(matrix: list) -> Union[int, float]:
        return (
            matrix[0][0] * matrix[1][1] -
            matrix[0][1] * matrix[1][0]
        )
    
    # Base for 3x3 Matrix
    @staticmethod
    def det3x3(matrix: list) -> Union[int, float]:
        part_1 = (
            matrix[0][0]*matrix[1][1]*matrix[2][2] +
            matrix[0][1]*matrix[1][2]*matrix[2][0] +
            matrix[0][2]*matrix[1][0]*matrix[2][1]
        )

        part_2 = (
            matrix[0][2]*matrix[1][1]*matrix[2][0] +
            matrix[0][0]*matrix[1][2]*matrix[2][1] +
            matrix[0][1]*matrix[1][0]*matrix[2][2]
        )
        
        return part_1 - part_2
    
    @staticmethod
    def detNxN (matrix: list, det: int = 1) -> Union[int, float]:
        """
        Create P.A = L.U fatoration and solve determinant signal problem
        """

        triangular_matrix = Matrix.gaussian_elimination(Matrix(matrix))
        for i in range(triangular_matrix.rows_num):
            det *= triangular_matrix.matrix[i][i]

        return det

    

        
if __name__ == "__main__":
    """
    Some tests...
    
    A = Matrix([
        [1, 2, 3, 2],
        [5,	6, 4, 4],
        [7, 8, 11, 5],
        [12, 13, 14, 15],
    ])

    B = Matrix((3, 3))

    C = Matrix((4, 4))

    D = Matrix((2, 4))

    print(f'A: \n{A}')
    print(f'B: \n{B}')
    print(f'C: \n{C}')
    print(f'D: \n{D}')

    print(A * C)

    print(A.det)

    print(Matrix.invert(A))

    print(D * ((A + C) * 2) - D * 10)
    """ 
    
    A = Matrix([
        [2, 1, 1, 0],
        [4, 3, 3, 1],
        [8, 7, 9, 5],
        [6, 7, 9, 8],
    ])
    
    print(A.det)
    
    