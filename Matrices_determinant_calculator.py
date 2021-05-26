class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.determinant = self.det(self.matrix)

    """
    In this Object,
    A is relative
    to the matrix.
    """

    def show_matrix(self) -> None:
        """
        Print the list as
        a matrix.
        """

        for i in range(len(self.matrix)):
            length = len(self.matrix)**0.5

            if i % length == 0:
                print(f'|{self.matrix[i]:2}', end=' ')
                continue

            if (i+1) % length == 0:
                print(f'{self.matrix[i]:2}|', end=' ')
                print()
                continue

            print(f'{self.matrix[i]:2}', end=' ')

    @staticmethod
    def det_2x2(A: list) -> int:
        """
        Return the determinant
        of 2x2 matrices.
        """
        if len(A) > 4:
            return 0

        return (A[0] * A[3]) - (A[1] * A[2])

    @staticmethod
    def Sarrus_theorem(A: list) -> int:
        """
        Return the determinant
        of 3x3 matrices.
        """
        if len(A) > 9:
            return 0

        part_1 = (A[0] * A[4] * A[8]) + \
            (A[1] * A[5] * A[6]) + \
            (A[2] * A[3] * A[7])

        part_2 = (A[2] * A[4] * A[6]) + \
            (A[0] * A[5] * A[7]) + \
            (A[1] * A[3] * A[8])

        return part_1 - part_2

    @staticmethod
    def smallest_complement(A: list, length: int, index: int) -> int:
        """
        Return the smallest complement
        from a matrix element.
        """

        new_matrix = []

        if index > 0:
            """
            make sure the elements left
            behind are took.
            """
            for numbers in range(1, index+1):
                new_matrix.append(A[length+index-numbers])

            new_matrix.reverse()

        for el in range(1, length+(length-1)**2):

            if el % length == 0:
                continue

            """
            try catch to only break when the
            loop reaches the matrix length
            and guarantee the generalism of
            this program.
            """

            try:
                new_matrix.append(A[index+length+el])
            except Exception:
                break

        return Matrix.det(new_matrix)

    @staticmethod
    def cofactor(A: list, index: int, length: int) -> int:
        """
        Return the cofactor of a Matrix
        element.

        Aij = (-1)**i+j * Mij
        """
        return ((-1)**(2+index))*Matrix.smallest_complement(A, length, index)

    @staticmethod
    def Laplace_theorem(A: list, length: int, __det: int = 0) -> int:
        """
        Return the determinant
        using Laplace theorem for
        matrices nxn.

        det(M) = x11*A11 + x12*A12 + x13*A13 + ... + x1n*A1n
        """
        for index in range(length):
            __det += A[index]*Matrix.cofactor(A, index, length)

        return __det

    @staticmethod
    def det(A: list) -> int:
        """
        Return the matrix
        determinant.
        """
        size = len(A)
        length = int(size**0.5)

        if size <= 4:
            return Matrix.det_2x2(A)

        if size <= 9:
            return Matrix.Sarrus_theorem(A)

        return Matrix.Laplace_theorem(A, length)


if __name__ == '__main__':
    from random import randint

    def random_matrix(size):
        matrix = []

        for _ in range(size**2):
            matrix.append(randint(1, 100))

        return matrix

    A = random_matrix(7)

    for i in range(len(A)):
        length = len(A)**0.5

        if i % length == 0:
            print(f'{A[i]:3}', end=' ')
            continue

        if (i+1) % length == 0:
            print(f'{A[i]:3}', end=' ')
            print()
            continue

        print(f'{A[i]:3}', end=' ')

    print()

    matrix = Matrix(A)

    matrix.show_matrix()

    print()

    print(f'Det(matrix) = {matrix.determinant}')
