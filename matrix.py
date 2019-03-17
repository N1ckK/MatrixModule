'''
    Module Name:
        Matrix.py
    Description:
        See Documentation
    Author:
        Nick
'''

from functools import reduce
from math import sqrt


class Matrix:
    '''
    Description:
        Class for a (n x m) matrix
    Args:
        entries (list)  - entries of the matrix
    Functions:
        row(j)          - returns j-th row
        col(i)          - returns i-th column
        set_row(r, j)   - sets j-th row to r
        set_col(c. i)   - sets i-th column to c
        empty(m,n)      - sets to (m x n) zero matrix
        entries         - getter
        m               - vertical dimension of the matrix
        n               - horizontal dimension of the matrix
        __getitem__     -
        __setitem__     -
        __iter__        -
        __next__        -
        __eq__          -
        __bool__        -
        __str__         - prints the matrix to the console
    '''
    def __init__(self, entries: list):
        self.__entries = entries
        self.__m = len(self.__entries)
        self.__n = len(self.__entries[0]) if entries else 0

    def row(self, j: int) -> list:
        return self.__entries[j]

    def col(self, i: int) -> list:
        return [j[i] for j in self.__entries]

    def set_row(self, r: list, j: int):
        assert len(r) == self.__n, "Length of list and row do not match."
        self.__entries[j] = r

    def set_col(self, c: list, i: int):
        assert len(c) == self.__m, "Length of list and column do not match."
        for j in range(len(self.__entries)):
            self.__entries[j][i] = c[j]

    def empty(self, m: int, n: int):
        values = []
        for i in range(m):
            values.append([0] * n)
        self.__init__(values)

    @property
    def entries(self):
        return self.__entries

    @property
    def m(self):
        return self.__m

    @property
    def n(self):
        return self.__n

    def __getitem__(self, key):
        if isinstance(key, tuple) and key[0] != -1:
            i, j = key
            return self.entries[i][j]
        elif isinstance(key, tuple) and key[0] == -1:
            return self.row(key[1])
        elif isinstance(key, int):
            return self.col(key)
        else:
            raise TypeError("Could not set entry of {}".format(type(self)))

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and key[0] != -1:
            i, j = key
            self.entries[i][j] = value
        elif isinstance(key, tuple) and key[0] == -1:
            self.set_row(value, key[1])
        elif isinstance(key, int):
            self.set_col(value, key)
        else:
            raise TypeError("Could not set {} entry".format(type(self)))

    def __eq__(self, other):
        if not isinstance(other, (type(self))):
            return False
        else:
            return self.entries == other.entries

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return add(self, scale(other, -1))

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return scale(self, other)
        return multiply(self, other)

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return scale(self, other)

    def __abs__(self):
        return determinant(self)

    def __bool__(self):
        return self.entries is not None

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        i = self.iter_index % self.m
        j = self.iter_index // self.m
        if i < self.m and j < self.n:
            self.iter_index += 1
            return self.entries[i][j]
        else:
            raise StopIteration

    def __str__(self):
        string = ''
        for i in range(0, self.__m):
            string += '| '
            for j in range(0, self.__n):
                string += '{0: <4}'.format(str(round(self.entries[i][j], 2))
                                           ) + ' '
            string += '|\n'
        return string


class Vector(Matrix):
    '''
    Description:
        Class for column Vectors
    Args:
        *args - Pass the entries of the vector or pass one iterable
    Examples:
        >>> Vector(1,2,3)
        | 1 |
        | 2 |
        | 3 |
        >>> Vector([1,2])
        | 1 |
        | 2 |
    '''
    def __init__(self, *args):
        args = _flatten(args)
        super().__init__([[i] for i in args])

    def empty(self, m: int, n: int = 1):
        super().empty(m, n)

    def __len__(self):
        return len(self.entries)

    def __abs__(self):
        return sqrt(sum(i[0]**2 for i in self.entries))

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return add(self, scale(other, -1))

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return scale(self, other)
        elif isinstance(other, Matrix):
            return other.__mul__(self)
        raise TypeError("You can't multiply\n{} \nand\n{}".format(other, self))

    def __getitem__(self, key):
        if isinstance(key, (float, int)):
            return self.entries[key][0]
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, (float, int)):
            self.entries[key][0] = value
        else:
            super().__setitem__(key, value)

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if len(self.entries) > self.iter_index:
            self.iter_index += 1
            return self.entries[self.iter_index - 1][0]
        else:
            raise StopIteration


def _copy(matrix):
    ''' Creates a true copy of a matrix '''
    new_entries = []
    for i in matrix.entries:
        new_entries.append(i[:])
    return type(matrix)(new_entries)


def _flatten(lst: list) -> list:
    ''' flattens a given list '''
    f_list = []
    for item in lst:
        if not isinstance(item, list):
            f_list.append(item)
        else:
            f_list += _flatten(item)
    return f_list


def test__flatten():
    '''
      Description:
          tests _flatten() with pytest
    '''
    assert _flatten([[1, 2, [3, 4], 5], [6, 7], 8]) == [1, 2, 3,
                                                        4, 5, 6, 7, 8]
    assert _flatten([1, [2, [3, [4, [5]]]]]) == [1, 2, 3, 4, 5]


def _cut(list: list, index: int) -> list:
    '''
    Description:
        cuts element from a list
    Args:
        list - list
        index - index to be cut out
    Returns:
        returns the list without the cut element
    Examples:
        >>> _cut([1,2,3], 1)
        [1,3]
    '''
    return list[:index] + list[index+1:]


def test__cut():
    '''
      Description:
          tests _cut() with pytest
    '''
    assert _cut([1, 2, 3, 4, 5], 3) == [1, 2, 3, 5]
    assert _cut([1, 2, 3, 4, 5], 0) == [2, 3, 4, 5]
    assert _cut([1, 2, 3, 4, 5], 4) == [1, 2, 3, 4]
    assert _cut([1, 2, 3, 4, 5], 10) == [1, 2, 3, 4, 5]


def _all(matrix, right=None):
    '''
    Description:
        yields all pairs (i: int, j: int) that have elements in matrix
        associated to them. If right is specified, this function yields all
        pairs (i, j) which have elements in matrix*right associated to them.
    Args:
        matrix
        right
    Yields:
        Pairs (i, j)
    '''
    for i in range(matrix.m):
        for j in range(right.n if right else matrix.n):
            yield i, j


def _extend(lst: list, length: int, fill=0):
    '''
    Description:
        Extends the given list to desired length by filling the list with
        specified element fill. If length is less or equal to length of the
        list, the list itsself will be returned.
    Args:
        lst: list
        length: int
        fill: Any object
    Returns:
        <fill in>
    Examples:
        >>> <fill in>
        <fill in>
        >>> <fill in>
        <fill in>
    '''
    return [lst[i] if i < len(lst) else fill
            for i in range(max(length, len(lst)))]


def test__extend():
    '''
      Description:
          tests _extend() with pytest
    '''
    assert _extend([1, 2, 3], 8) == [1, 2, 3] + [0] * 5


def normalized(vector):
    '''
    Description:
        Normalizes a vector to length = 1
    Args:
        vector - the vector that should be normalized
    Returns:
        the normalized vector
    Examples:
        >>> <fill in>
        <fill in>
        >>> <fill in>
        <fill in>
    '''
    length = abs(vector)
    return Vector(list(i[0] / length for i in vector.entries))


def test_normalized():
    '''
      Description:
          tests normalized() with pytest
    '''
    assert abs(normalized(Vector(1, 2, 3))) == 1
    assert normalized(Vector(1, 2, 3)) * abs(Vector(1, 2, 3)
                                             ) == Vector(1, 2, 3)


def isLinearIndependent(*vectors) -> bool:
    '''
    Description:
        Takes in vectors and determines wheather they are linear
        linear independent. If vectors are of different dimension,
        all vectors will be converted to same dimension by appending
        zeros.
    Args:
        vectors - list of vectors
    Returns:
        True - vectors are linear independent
        False - vectors are linear dependent
    '''
    max_dim = max(v.m for v in vectors)
    vectors = list(vectors)
    for i in range(len(vectors)):
        if vectors[i].m < max_dim:
            vectors[i] = Vector(_extend(list(vectors[i]), max_dim))
    mat = Matrix([])
    mat.empty(max_dim, len(vectors))
    for i, v in enumerate(vectors):
        mat.set_col(list(v), i)
    return rank(mat) == len(vectors)


def test_isLinearIndependent():
    '''
      Description:
          tests isLinearIndependent() with pytest
    '''
    assert isLinearIndependent(Vector(1, 0), Vector(0, 1)) is True
    assert isLinearIndependent(Vector(1, 0), Vector(1, 1)) is True
    assert isLinearIndependent(Vector(1, 1, 1, 1), Vector(1, 1)) is True
    assert isLinearIndependent(Vector(2, 3, 4), Vector(1, 1.5, 2)) is False
    assert isLinearIndependent(Vector(1), Vector(1)) is False
    assert isLinearIndependent(Vector(0)) is False


def zero_matrix(m: int, n: int):
    '''
    Description:
        Returns the m x n Zero Matrix
    '''
    values = []
    for i in range(m):
        values.append([0] * n)
    return Matrix(values)


def identity_matrix(n: int):
    '''
    Description:
        Returns the n x n Identity Matrix
    '''
    mat = zero_matrix(n, n)
    for i in range(n):
        mat[i, i] = 1
    return mat


def add(matrix, matrix2):
    '''
    Description:
        Adds two matricies
    Args:
        matrix - summand
        matrix - summand
    Returns:
        sum of the matricies
    Examples:
        >>> add(Matrix[[2,1],[1,2]], Matrix([[1,2],[2,1]]))
        Matrix([[3,3],[3,3]])
    '''
    assert matrix.n == matrix2.n, "mtrix is not quadratic."
    assert matrix.m == matrix2.m, "matrix2 is not quadratic."
    new_mat = _copy(matrix)
    for i, j in _all(matrix):
        new_mat[i, j] = matrix[i, j] + matrix2[i, j]
    return new_mat


def test_add():
    '''
      Description:
          tests add() with pytest
    '''
    assert add(Matrix([[1, 2, 3], [22, 11, 1], [-1, -2, -3]]),
               Matrix([[-1, -2, -3], [22, 11, 1], [1, 2, 3]])) == Matrix(
                   [[0, 0, 0], [44, 22, 2], [0, 0, 0]])
    assert add(Vector(-22, 1, 5), Vector(11, 8, 19)) == Vector(-11, 9, 24)
    assert add(Vector(2), Vector(3)) == Vector(5)
    assert add(Matrix([[2]]), Matrix([[-3]])) == Matrix([[-1]])


def multiply(matrix, matrix2):
    '''
    Description:
        Multiplies two matricies
    Args:
        matrix - factor
        matrix2 - factor
    Returns:
        product of the two matricies
    Examples:
        >>> multiply(Matrix([[1,0],[0,1]]), Matrix([[3,2],[11,2]]))
        Matrix([[3,2],[11,2]])
    '''
    assert matrix.n == matrix2.m, "matrix is not quadratic."
    new_mat = type(matrix2)([])
    new_mat.empty(matrix.m, matrix2.n)
    for i, j in _all(matrix, right=matrix2):
        new_mat[i, j] = sum(matrix[i, k] * matrix2[k, j]
                            for k in range(matrix2.m))
    return new_mat


def test_multiply():
    '''
      Description:
          tests multiply() with pytest
    '''
    assert multiply(Matrix([
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [3, 1, 2],
        [3, 1, 2],
        [1, 2, 3],
        [1, 3, 2]
    ]), Vector(0.8, 0.4, 0.2)) == Vector(2.2, 2.4, 2.6, 3.2, 3.2, 2.2, 2.4)


def power(matrix, p: int):
    '''
    Description:
        Raises the matrix to the p-th power
    Args:
        matrix - the matrix
        p - the exponent
    Returns:
        matrix multiplied by itsself p times
    Examples:
        >>> power(Matrix([[2,0],[0,4]]),2)
        Matrix([[4,0],[0,16]]
    '''
    assert matrix.n == matrix.m, "matrix is not quadratic"
    if p > 0:
        return reduce(lambda x, y: multiply(x, y),
                      [_copy(matrix) for i in range(p)])
    elif p == 0:
        return identity_matrix(matrix.n)
    else:
        return power(inverse(matrix), -p)


def test_power():
    '''
      Description:
          tests multiply() with pytest
    '''
    assert power(Matrix([[3, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]]), 2) == Matrix([[38, 40, 48],
                                                    [74, 81, 96],
                                                    [116, 126, 150]])
    assert power(Matrix([[0, 2], [1, 0]]), 0) == identity_matrix(2)
    assert power(Matrix([[0, 2], [1, 0]]), -2) == Matrix([[0.5, 0], [0, 0.5]])


def scale(matrix, s: float):
    '''
    Description:
        Scales a matrix by s
    Args:
        matrix - matrix to scale
        s - scaler
    Returns:
        scaled matrix
    Examples:
        >>> scale(Matrix([[1,2],[3,4]]),2)
        Matrix([[2,4],[6,8]])
    '''
    new_mat = _copy(matrix)
    for i, j in _all(matrix):
        new_mat[i, j] *= s
    return new_mat


def test_scale():
    '''
      Description:
          tests scale() with pytest
    '''
    assert scale(Matrix([[3, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]]), 2) == Matrix([[6, 4, 6],
                                                    [8, 10, 12],
                                                    [14, 16, 18]])
    assert scale(Matrix([[3, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]]), 0.5) == Matrix([[1.5, 1, 1.5],
                                                      [2, 2.5, 3],
                                                      [3.5, 4, 4.5]])


def trace(matrix) -> float:
    '''
    Description:
        calculates the trace of a matrix
    Args:
        matrix - the matrix
    Returns:
        returns the trace of the matrix
    Examples:
        >>> trace(Matrix([[1,2],[3,4]]))
        5
    '''
    assert matrix.n == matrix.m, "matrix is not quadratic"
    return sum(matrix[i, i] for i in range(0, matrix.n))


def test_trace():
    '''
      Description:
          tests trace() with pytest
    '''
    assert trace(Matrix([[-1, 0, -1], [8, 14, 0], [131, 12, -13]])) == 0
    assert trace(Matrix([[1]])) == 1


# def determinant(matrix) -> float:
#     '''
#     Description:
#         calculates the determinat of the (n x n) matrix using laplace
#         recursive formula.
#     Args:
#         matrix - of which determinante should be calculated
#     Returns:
#         determinate of the matrix
#     Examples:
#         >>> determinant(Matrix([[1,2,3],[4,5,6],[7,8,9]]))
#         0
#     '''
#     assert matrix.n == matrix.m, "matrix is not quadratic"
#     if matrix.n == 1:
#         return matrix[0, 0]
#     else:
#         det = 0
#         for i, entry in enumerate(matrix.col(0)):
#             det += (-1)**i * entry * determinant(minor(matrix, i, 0))
#         return det


def determinant(matrix) -> float:
    '''
    Description:
        Calculates the determinant of matrix using gaussian elimination.
    Args:
        matrix - of which determinante should be calculated
    Returns:
        determinant of the matrix
    Examples:
        >>> det(Matrix([[1,2,3],[4,5,6],[7,8,9]]))
        0
    '''
    assert matrix.n == matrix.m, "matrix is not quadratic"
    matrix = triangular(matrix)
    det = 1
    for i in range(matrix.n):
        det *= matrix[i, i]
    return round(det, 7)


def test_determinant():
    '''
      Description:
          tests determinant() with pytest
    '''
    assert determinant(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) == 0
    assert determinant(Matrix([[1]])) == 1
    assert determinant(Matrix([[1, 2, 3, 1], [3, 2, 16, 1],
                               [15, 9, 0, 0], [0, 0, 1, -1]])) == -345
    assert determinant(Matrix([[1, 6, 11, 16], [32, 73, 13, 15],
                               [-15, -99, 17, -9],
                               [44, -2, 3, 8]])) == -1406828


def minor(matrix, i: int, j: int):
    '''
    Description:
        creates a matrix with i-th column and j-th row missing
    Args:
        matrix - of which the minor should be produced
        i - column to be removed
        j - row to be removed
    Returns:
        returns a matrix object of the minor matrix
    Examples:
        >>> minor(Matrix([[1,2,3],[4,5,6],[7,8,9]]), 0, 1)
        Matrix([4, 6], [7,9])
    '''
    # assert matrix.m == matrix.n, "matrix is not quadratic"
    return Matrix([_cut(row, j) for row in _cut(matrix.entries, i)])


def test_minor():
    '''
      Description:
          tests minor() with pytest
    '''
    assert minor(Matrix([[1]]), 0, 0) == Matrix([])
    assert minor(Matrix([[1, 2], [3, 4]]), 0, 0) == Matrix([[4]])


def transpose(matrix):
    '''
    Description:
        transposes the matrix
    Args:
        matrix - to transpose
    Returns:
        transposed matrix
    Examples:
        >>> Matrix([[1, 2, 3], [4, 5, 6],[7, 8, 9]])
        Matrix([1, 4, 7], [2, 5, 8], [3, 6, 9])
    '''
    new_mat = type(matrix)([])
    new_mat.empty(matrix.n, matrix.m)
    for i, j in _all(new_mat):
        new_mat[i, j] = matrix[j, i]
    return new_mat


def test_transpose():
    '''
      Description:
          tests transpose() with pytest
    '''
    assert transpose(Matrix([[1, 2], [4, 5]])) == Matrix([[1, 4], [2, 5]])


def triangular(matrix):
    '''
    Description:
        Transforms a matrix into the unnormalized triangular form.
    Args:
        matrix - matrix that should be transoformed
    Returns:
        the transformed matrix
    Examples:
        >>> triangular(Matrix([[1,2,3],[4,-2,3]]))
        Matrix([[1, 2, 3], [0, -10, -9]])
    '''
    # Diagonal Matrix
    def D(lst: list):
        n = len(lst)
        return Matrix([[0]*(i) + [v] + [0]*(n - i - 1)
                       for i, v in enumerate(lst)])

    # Elementary Matrix
    def E(n: int, i: int, j: int, l: float):
        e_n = D([1] * n)
        e_n[i, j] = l
        return e_n

    # Switch Matrix
    def P(n: int, i: int, j: int):
        e_n = D([1] * n)
        e_n[-1, i], e_n[-1, j] = e_n[-1, j], e_n[-1, i]
        return e_n

    column = 0
    while column < min(matrix.n, matrix.m):
        if matrix[column, column] == 0:
            for i in range(column + 1, matrix.m):
                if matrix[i, column] != 0:
                    matrix = multiply(P(matrix.m, i, column), matrix)
                    break
            else:
                column += 1
                continue
        for i in range(column + 1, matrix.m):
            matrix = multiply(
                E(matrix.m, i, column, - (matrix[i, column] /
                                          matrix[column, column])), matrix)
        column += 1
    return matrix


def test_triangular():
    '''
      Description:
          tests triangular() with pytest
    '''
    assert triangular(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) == Matrix(
        [[1, 2, 3], [0, -3, -6], [0, 0, 0]]
    )


def gaussian_elimination(A, b):
    '''
    Description:
        Returns the linear system of eqations in triangular form
    '''
    rows = []
    for i in range(A.m):
        rows.append(A[-1, i] + [b[i]])
    return triangular(Matrix(rows))


def cramers_rule(A, b):
    '''
    Description:
        Applied Cramers Rule on a linear system of equations.
    Args:
        A - Matrix
        b - Vector
    Returns:
        Returns the Vector that satisfies Ax = b
    '''
    det = determinant(A)
    assert det != 0, "unambiguous"
    solution = []
    for i in range(len(b)):
        C = _copy(A)
        C[i] = list(b)
        solution.append(determinant(C) / det)
    return Vector(solution)


def inverse(matrix):
    assert matrix.n == matrix.m, "matrix is not quadratic"
    assert determinant(matrix) != 0, "determinant of matrix is zero"
    inv = _copy(matrix)
    for i, j in _all(inv):
        inv[i, j] = (1 / determinant(matrix) * (-1) ** (i + j) *
                     determinant(minor(matrix, j, i)))
    return inv


def test_inverse():
    '''
      Description:
          tests inverse() with pytest
    '''
    # assert Empty is False
    # todo
    pass


def rank(matrix) -> int:
    tr = triangular(_copy(matrix))
    return max([i+1 for i in range(0, tr.m) if any(tr[-1, i])] + [0])


def eigenvalues(matrix):
    assert matrix.m == matrix.n, "matrix is not quadratic"

    def permutations(n, n_s):
        if n == 1:
            for i in range(1, n_s + 1):
                yield [i]
        else:
            for solution in permutations(n - 1, n_s):
                for i in range(1, n_s + 1):
                    if i not in solution:
                        yield solution + [i]

    def symmetric_group(n: int):
        def createAdder(perm):
            return lambda x: perm[x]
        return [createAdder(i) for i in permutations(n, n)]

    def d(i, j):
        return True if i == j else False

    def det(lb):
        # s_n = symmetric_group(matrix.n - 1)
        pass
        # return sum(_prod([matrix[i,sigma(i)] - lb * d(i, sigma(i))
        #                   for i in range(1, matrix.m)]) for sigma in s_n)


def diagonalize(matrix):
    pass


if __name__ == '__main__':
    mat = Matrix([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    mat2 = Matrix([[-1, -2, -3],
                   [-4, -5, -6],
                   [-7, -8, -9]])
    mat3 = Matrix([[0.3, 0.4, 0.2],
                   [0.6, 0.1, 0.7],
                   [0.1, 0.5, 0.1]])
    mat4 = Matrix([
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [3, 1, 2],
        [3, 1, 2],
        [1, 2, 3],
        [1, 3, 2]
    ])
    mat5 = Matrix([
        [0, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [16, -2, 4.3]
    ])
    vec1 = Vector(5.3, -1, 3.45)
    vec2 = Vector(2, 1, 3)

    print(determinant(mat3))
    print(det_fast(mat3))
