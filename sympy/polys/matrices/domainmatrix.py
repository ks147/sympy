"""

Module for the DomainMatrix class.

A DomainMatrix represents a matrix with elements that are in a particular
Domain. Each DomainMatrix internally wraps a DDM which is used for the
lower-level operations. The idea is that the DomainMatrix class provides the
convenience routines for converting between Expr and the poly domains as well
as unifying matrices with different domains.

"""
from sympy.core.sympify import _sympify

from ..constructor import construct_domain

from .exceptions import NonSquareMatrixError, ShapeError

from .ddm import DDM

from .sdm import SDM


class DomainMatrix:

    def __init__(self, rows, shape, domain):
        if isinstance(rows, list):
            self.rep = SDM.from_list(rows, shape, domain)
        else:
            self.rep = SDM(rows, shape, domain)
        self.shape = shape
        self.domain = domain

    @classmethod
    def from_rep(cls, ddm):
        return cls(ddm, ddm.shape, ddm.domain)

    @classmethod
    def from_list_sympy(cls, nrows, ncols, rows, **kwargs):
        """Convert a list of lists of Expr into a DomainMatrix
        using construct_domain

        Parameters
        ==========

        nrows: number of rows
        ncols: number of columns
        rows: list of lists

        Returns
        =======

        DomainMatrix containing elements of rows

        Examples
        ========

        Define a DomainMatrix on the field of Integers

        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix.from_list_sympy(1, 2, [[1, 0]])
        >>> A
        DomainMatrix([[1, 0]], (1, 2), ZZ)

        See Also
        ========

        polys.constructor.construct_domain

        """
        assert len(rows) == nrows
        assert all(len(row) == ncols for row in rows)

        items_sympy = [_sympify(item) for row in rows for item in row]

        domain, items_domain = cls.get_domain(items_sympy, **kwargs)

        domain_rows = [[items_domain[ncols*r + c] for c in range(ncols)] for r in range(nrows)]

        return DomainMatrix(domain_rows, (nrows, ncols), domain)

    @classmethod
    def from_Matrix(cls, M, **kwargs):
        """Converts Matrix to DomainMatrix

        Parameters
        ==========

        M: Matrix

        Returns
        =======

        Returns DomainMatrix with identical elements as M

        Examples
        ========

        Define a DomainMatrix on the field of Rationals

        >>> from sympy import Matrix, Rational
        >>> from sympy.polys.matrices import DomainMatrix
        >>> M = Matrix([
        ... [Rational(1,2), Rational(3,4)],
        ... [2, Rational(1,5)]])
        >>> A = DomainMatrix.from_Matrix(M)
        >>> A
        DomainMatrix([[1/2, 3/4], [2, 1/5]], (2, 2), QQ)


        Define a DomainMatrix on the field of Reals.
        >>> from sympy import Matrix
        >>> from sympy.polys.matrices import DomainMatrix
        >>> M = Matrix([
        ... [1.0, 3.4],
        ... [2.4, 1]])
        >>> A = DomainMatrix.from_Matrix(M)
        >>> A
        DomainMatrix([[1.0, 3.4], [2.4, 1.0]], (2, 2), RR)

        See Also
        ========

        :py:class:~.Matrix.

        """
        return cls.from_list_sympy(*M.shape, M.tolist(), **kwargs)

    @classmethod
    def get_domain(cls, items_sympy, **kwargs):

        K, items_K = construct_domain(items_sympy, **kwargs)
        return K, items_K

    def convert_to(self, K):
        """Change domain of self

        Parameters
        ==========

        K: Domain. The domain to convert the elements to.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ, ZZ
        >>> A = DomainMatrix([[ZZ(1), ZZ(3)]], (1, 2), ZZ)
        >>> A.convert_to(QQ)
        DomainMatrix([[1, 3]], (1, 2), QQ)

        See Also
        ========

        DomainMatrix.unify, DomainMatrix.convert_to

        """
        return self.from_rep(self.rep.convert_to(K))

    def to_field(self):
        K = self.domain.get_field()
        return self.convert_to(K)

    def unify(self, other):
        """Converts self and other to a common domain

        Parameters
        ==========

        other: other DomainMatrix to be unified

        Returns
        =======

        Two Matrices that contain the elements of the input
        matrices but they now belong to a unified Domain

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import ZZ, QQ
        >>> A = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)]], (1, 3), ZZ)
        >>> B = DomainMatrix([[QQ(1, 2), QQ(3, 5)]], (1, 2), QQ)
        >>> A.unify(B)
        (DomainMatrix([[1, 2, 3]], (1, 3), QQ), DomainMatrix([[1/2, 3/5]], (1, 2), QQ))

        """
        K1 = self.domain
        K2 = other.domain
        if K1 == K2:
            return self, other
        K = K1.unify(K2)
        if K1 != K:
            self = self.convert_to(K)
        if K2 != K:
            other = other.convert_to(K)
        return self, other

    def to_Matrix(self):
        """ Convert DomainMatrix to Matrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(2), ZZ(3)]], (2, 2), ZZ)
        >>> A.to_Matrix()
        Matrix([
        [1, 2],
        [2, 3]])

        """
        from sympy.matrices.dense import MutableDenseMatrix
        elemlist = self.rep.to_list()
        rows_sympy = [[self.domain.to_sympy(e) for e in row] for row in elemlist]
        return MutableDenseMatrix(rows_sympy)

    def __repr__(self):
        elemlist = self.rep.to_list()
        rows_str = ['[%s]' % (', '.join(map(str, row))) for row in elemlist]
        rowstr = '[%s]' % ', '.join(rows_str)
        return 'DomainMatrix(%s, %r, %r)' % (rowstr, self.shape, self.domain)

    def hstack(A, B):
        """ Horizontally stack 2 Domain Matrices

        Examples
        ========

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)]], (1, 3), ZZ)
        >>> B = DomainMatrix([[QQ(-1, 2), QQ(1, 2), QQ(1, 3)]],(1, 3), QQ)

        >>> A.hstack(B)
        DomainMatrix([[1, 2, 3, -1/2, 1/2, 1/3]], (1, 6), QQ)

        """
        A, B = A.unify(B)
        return A.from_rep(A.rep.hstack(B.rep))

    def __add__(A, B):
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        return A.add(B)

    def __sub__(A, B):
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        return A.sub(B)

    def __neg__(A):
        return A.neg()

    def __mul__(A, B):
        """A * B"""
        if isinstance(B, DomainMatrix):
            return A.matmul(B)
        elif B in A.domain:
            return A.from_rep(A.rep * B)
        else:
            return NotImplemented

    def __rmul__(A, B):
        if B in A.domain:
            return A.from_rep(A.rep * B)
        else:
            return NotImplemented

    def __pow__(A, n):
        """A ** n"""
        if not isinstance(n, int):
            return NotImplemented
        return A.pow(n)

    def add(A, B):
        if A.shape != B.shape:
            raise ShapeError("shape")
        if A.domain != B.domain:
            raise ValueError("domain")
        return A.from_rep(A.rep.add(B.rep))

    def sub(A, B):
        if A.shape != B.shape:
            raise ShapeError("shape")
        if A.domain != B.domain:
            raise ValueError("domain")
        return A.from_rep(A.rep.sub(B.rep))

    def neg(A):
        return A.from_rep(A.rep.neg())

    def mul(A, b):
        return A.from_rep(A.rep.mul(b))

    def matmul(A, B):
        return A.from_rep(A.rep.matmul(B.rep))

    def pow(A, n):
        """Return Matrix A raised to the power n,
        n must be a non-negative integer

        A ** n

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import ZZ
        >>> A = DomainMatrix([[ZZ(1), ZZ(3)], [ZZ(2), ZZ(1)]], (2, 2), ZZ)
        >>> pow(A, 3)
        DomainMatrix([[19, 27], [18, 19]], (2, 2), ZZ)

        """
        if n < 0:
            raise NotImplementedError('Negative powers')
        elif n == 0:
            m, n = A.shape
            rows = [[A.domain.zero] * m for _ in range(m)]
            for i in range(m):
                rows[i][i] = A.domain.one
            return type(A)(rows, A.shape, A.domain)
        elif n == 1:
            return A
        elif n % 2 == 1:
            return A * A**(n - 1)
        else:
            sqrtAn = A ** (n // 2)
            return sqrtAn * sqrtAn

    def rref(self):
        """Return reduced row-echelon form of a DomainMatrix and indices of pivot vars.

        Raises
        ======

         ValueError('Not a field')
            if domain of self is not a field

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([
        ...     [QQ(2), QQ(-1), QQ(0)],
        ...     [QQ(-1), QQ(2), QQ(-1)],
        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)
        >>> rref_matrix, rref_pivots = A.rref()
        >>> rref_matrix
        DomainMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3), QQ)
        >>> rref_pivots
        (0, 1, 2)

        """
        if not self.domain.is_Field:
            raise ValueError('Not a field')
        rref_ddm, pivots = self.rep.rref()
        return self.from_rep(rref_ddm), tuple(pivots)

    def nullspace(self):
        """Returns the Nullspace of a DomainMatrix.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[QQ(1), QQ(-1)], [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> A.nullspace()
        DomainMatrix([[1, 1]], (1, 2), QQ)

        """
        return self.from_rep(self.rep.nullspace()[0])


    def inv(self):
        """Returns the inverse of a DomainMatrix if the Domain is a field.

        Raises
        ======

         ValueError: Not a field
            if domain of self is not a field

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([
        ...     [QQ(2), QQ(-1), QQ(0)],
        ...     [QQ(-1), QQ(2), QQ(-1)],
        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)
        >>> A.inv()
        DomainMatrix([[2/3, 1/3, 1/6], [1/3, 2/3, 1/3], [0, 0, 1/2]], (3, 3), QQ)

        """
        if not self.domain.is_Field:
            raise ValueError('Not a field')
        m, n = self.shape
        if m != n:
            raise NonSquareMatrixError
        inv = self.rep.inv()
        return self.from_rep(inv)

    def det(self):
        """Determinant of a square DomainMatrix.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([
        ...     [QQ(2), QQ(-1), QQ(0)],
        ...     [QQ(-1), QQ(2), QQ(-1)],
        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)
        >>> A.det()
        6

        """
        m, n = self.shape
        if m != n:
            raise NonSquareMatrixError
        return self.rep.det()

    def lu(self):
        """LU decomposition of self

        Returns
        =======

        (L, U, swaps): where L is a lower triangular matrix with unit diagonal, U is an upper triangular                     matrix and swaps is a list of row swap index pairs.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([
        ...     [QQ(2), QQ(-1), QQ(0)],
        ...     [QQ(-1), QQ(2), QQ(-1)],
        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)
        >>> L, U, swaps = A.lu()
        >>> L
        DomainMatrix([[1, 0, 0], [-1/2, 1, 0], [0, 0, 1]], (3, 3), QQ)
        >>> U
        DomainMatrix([[2, -1, 0], [0, 3/2, -1], [0, 0, 2]], (3, 3), QQ)
        >>> swaps
        []
        >>> L*U == A
        True

        """
        if not self.domain.is_Field:
            raise ValueError('Not a field')
        L, U, swaps = self.rep.lu()
        return self.from_rep(L), self.from_rep(U), swaps

    def lu_solve(self, rhs):
        """Solve the linear system Ax = rhs for x using LU decomposition
        of the DomainMatrix A.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([
        ...     [QQ(2), QQ(-1), QQ(0)],
        ...     [QQ(-1), QQ(2), QQ(-1)],
        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)
        >>> b = DomainMatrix([[QQ(1)],[QQ(4)],[QQ(0)]], (3, 1), QQ)
        >>> x = A.lu_solve(b)
        >>> x
        DomainMatrix([[2], [3], [0]], (3, 1), QQ)

        """
        if self.shape[0] != rhs.shape[0]:
            raise ShapeError("Shape")
        if not self.domain.is_Field:
            raise ValueError('Not a field')
        sol = self.rep.lu_solve(rhs.rep)
        return self.from_rep(sol)

    def charpoly(self):
        """Characteristic polynomial of a square matrix

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import ZZ
        >>> A = DomainMatrix([[ZZ(1), ZZ(2)],
        ... [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> A.charpoly()
        [1, -5, -2]

        """
        m, n = self.shape
        if m != n:
            raise NonSquareMatrixError("not square")
        return self.rep.charpoly()

    @classmethod
    def eye(cls, n, domain):
        """Return Identity matrix of size n x n, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.eye(3, QQ)
        DomainMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3), QQ)

        """

        return cls.from_rep(DDM.eye(n, domain))
      
    @classmethod
    def zeros(cls, shape, domain):
        """Returns a zero DomainMatrix of size shape, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.zeros((2, 3), QQ)
        DomainMatrix([[0, 0, 0], [0, 0, 0]], (2, 3), QQ)

        """

        return cls.from_rep(DDM.zeros(shape, domain))

    def __eq__(A, B):
        """A == B"""
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        return A.rep == B.rep
