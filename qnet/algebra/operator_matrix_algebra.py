from operator_algebra import *

    

class OperatorMatrix(Algebra):
    
    @property
    def shape(self):
        raise NotImplementedError('Please implement the shape property for the class %s' % self.__class__.__name__)

    @property
    def nrows(self):
        return self.shape[0]

    @property
    def ncols(self):
        return self.shape[1]

    
    @property
    def space(self):
        raise NotImplementedError('Please implement the space property for the class %s' % self.__class__.__name__)
    
    @property
    def algebra(self):
        return OperatorMatrix
        # 
        # zero = 0
        # one = None

    
    
    mul_map = {}
    rmul_map = {}
    
    add_map = {}
    radd_map = add_map #commutative add
    
    sub_map = {}
    rsub_map = {}
    
    div_map = {}
    rdiv_map = {}
    
    pow_map = {}
    rpow_map = {}
    
    lshift_map = {}
    rlshift_map = {}
    
    rshift_map = {}
    rrshift_map = {}        
    
    def conjugate(self):
        return ConjugateMatrix(self)
    
    def adjoint(self):
        return AdjointMatrix(self)
    
    def transpose(self):
        return TransposeMatrix(self)
    
#    def __getitem__(self, key):
#        if isinstance(key, tuple) and len(key) == 2 \
#            and isinstance(key[0], int) and isinstance(key[1], int):
#                return MatrixEntry(self, key)
#        return SubMatrix(self, key)
        
    def __len__(self):
        return self.nrows
    
    def inverse(self):
        raise NotImplementedError('Not Implemented for class %s' % self.__class__.__name__)

    def __neg__(self):
        return (-1)*self
            



# class OperatorZero(Singleton, Operator):
#     def evalf(self):
#         return 0
# 
#     def __repr__(self):
#         return "OperatorZero()"
#     
#     __str__ = lambda s: "0"
#     
#     tex = __str__
#     
#     
#     
#




class OperatorMatrixSymbol(OperatorMatrix, Symbol):
    symbol_cache = {}
    
    __slots__ = ['_space', '_shape']
    
    def __str__(self):
        return "%s(%dx%d)^[%s]" % (self.identifier, self.nrows, self.ncols, ", ".join(self.space))

    def tex(self):
        return "{%s}^{[%s]}" % (self.identifier, ", ".join(self.space))
        
    def __repr__(self):
        return "OperatorSymbol(%s, %r)" % (self.identifier, self._space, self._shape)
        
    def __init__(self, identifier, space, shape):
        Symbol.__init__(self, identifier, space, shape)
        self._space = space
        self._shape = shape
        
    @property
    def space(self):
        return self._space

    @property
    def shape(self):
        return self._shape
        
class IdentityMatrix(OperatorMatrixSymbol):
    __slots__ = []
    
    def __new__(cls, dim):
        return OperatorMatrixSymbol.__new__(cls, "I_%d" % dim, HilbertSpace(()), (dim, dim))
    
    def __init__(self, dim):
        OperatorMatrixSymbol.__init__(self, "I_%d" % dim, HilbertSpace(()), (dim, dim))
        
    def __repr__(self):
        return "IdentityMatrix(%d)" % self.ncols
        
    def __str__(self):
        return self._identifier
        
    def tex(self):
        return "I_{%d}" % self.ncols
        
    def substitute(self, var_map):
        return self
        
    def __eq__(self, other):
        return self is other or self.evalf() == other
    
    def evalf(self):
        return diagonal_matrix([1]*self.ncols)
    
    def adjoint(self):
        return self
    
    def conjugate(self):
        return self
        
    def transpose(self):
        return self
    
    def inverse(self):
        return self


class MatrixOperation(OperatorMatrix, Operation):
    @property
    def space(self):
        return set_union(map(space, self._operands))

def shape(m):
    return m.shape

class MatrixAddition(MatrixOperation, Addition):
    
    @classmethod
    def filter_operands(cls, operands):
        operands = filter(lambda o: o != 0, operands)
        return operands
        
    @classmethod
    def simplify_binary(cls, lhs, rhs, **rules):
        if lhs.shape != rhs.shape:
            raise Exception("%s and %s have different shapes" % (lhs, rhs))
        if lhs == 0:
            return rhs
        if rhs == 0:
            return lhs

        if isinstance(lhs, OperatorMatrixInstance) and isinstance(rhs, OperatorMatrixInstance):
            return lhs.add_matrix_instance(rhs)
            
        raise CannotSimplify()
        
        
    
    @property
    def shape(self):
        return self.operands[0].nrows, self.operands[0].ncols
    
    # @classmethod
    # def apply_with_rules(cls, *operands):
    #     operands = filter(non_zero, operands) 
    #     if len(operands) == 0:
    #         return 0
    #     if len(operands) == 1:
    #         return operands[0]
    #     
    #     if len(set(map(shape, operands))) > 1:
    #         raise Exception("Matrices not aligned")
    #            
    #         
    #     operands = expand_operands_associatively(MatrixAddition, operands)
    #     
    #     operands = collect_distributively(MatrixAddition, OpOpMatrixProduct, IdentityOperator, operands)
    #     
    #     operands = filter(non_zero, operands)
    #     
    #     if len(operands) == 0:
    #         return 0
    #     if len(operands) == 1:
    #         return operands[0]
    #         
    #     
    #     return cls(*operands)

    
    
class MatrixMultiplication(MatrixOperation, Multiplication):
    filters = []
    
    # @property
    # def check_operands(cls, operands):
        

    @property
    def shape(self):
        return self.operands[0].nrows, self.operands[-1].ncols
        
    @classmethod 
    def simplify_binary(cls, lhs, rhs, **rules):
        
        if lhs == 0:
            return 0
            
        if rhs == 0:
            return 0
        
        if lhs.ncols != rhs.nrows:
            raise Exception('%s * %s misaligned' % (lhs, rhs))
        
        if isinstance(lhs, IdentityMatrix):
            return rhs
        if isinstance(rhs, IdentityMatrix):
            return lhs
        
        if isinstance(lhs, OperatorMatrixInstance):            
            if isinstance(rhs, OperatorMatrixInstance):
                return lhs.mul_matrix_instance(rhs)
        raise CannotSimplify()
            
        
    # @classmethod
    # def apply_with_rules(cls, *operands):
    #     
    #     for lhs, rhs in izip(operands[:-1], operands[1:]):
    #         if lhs.ncols != rhs.nrows:
    #             # print lhs.ncols, rhs.nrows
    #             raise Exception('matrix multiplication mis-aligned: %r * %r' % (lhs, rhs))
    #     
    #     operands = expand_operands_associatively(MatrixMultiplication, operands)
    #     coeff = IdentityOperator()
    #     
    #     n_operands = list(operands)
    #     lhs_space = HilbertSpace(())
    #     
    #     for i, o in enumerate(operands):
    #         if isinstance(o, OpOpMatrixProduct):
    #             if len(o.term.space & lhs_space):
    #                 coeff *= o.coeff
    #                 n_operands[i] = o.term
    #                 o = o.term
    #         lhs_space = lhs_space | o.space
    # 
    #     
    #     r_operands = []
    #     r_coeff = IdentityOperator()
    #     for i, o in enumerate(n_operands[:-1]):
    #         if not isinstance(o, IdentityMatrix):
    #             if isinstance(o, OpOpMatrixProduct) and isinstance(o.term, IdentityMatrix):
    #                 r_coeff = r_coeff * o.coeff
    #             else:
    #                 if r_coeff != IdentityOperator():
    #                     r_operands.append(r_coeff * o)
    #                     r_coeff = IdentityOperator()
    #                 else:
    #                     r_operands.append(o)
    #     if r_coeff != IdentityOperator():
    #         r_operands.append(r_coeff * n_operands[-1])
    #     else:
    #         r_operands.append(n_operands[-1])
    #     
    #     
    #     if coeff != IdentityOperator():
    #         return coeff * cls(*r_operands)
    #     return cls(*r_operands)


    
# class OperatorPower(OperatorPower, Power):
#     filters = []
    

def Im(obj):
    if isinstance(obj, Operator):
        return (I/2) * (obj.adjoint() - obj)
    return (I/2) * (obj.conjugate() - obj)

def Re(obj):
    if isinstance(obj, Operator):
        return (sympyOne/2) * (obj.adjoint() + obj)
    return (sympyOne/2) * (obj.conjugate() + obj)


class OpOpMatrixProduct(MatrixOperation, CoefficientTermProduct):
    filters = []

    @classmethod
    def create(cls, coeff, term):
        if coeff == 0 or term == 0:
            return 0
            
        if coeff == 1 or coeff == IdentityOperator():
            return term
        
        if isinstance(term, OperatorMatrixInstance):
            return OperatorMatrixInstance(coeff * term.array)
            
        if isinstance(term, OpOpMatrixProduct):
            return OpOpMatrixProduct.create(coeff * term.coeff, term.term)
        
        return OpOpMatrixProduct(coeff, term)


    @property
    def shape(self):
        return self.term.nrows, self.term.ncols

        
    def __str__(self):

        return "%s * %s" % (self.coeff, self.term)
        
    def tex(self):

        return "%s * %s" % (tex(self.coeff), tex(self.term))
        

# class OpMatrixOpProduct(MatrixOperation, TermCoefficientProduct):
#     filters = []
#     
#     
#     @classmethod
#     def apply_with_rules(cls, term, coeff):
#         if len(term.space & coeff.space) == 0:
#             return OpOpMatrixProduct.apply_with_rules(coeff, term)
#         if isinstance(term, OpOpMatrixProduct):
#             if len(term.term.space & operator.space) == 0:
#                 return OpOpMatrixProduct.apply_with_rules(term.coeff*coeff, term.term)
#             return OpOpMatrixProduct.apply_with_rules(term.coeff, term.term * coeff)
#             
# 
#     @property
#     def shape(self):
#         return self.term.nrows, self.term.ncols
# 
#         
#     def __str__(self):
# 
#         return "%s * %s" % (self.term, self.coeff)
#         
#     def tex(self):
# 
#         return "%s %s" % (tex(self.term), tex(self.coeff))
#         
#     @classmethod
#     def handle_single_operand(cls, operand):
#         if isinstance(operand, OperatorMatrix):
#             return operand
#         raise Exception, 'Something has gone wrong here: %s(%r)' % (cls.__name__, operand)


# class MatrixEntry(Operator, Expression):
#     __slots__ = ['_operator_matrix', '_key']
#     
#     def __init__(self, operator_matrix, key):
#         self._operator_matrix = operator_matrix
#         self._key = key
#         
#     @property
#     def operator_matrix(self):
#         return self._operator_matrix
#     
#     @property
#     def key(self):
#         return self._key
#         
#     def substitute(self, var_map):
#         return MatrixEntry(self._operator_matrix.substitute(var_map), self._key)
#     
#     def evalf(self):
#         return evalf(self._operator_matrix)[self._key]
#     
#     def __str__(self):
#         return "%s[%d, %d]" % (self._operator_matrix,) + self._key
#     
#     def __repr__(self):
#         return "MatrixEntry(%r, %r)" % (self._operator_matrix, self._key)
#     
#     def tex(self):
#         return "\left( %s \\right)_{%d %d}" % (tex(self._operator_matrix),) + self._key
# 
# class SubMatrix(OperatorMatrix, Expression):
#     __slots__ = ['_operator_matrix', '_key', '_shape']
#     
#     def __init__(self, operator_matrix, key):
#         self._operator_matrix = operator_matrix
#     
#         if isinstance(key, int):
#             key = (key, slice(0, operator_matrix.ncols, 1))
#             shape = (1, operator_matrix.ncols)
#         elif isinstance(key, tuple):
#             assert len(key) == 2
#             if isinstance(key[0], int):
#                 nrows = 1
#             else:
#                 from math import ceil
#                 assert isinstance(key[0], slice)
#                 triplet = key[0].indices(operator_matrix.nrows)
#                 nrows = ceil((triplet[1]-triplet[0])/float(triplet[2]))
#             if isinstance(key[1], int):
#                 ncols = 1
#             else:
#                 from math import ceil
#                 assert isinstance(key[1], slice)
#                 triplet = key[1].indices(operator_matrix.ncols)
#                 ncols = ceil((triplet[1]-triplet[0])/float(triplet[2]))                
#             shape = nrows, ncols
#         self._shape = shape
#         self._key = key
#     
#     @property
#     def shape(self):
#         return self._shape
#     
#     @property
#     def operator_matrix(self):
#         return self._operator_matrix
#     
#     @property
#     def key(self):
#         return self._key
#     
#     def substitute(self, var_map):
#         return SubMatrix(self._operator_matrix.substitute(var_map), self._key)
#     
#     def evalf(self):
#         return evalf(self._operator_matrix)[self._key]
#     
#     def __str__(self):
#         return "%s[%s]" % (self._operator_matrix, self._key)
#     
#     def __repr__(self):
#         return "MatrixEntry(%r, %r)" % (self._operator_matrix, self._key)
#     
#     def tex(self):
#         return "\left( %s \\right)_{%s}" % (tex(self._operator_matrix), self._key)


    

# scalar_as_operator = lambda n: (ScalarOperatorProduct(n, 1))
# number_as_operator = lambda n: scalar_as_operator(number_as_scalar(n))


# second_arg_as_operator = lambda fn: modify_second_arg(fn, scalar_as_operator)
# second_arg_number_as_operator = lambda fn: modify_second_arg(fn, scalar_as_operator)
            
# matrix_operator_collect_distributively_lhs = lambda cls, ops: collect_distributively(cls, OpOpMatrixProduct, ops)
# matrix_operator_collect_distributively_rhs = lambda cls, ops: collect_distributively(cls, OpMatrixOpProduct, ops)

add_to_zero = lambda om, zero: om if (is_number(zero) and zero == 0) else NotImplemented

def add_to_non_matrix(matrix, other):
    if other == 0:
        return matrix
    if matrix == 0:
        return other
    if not isinstance(other, Operator):
        other = other*IdentityOperator()
    if matrix.ncols == matrix.nrows:
        return matrix + other * IdentityMatrix(matrix.ncols)
    raise Exception('Cannot add non-square matrix to non-matrix type')



OperatorMatrix.add_map[OperatorMatrix] = MatrixAddition.create
OperatorMatrix.add_map[Number] = add_to_non_matrix
OperatorMatrix.radd_map[Number] = add_to_non_matrix
# OperatorMatrix.add_map[Scalar] = add_to_non_matrix
# OperatorMatrix.radd_map[Scalar] = add_to_non_matrix
OperatorMatrix.add_map[Operator] = add_to_non_matrix
OperatorMatrix.radd_map[Operator] = add_to_non_matrix
OperatorMatrix.add_map[SympyBasic] = add_to_non_matrix
OperatorMatrix.radd_map[SympyBasic] = add_to_non_matrix


# 
# def check_matrix_alignment_add(cls, operands):
#     if len(operands)>1:
#         if len(set(map(lambda o: o.shape, operands))) > 1:
#             raise Exception("Matrices not aligned")
#     return operands


# MatrixAddition.filters = [check_matrix_alignment_add, \
#                                     filter_out_zeros, \
#                                     matrix_operator_collect_distributively_rhs, \
#                                     matrix_operator_collect_distributively_lhs, \
#                                     expand_operands_associatively]




    
# def factors_sorted_by_space(cls, operands):
#     return sorted(operands, cmp = cmp_operators_by_space)

def mul_by_non_matrix(matrix, other):
    if other == 0:
        return 0
    if matrix == 0:
        return other
    if not isinstance(other, Operator):
        return other * matrix
    if len(other.space & matrix.space) == 0:
        return other * matrix
    return matrix * (other * IdentityMatrix(matrix.ncols))

def rmul_by_non_matrix(matrix, other):
    if not isinstance(other, Operator):
        other = other * IdentityOperator()
    return OpOpMatrixProduct.create(other, matrix)



OperatorMatrix.mul_map[Operator] = mul_by_non_matrix
# OperatorMatrix.mul_map[Scalar] = mul_by_non_matrix
OperatorMatrix.mul_map[Number] = mul_by_non_matrix
OperatorMatrix.mul_map[SympyBasic] = mul_by_non_matrix

# from  itertools import izip
# def check_matrix_alignment_mul(cls, operands):
#     if len(operands)>1:
#         if not all((o1.ncols == o2.nrows for o1,o2 in izip(operands[:-1], operands[1:]))):
#             raise Exception("Matrices not aligned")
#     return operands
# 
# MatrixMultiplication.filters = [check_matrix_alignment_mul, zero_factor]

# def combine_right_coeffs(cls, operands):
#     coeff, term = operands
#     if isinstance(term, cls):
#         return (term.term, term.coeff * coeff)
#     return operands


OperatorMatrix.rmul_map[Operator] = rmul_by_non_matrix
# OperatorMatrix.rmul_map[Scalar] = rmul_by_non_matrix
OperatorMatrix.rmul_map[Number] = rmul_by_non_matrix
OperatorMatrix.rmul_map[SympyBasic] = rmul_by_non_matrix

OperatorMatrix.mul_map[OperatorMatrix] = MatrixMultiplication.create



# OpOpMatrixProduct.filters = [expand_rhs, zero_factor, combine_left_coeffs]
# OpMatrixOpProduct.filters = [expand_lhs, zero_factor, combine_right_coeffs]


OperatorMatrix.sub_map[OperatorMatrix] = subtract
OperatorMatrix.sub_map[Operator] = subtract
# OperatorMatrix.sub_map[Scalar] = subtract
OperatorMatrix.sub_map[Number] = subtract
OperatorMatrix.sub_map[SympyBasic] = subtract


OperatorMatrix.rsub_map[Operator] = reverse_args(subtract)
# OperatorMatrix.rsub_map[Scalar] = reverse_args(subtract)
OperatorMatrix.rsub_map[Number] = reverse_args(subtract)
OperatorMatrix.rsub_map[SympyBasic] = reverse_args(subtract)


# OperatorMatrix.div_map[Scalar] = divide_by_scalar
OperatorMatrix.div_map[Number] = lambda opmat, number: opmat / sympify(number)
OperatorMatrix.div_map[SympyBasic] = divide_by_sympy_expr

class ConjugateMatrix(OperatorMatrix, UnaryOperation):
    
    def evalf(self):
        return evalf(self.operand).conjugate()

    @property
    def space(self):
        return self.operand.space
    
    @property
    def shape(self):
        return tuple(reversed(self.operand.shape))
        
    
        
    @classmethod
    def create(cls, operand):
        if isinstance(operand, ConjugateMatrix):
            return operand.operand
            
        if isinstance(operand, TransposeMatrix):
            return AdjointMatrix(operand.operand)
            
        if isinstance(operand, AdjointMatrix):
            return TransposeMatrix(operand.operand)
            
        if isinstance(operand,  MatrixAddition):
            return operand.__class__(*map(conjugate, operand.operands))
            
        if isinstance(operand, (MatrixMultiplication, OpMatrixOpProduct, OpOpMatrixProduct)):
            return operand.__class__(*map(conjugate, reversed(operand.operands)))
            
        return cls(operand)
        
    
    def __str__(self):
        return "%s^#" % self.operand
    
    def tex(self):
        return "{%s}^\#" % self.operand
    
def transpose(obj):
    if hasattr(obj,'transpose'):
        return obj.transpose()
    return obj


class TransposeMatrix(OperatorMatrix, UnaryOperation):
    
    def evalf(self):
        return evalf(self.operand).transpose()

    @property
    def space(self):
        return self.operand.space
    
    @property
    def shape(self):
        return tuple(reversed(self.operand.shape))

    @classmethod
    def create(cls, operand):
        if isinstance(operand, TransposeMatrix):
            return operand.operand
            
        if isinstance(operand, AdjointMatrix):
            return ConjugateMatrix(operand.operand)
            
        if isinstance(operand, ConjugateMatrix):
            return AdjointMatrix(operand.operand)
            
            
        if isinstance(operand,  MatrixAddition):
            return operand.__class__(*map(transpose, operand.operands))
            
        if isinstance(operand, (MatrixMultiplication, OpMatrixOpProduct, OpOpMatrixProduct)):
            return operand.__class__(*map(transpose, reversed(operand.operands)))
             
        return cls(operand)
    
        
    def __str__(self):
        return "%s^T" % self.operand
    
    def tex(self):
        return "{%s}^T" % self.operand


class AdjointMatrix(OperatorMatrix, UnaryOperation):
    
    @property
    def space(self):
        return self.operand.space
    
    @property
    def shape(self):
        return tuple(reversed(self.operand.shape))    
    
    def evalf(self):
        return evalf(self.operand).adjoint()

    @classmethod
    def create(cls, operand):
        if isinstance(operand, AdjointMatrix):
            return operand.operand

        if isinstance(operand, TransposeMatrix):
            return ConjugateMatrix(operand.operand)

        if isinstance(operand, ConjugateMatrix):
            return TransposeMatrix(operand.operand)

            
        if isinstance(operand,  MatrixAddition):
            return operand.__class__(*map(adjoint, operand.operands))
            
        if isinstance(operand, (MatrixMultiplication, OpMatrixOpProduct, OpOpMatrixProduct)):
            return operand.__class__(*map(adjoint, reversed(operand.operands)))
             
        return cls(operand)

    
    def __str__(self):
        return "%s^+" % self.operand
    
    def tex(self):
        return "{%s}^\dagger" % self.operand

from numpy import array as np_array, ndarray, zeros as np_zeros, ones as np_ones, eye as np_eye, \
                diag as np_diag, dot as np_dot, concatenate as np_concatenate, \
                conjugate as np_conjugate, ravel as np_ravel

def zeros(shape):
    return OperatorMatrixInstance(np_zeros(shape, dtype = object))

def ones(shape):
    return OperatorMatrixInstance(np_ones(shape, dtype = object))

def diagonal_matrix(diag_elements):
    return OperatorMatrixInstance(np_diag(diag_elements))

class OperatorMatrixInstance(OperatorMatrix, Expression):
    
    slots = ['_hash', '_array']
    def __init__(self, array_arg):
        if isinstance(array_arg, ndarray):
            if len(array_arg.shape) == 1: #only allow explicitly two-dimensional arrays
                array_arg = array_arg.reshape((array_arg.shape[0], 1))
            assert len(array_arg.shape) == 2
            self._array = array_arg
        else:

            self._array = np_array(array_arg, ndmin = 2)
            # print repr(self._array)
            # self._array*=2
            # print self._array
            assert len(self._array.shape) == 2

        self._hash = None
    
    @property
    def array(self):
        return self._array
    
    @property
    def shape(self):
        return self._array.shape
        
    def evalf(self):
        return OperatorMatrixInstance([[evalf(ajk) for ajk in aj] for aj in self._array])
        

    def mul_matrix_instance(self, other):
        return OperatorMatrixInstance(np_dot(self.array, other.array))
    
    
    def add_matrix_instance(self, other):
        return OperatorMatrixInstance(self._array + other._array)
        

        
    def substitute(self, var_map):
        return OperatorMatrixInstance([[substitute(ajk, var_map) for ajk in aj] for aj in self._array])


    
    def __repr__(self):
        return "OperatorMatrixInstance( \\\n%r)" % (self._array)
    
    def __str__(self):
        return str(self.array)
    
    def mathematica(self):
        return "{{ %s }}" % "},{".join((", ".join(map(mathematica, row)) for row in self._array[:]))
    
    def tex(self):
        # print "YAY"
        return "\\begin{pmatrix} %s \\end{pmatrix}" % " \\\ \n ".join((" & ".join(map(tex, row)) for row in self._array[:]))
        
    def conjugate(self):
        return OperatorMatrixInstance(np_conjugate(self._array))
    
    def transpose(self):
        return OperatorMatrixInstance(self.array.T)
    
    def adjoint(self):
        return self.transpose().conjugate()
    
    def decompose_to_block_diagonal_form(self):
        pass #TODO
    
    def __getitem__(self, key):
        ret = self._array[key]
        if isinstance(ret, ndarray):
            return OperatorMatrixInstance(ret)
        return ret

    def __setitem__(self, key, value):
        if isinstance(value, OperatorMatrixInstance):
            if not self._array[key].shape == value.shape:
                raise Exception, "Array shapes don't match"
            self._array[key] = value._array[:,:]
        elif isinstance(value, OperatorMatrix):
            raise NotImplementedError
        self.array[key] = value
    
    def __eq__(self, other):
        return self.__class__ == other.__class__ and (self._array == other._array).all()
    
    def __hash__(self):
        if self._hash == None:
            self._hash = hash((self.__class__, self._array))
        return self._hash

    @property
    def space(self):
        return set_union(*map(space, self._array.ravel()))
        
    def trace(self):
        assert self.shape[0] == self.shape[1]
        return sum(self.array[k,k] for k in range(self.shape[0]))

def concatenate_instances(operator_matrices, axis = 0):
    return OperatorMatrixInstance(np_concatenate(map(lambda om: om.array, operator_matrices), axis = axis))




def block_diagonal(A, D):
    return block_matrix(A, 0, 0, D, handle_non_matrices = 'fill')


def block_matrix(A, B, C, D, handle_non_matrices = 'fill'):
    
    if isinstance(A, OperatorMatrix) and isinstance(D, OperatorMatrix):
        shape = A.shape[0] + D.shape[0], A.shape[1] + D.shape[1]
        rdiv, cdiv = A.shape
        
        if not isinstance(B, OperatorMatrix):
            assert is_number(B) or isinstance(B, (SympyBasic, Operator))
            if handle_non_matrices == 'fill':
                B = OperatorMatrixInstance(B*np_ones((rdiv, shape[1] - cdiv), dtype = int))
            elif handle_non_matrices == 'diag':
                B = OperatorMatrixInstance(B*np_eye(rdiv, shape[1]- cdiv, dtype = int))
                
        if not isinstance(C, OperatorMatrix):
            assert is_number(C) or isinstance(C, (SympyBasic, Operator))            
            if handle_non_matrices == 'fill':
                C = OperatorMatrixInstance(C*np_ones((shape[0] - rdiv, cdiv), dtype = int))
            elif handle_non_matrices == 'diag':
                C = OperatorMatrixInstance(C*np_eye(shape[0] - rdiv, cdiv, dtype = int))

    elif isinstance(B, OperatorMatrix) and isinstance(C, OperatorMatrix):
        shape = B.shape[0] + C.shape[0], B.shape[1] + C.shape[1]
        rdiv, cdiv = B.shape[0], C.shape[1]
        
        if not isinstance(A, OperatorMatrix):
            assert is_number(A) or isinstance(A, (SympyBasic, Operator))            
            if handle_non_matrices == 'fill':
                A = OperatorMatrixInstance(A*np_ones((rdiv, cdiv), dtype = int))
            elif handle_non_matrices == 'diag':
                A = OperatorMatrixInstance(A*np_eye(rdiv, cdiv, dtype = int))
                
        if not isinstance(D, OperatorMatrix):
            assert is_number(D) or isinstance(D, (SympyBasic, Operator))
            if handle_non_matrices == 'fill':
                D = OperatorMatrixInstance(D*np_ones((shape[0] - rdiv, shape[1] - cdiv), dtype = int))
            elif handle_non_matrices == 'diag':
                D = OperatorMatrixInstance(D*np_eye(shape[0] - rdiv, shape[1] - cdiv, dtype = int))                

    else:
        raise Exception('At least either  A and D  or  B and C  need to be actual operator matrix instances' \
                        + 'A = %s,\n B = %s,\n C = %s,\n D = %s' % (A,B,C,D))
    
    
    return concatenate((concatenate((A, B), axis = 1), concatenate((C, D), axis = 1)), axis = 0)

def concatenate(matrices, axis = 0):
    if all((isinstance(m, OperatorMatrixInstance) for m in matrices)):
        return concatenate_instances(matrices, axis)
    # if axis == 0:
    #     if not len(set((m.ncols for m in matrices))) == 1:
    #         raise Exception('Matrices not aligned')
    #     layout_matrix = [[m] for m in matrices]
    # elif axis == 1:
    #     if not len(set((m.nrows for m in matrices))) == 1:
    #         raise Exception('Matrices not aligned')
    #     layout_matrix = [[m for m in matrices]]
    # else:
    #     raise Exception('axis can only take on the values 0 and 1')
    # print matrices
    raise Exception('Cannot concatenate symbolic matrices')


# def get_matrix_layout(matrices):
#     widths = tuple((m.ncols for m in matrices[0]))
#     heights = ()
#     for row in matrices[1:]:
#         height = row[0].nrows
#         if any((m.nrows != height for m in row[1:])):
#             raise Exception('Row height not unique %r' % row)
#         if tuple((m.ncols for m in row)) != widths:
#             raise Exception('Column widths do not match head row.')
#         heights += (height,)
#     return heights, widths

# class CompositeMatrix(OperatorMatrix, Expression):
#     
#     __slots__ = ['_layout', '_matrices','_hash']
#     
#     def __init__(self, matrices):
#         self._layout = get_matrix_layout(matrices)
#         if isinstance(matrices, ndarray):
#             self._matrices = matrices
#         else:
#             self._matrices = np_array(matrices)
#         self._hash = None
#         
#     @property
#     def layout(self):
#         return self._layout
#     
#     @property
#     def matrices(self):
#         return self._matrices
#     
#     @property
#     def shape(self):
#         return sum(self._layout[0]), sum(self._layout[1])
#     
#     def __mul__(self, other):
#         if isinstance(other, CompositeMatrix):
#             if self.ncols == other.nrows:
#                 if self.layout[1] == other.layout[0]:
#                     return CompositeMatrix(np_dot(self._matrices, other._matrices))
#             else:
#                 raise Exception('Misaligned composite matrices')
#         if isinstance(other, (Scalar, Operator)) or is_number(other):
#             return CompositeMatrix(self._matrices * other)
#         return OperatorMatrix.__mul__(self, other)
#     
#     def __rmul__(self, other):
#         if isinstance(other, (Scalar, Operator)) or is_number(other):
#             return CompositeMatrix(other * self._matrices)            
#         return OperatorMatrix.__rmul__(self, other)
#     
#     def __add__(self, other):
#         if isinstance(other, CompositeMatrix) and self._layout == other._layout:
#             return CompositeMatrix(self._matrices + other._matrices)
#         return OperatorMatrix.__add__(self, other)
#             
#     
#     def adjoint(self):
#         adjoint_matrices = np_array(map(lambda m: m.adjoint(), \
#                                 self._matrices.flat)).reshape(len(self.layout[0]), len(self.layout[1])).transpose()
#         return CompositeMatrix(adjoint_matrices)
#         
#     def transpose(self):
#         transposed_matrices = np_array(map(lambda m: m.transpose(), \
#                                 self._matrices.flat)).reshape(len(self.layout[0]), len(self.layout[1])).transpose()
#         return CompositeMatrix(transposed_matrices)
#     
#         
#     def conjugate(self):
#         conjugate_matrices = np_array(map(lambda m: m.conjugate(), self._matrices.flat)).reshape(len(self.layout[0]), len(self.layout[1]))
#         return CompositeMatrix(conjugate_matrices)
#         
#     def substitute(self, var_map):
#         substituted_matrices_flat = map(lambda m: m.substitute(var_map), self._matrices.flat)
#         if all((isinstance(m, OperatorMatrixInstance) for m in substituted_matrices_flat)):
#             
#             return concatenate_instances(\
#                     (concatenate_instances(\
#                         substituted_matrices_flat[k:k+len(self.layout[1])], axis = 1\
#                         ) \
#                     for k in xrange(0,len(substituted_matrices_flat), len(self.layout[1]))))
#                     
#         return CompositeMatrix(np_array(substituted_matrices_flat).reshape(len(self.layout[0]), len(self.layout[1])))
#         
#     def evalf(self):
#         conjugate_matrices = np_array(map(evalf, self._matrices.flat)).reshape(len(self.layout[0]), len(self.layout[1]))
#         return CompositeMatrix(conjugate_matrices)
#         
#     def __str__(self):
#         return str(self._matrices)
#     
#     def __repr__(self):
#         return "CompositeMatrix(%r)" % self._matrices
#     
#     def tex(self):
# 
#         return "\\begin{pmatrix} %s \\end{pmatrix}" % " \\\ \n ".join((" & ".join(map(tex, row)) for row in self._matrices[:]))
#     
#     def __eq__(self, other):
#         return self.__class__ == other.__class__ and self._layout == other._layout and (self._matrices == other._matrices).all()
#     
#     def __hash__(self, other):
#         if self._hash is None:
#             self._hash = hash((self.__class__, self._matrices))
#         return self._hash
        
    # def is_block_diagonal(self):
    #     if self.nrows != self.ncols:
    #         return False
    #     row_blocks, col_blocks = self._layout
    #     if row_blocks != col_blocks:
    #         return False
    #     block_sizes = []
    #     currentblock_size = len(row_blocks)
    #     j = 0
    #     while(j < len(row_blocks)):
    #         for k in xrange(len(row_blocks)-1, j, -1):
    #             if 
        
        
        
        
def permutation_matrix(permutation):
    """
    Return an orthogonal permutation matrix M_sigma
    for a permutation sigma given by a tuple
    (sigma(1), sigma(2),... sigma(n)), such that
    such that M_sigma e_i = e_sigma(i), where e_k 
    is the k-th standard basis vector.
    This definition ensures a composition law:
    M_{sigma . pi} = M_sigma M_pi.
    In column form M_sigma is thus given by 
    M = [e_sigma(1), e_sigma(2), ... e_sigma(n)].
    """
    assert(check_permutation(permutation))
    n = len(permutation)
    op_matrix = zeros((n,n))
    for i,j in enumerate(permutation):
        op_matrix[j,i] = 1
    return op_matrix
    


        

        

