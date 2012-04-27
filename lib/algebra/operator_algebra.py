from __future__ import division
from abstract_algebra import *
from helpers import *
from sympy import exp, log, cos, sin, cosh, sinh, tan, cot, \
                            acos, asin, acosh, asinh, atan, atan2, atanh, acot, sqrt, \
                            factorial, pi, I, sympify

sympyOne = sympify(1)

import qutip



class SpaceExists(AlgebraError):
    pass


class HilbertSpace(frozenset):
    """
    A HilbertSpace object is a set of sid's, 
    i.e. integers that denote the index of a local factor space.
    An operator associated with a HilbertSpace set that is empty is
    assumed to act trivially on all spaces.
    """
    
    _local_spaces_count = 0
    _local_spaces = {}
    _local_spaces_identifiers = []
    __slots__ = []
    
    def __new__(cls, iterable):
        iterable = list(iterable)
        for i, k in enumerate(iterable):
            try:
                descriptor, sid, states = cls.retrieve(k)
                iterable[i] = sid
            except KeyError:
                if isinstance(k, str):
                    iterable[i] = cls.register_local_space(k, ())
                else:
                    raise Exception('Could not retrieve the space by this sid.')
        
        return frozenset.__new__(cls, iterable)
                

    @classmethod
    def register_local_space(cls, descriptor, states):
        """Return a unique sid for the space described by the descriptor string and the states sequence."""
        if descriptor in cls._local_spaces:
            raise SpaceExists('There already is a registered space with that descriptor: %s' % descriptor)
        if not isinstance(states, tuple):
            states = tuple(states)
            
        sid = cls._local_spaces_count
        cls._local_spaces[descriptor] = sid, states
        cls._local_spaces_identifiers.append(descriptor)
        cls._local_spaces_count += 1
        return sid
    
    @classmethod
    def retrieve_by_sid(cls, sid):
        """Return descriptor, states associated with the local space."""
        return cls._local_spaces_identifiers[sid], cls._local_spaces[cls._local_spaces_identifiers[sid]][1]
    
    @classmethod
    def retrieve_by_descriptor(cls, descriptor):
        """Return sid, states associated with descriptor."""
        return cls._local_spaces[descriptor]

    @classmethod
    def retrieve(cls, identifier):
        if isinstance(identifier, str):
            descriptor = identifier
            sid, states = cls._local_spaces[descriptor]
        else:
            descriptor = cls._local_spaces_identifiers[int(identifier)]
            sid, states = cls._local_spaces[descriptor]
            assert sid == int(identifier)
        return descriptor, sid, states
    
    
    @classmethod
    def rename_space(cls, identifier, new_descriptor):
        descriptor, sid, states = cls.retrieve(identifier)
        if new_descriptor in cls._local_spaces:
            raise Exception('There already exists a local space identified by the new descriptor %s' % new_descriptor)
        del cls._local_spaces[descriptor]
        cls._local_spaces[new_descriptor] = sid, states
        cls._local_spaces_identifiers[sid] = new_descriptor
    
    @classmethod
    def set_states(cls, identifier, new_states):
        descriptor, sid, _ = cls.retrieve(identifier)
        cls._local_spaces[descriptor] = sid, new_states
        
        
        
        
    def __repr__(self):
        return "HilbertSpace(%r)" % (list(self))
    
    def __str__(self):
        return '{%s}' % (",".join((self.__class__.retrieve_by_sid(s)[0] for s in self)))
    
    def tex(self):
        return "%s" % "\otimes".join("%d" % sid for sid in self)
    
    def __cmp__(self, other):
        """
        When the OperatorTimes tries to permute commuting operators, it 
        compares their respective HilbertSpace sets. The resulting operator order is determined 
        by the order of the spaces.
        
        Comparison rules:
        1) The empty Hilbertspace always compares to less than a non-empty space.
        2) If self and other have no common factor spaces their respective sid are compared as tuples.
        3) If self and other have factors in common, they cannot be ordered in any way.
        Rule 3) ensures that non-commuting operators will not be permuted.
        """
        if not isinstance(other, HilbertSpace):
            return NotImplemented
        if len(self) == 0:
            if len(other) == 0:
                return 0
            return -1
        if len(other) == 0:
            return +1
        if len(self & other) == 0:
            return cmp(tuple(self), tuple(other))
        return 0
    
    def __lt__(self, other):
        """Override the frozenset < operator."""
        return NotImplemented
        
    def __gt__(self, other):
        """Override the frozenset > operator."""        
        return NotImplemented
        
    def get_basis_states(self):
        bases = [self.__class__.retrieve_by_sid(s)[1] for s in self]
        from itertools import product as cartesian_product
        return list(cartesian_product(*bases))
        
    @property
    def dimension(self):
        bases = [self.__class__.retrieve_by_sid(s)[1] for s in self]
        return reduce(lambda a,b: a*b, [len(b) for b in bases], 1)
    
    def mathematica(self):
        return "ProductSpace[%s]" % (", ".join(self))



class Operator(Algebra):
    """
    The basic operator class, which fixes the abstract interface of operator objects 
    and where possible also defines the default behavior under operations.
    Any operator contains an associated HilbertSpace object, 
    on which it is taken to act non-trivially.
    """
    
    @property
    def space(self):
        raise NotImplementedError('Please implement the space property for the class %s' % self.__class__.__name__)
    
    @property
    def algebra(self):
        return Operator
        # 
        # zero = 0
        # one = None
    # coeff_algebra = Scalar
    
    
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
    
    def n(self):
        return self
    
    def __neg__(self):
        return (-1)*self
    
    def adjoint(self):
        return Adjoint.apply_with_rules(self)
    
    conjugate = adjoint
    
    def support_space(self):
        """
        Return minimal subset of states necessary to
        fully express any state from the 
        operators mathematical support space in.
        """
        raise NotImplementedError()
        
    def image_space(self):
        return self.adjoint().support_space()
        
    def representation_matrix(self):
        raise NotImplementedError(str(self.__class__))
        
        
    def to_qutip(self, full_space = None):
        raise NotImplementedError(str(self.__class__))
    
    def extend_to_space(self, space):
        # print space, self.space
        missing_spaces = space.difference(self.space)
        # print self, missing_spaces
        return OperatorTimes.apply_with_rules(self, *[explicit_identity_operator(s) for s in missing_spaces])
        
        
    # def map(self, rep_amplitude_dict):
    #     """
    #     Provided that the method 'map_rep' is implemented, 
    #     this method will map a sparse vector which is stored 
    #     in a dict to a resulting dict. Here the keys are sets that
    #     store the state information as a mapping local space sid => local space state.
    #     E.g the state 
    #     |psi> = (1+i) |e>|g>|3> + (1+2i) |r>|h>|0>
    #     would be represented as:
    #     
    #     psi_dict = {
    #         frozendict({0: 'e',        # atom in state 'e'
    #                     1: 'g',         # atom in state 'g'
    #                     2: 3            # cavity mode in fock state n=3
    #                     }): (1+1j),
    #                     
    #         frozendict({0: 'r',        # atom in state 'e'
    #                     1: 'h',         # atom in state 'g'
    #                     2: 0            # cavity mode in fock state n=3
    #                     }): (1+2j)
    #     }
    #     This is not a terribly efficient way to implement states, but it
    #     offers a great flexibility because the full Hilbert space need not be
    #     fixed from the outset.
    #     """
    #     ret_dict = {}
    #     for rep, amp in rep_amplitude_dict.items():
    #         for mrep, mamp in self.map_rep(rep).items():
    #             previous_val = ret_dict.get(mrep, 0)
    #             new_val = amp*mamp + previous_val
    #             if new_val != 0:
    #                 ret_dict[mrep] = new_val
    #             elif previous_val != 0:
    #                 del ret_dict[mrep]
    #     return ret_dict
    # 
    # def map_rep(self, rep):
    #     raise NotImplementedError

def zero_lil_matrix(dimension):
    from scipy.sparse import coo_matrix
    return coo_matrix((dimension, dimension), dtype = complex).tolil()

def id_lil_matrix(dimension):
    from scipy.sparse import identity
    return identity(dimension, dtype = complex, format = 'lil')



matrix_cache = {}


def representation_matrix(obj, full_space = None):
    
    if is_number(obj):
        if full_space is None:
            raise Exception("Zero Operator cannot be represented without information on space dimension.")
        if obj == 0:
            return zero_lil_matrix(full_space.dimension)
        return obj * id_lil_matrix(full_space.dimension)
    elif algebra(obj) == Operator:
        if full_space is None:
            full_space == obj.space
        else:
            assert full_space >= obj.space

                
        if isinstance(obj, ScalarTimesOperator):
            coeff = obj.coeff
            obj = obj.term
        else:
            coeff = 1
            
        if isinstance(obj, OperatorTimes):
            if (obj, full_space) in matrix_cache:
                if coeff == 1:
                    return matrix_cache[obj, full_space]
                return coeff * matrix_cache[obj, full_space]
            else:
                rep_mat = obj.extend_to_space(full_space).representation_matrix()
                matrix_cache[obj, full_space] = rep_mat
                if coeff == 1:
                    return rep_mat
                return coeff * rep_mat
                
        return obj.extend_to_space(full_space).representation_matrix()
    else:
        raise Exception('Cannot calculate numerical representation matrix for %r' % obj)
    





def extend_to_space(obj, space):
    if isinstance(obj, Operator):
        return obj.extend_to_space(space)
    elif obj == 0:
        return 0
    elif algebra(obj) in (SympyBasic, Number):
        return obj * OperatorTimes.apply_with_rules(*[explicit_identity_operator(s) for s in space])
    else:
        raise Exception("algebra(%r) = %s" % (obj, algebra(obj)))


class OperatorSymbol(Operator, Symbol):
    symbol_cache = {}
    
    __slots__ = ['_space']
    

    
    def __str__(self):
        return "%s^%s" % (self.identifier, self.space)

    def tex(self):
        return "{%s}^{[%s]}" % (identifier_to_tex(self.identifier), tex(self.space))
        
    def __repr__(self):
        return "OperatorSymbol(%s, %r)" % (self.identifier, self._space)
        
    def __init__(self, identifier, space):
        Symbol.__init__(self, identifier, space)
        self._space = space
        if not isinstance(space, HilbertSpace):
            raise Exception
            
    def to_qutip(self, full_space = None):
        raise AlgebraError("Cannot convert operator symbol to representation matrix. Substitute first.")
        
    @property
    def space(self):
        return self._space



class IdentityOperator(OperatorSymbol):
    
    def __new__(cls):
        return OperatorSymbol.__new__(cls, "id", HilbertSpace([]))
    
    def __init__(self):
        OperatorSymbol.__init__(self, "id", HilbertSpace([]))
    
    def __str__(self):
        return self._identifier

    def tex(self):
        return r"{\rm id}"

    def __repr__(self):
        return "IdentityOperator()"
    
    def __mul__(self, other):
        if isinstance(other, Operator):
            return other
        return Operator.__mul__(self, other)
    
    
    def adjoint(self):
        return self
        
    def map(self, rep_dict):
        return rep_dict
        
    def to_qutip(self, full_space):
        return qutip.tensor(*[qutip.qeye(HilbertSpace((s,)).dimension) for s in full_space])
    
    def mathematica(self):
        return "IdentityOperator"
        
    # def map_rep(self, rep):
    #     return {rep:1}
        

class LocalOperator(OperatorSymbol):
    """
    A local operator only acts non-trivially on a single factor space.
    """
    __slots__ = ['_identifier', '_local_space']
    
    def __new__(cls, descriptor, local_space):
        if isinstance(local_space, HilbertSpace):
            if len(local_space) == 1:
                local_space = list(local_space).pop()
            else:
                print local_space
                raise Exception()
        return OperatorSymbol.__new__(cls, "%s_[%s]" % (descriptor, local_space), HilbertSpace((local_space,)))
    
    def __init__(self, descriptor, local_space):
        if isinstance(local_space, HilbertSpace):
            if len(local_space) == 1:
                local_space = list(local_space).pop()
            else:
                raise Exception()
        OperatorSymbol.__init__(self, "%s_[%s]" % (descriptor, local_space), HilbertSpace((local_space,)))
        self._local_space = local_space
        # self._identifier = identifier

    @property
    def local_space(self):
        return self._local_space
    
    # @property
    # def space(self):
    #     return HilbertSpace([self._local_space])
    
    # @property
    # def identifier(self):
    #     return self._identifier
    
    # def map_rep(self, rep):
    #     local_rep = rep.get(self.local_space, False)
    #     if local_rep is False:
    #         return frozendict(rep = 1)
    #     return_reps = {}
    #     for mapped_local_rep, amplitude in self.map_local_rep(local_rep).items():
    #         if amplitude == 0:
    #             continue
    #         changed_rep = dict(rep)
    #         changed_rep[self.local_space] = mapped_local_rep
    #         return_reps[frozendict(changed_rep)] = amplitude
    #     return return_reps
        
    def get_local_basis(self):
        descriptor, states = HilbertSpace.retrieve_by_sid(self._local_space)
        return states
        
        
    def map_local_rep(self, local_rep):
        pass
    
    # def substitute(self, var_map):
    #     return self
        
from scipy.sparse import coo_matrix, kron
from numpy import array as np_array, sqrt as np_sqrt

class LocalProjector(LocalOperator):
    __slots__ = ['_reps']
    
    def __new__(cls, local_space, reps):
        return LocalOperator.__new__(cls, 'Pi_{%s}' % (",".join(map(str, reps))), local_space)
    
    def __init__(self, local_space, reps):
        LocalOperator.__init__(self, 'Pi_{%s}' % (",".join(map(str, reps))), local_space)
        self._reps = set(reps)
    
    @property
    def reps(self):
        return self._reps
        
    def adjoint(self):
        return self
    
    def representation_matrix(self):
        states = self.get_local_basis()
        indices = sorted([states.index(r) for r in self._reps])
        index_array = np_array([indices, indices])
        coo_m = coo_matrix((np_array([1+0j]*len(indices)), index_array), shape = (len(states), len(states)), dtype = complex)
        return coo_m.tolil()
        
    def to_qutip(self, full_space = None):
        if full_space == None or full_space == self.space:
            states = self.get_local_basis()
            indices = sorted([states.index(r) for r in self._reps])
            return sum((qutip.fock_dm(self.space.dimension, index) for index in indices), 0) 
        else:
            return self.extend_to_space(full_space).to_qutip()
        
    
    def __str__(self):
        if len(self._reps) == 1:
            return "Pi_%s^%s" % (list(self._reps).pop(), self.space)
        return "(%s)" % (" + ".join(("Pi_%s^(%s)" % (r, self.space) for r in self._reps)))

    def __repr__(self):
        return "%s(%d, %r)" % (self.__class__.__name__, self.local_space, self.reps)
    
    def tex(self):
        return "{%s}" % (" + ".join(("\Pi_{%s}^{[%s]}" % (r, tex(self.space)) for r in self.reps)))

    def mathematica(self):
        return "(%s)" % (" + ".join(["Sigma[%s,%s]" % (self.local_space, r) for r in self.reps]))

def explicit_identity_operator(local_space):
    space_descriptor, states = HilbertSpace.retrieve_by_sid(local_space)
    return LocalProjector(local_space, states)

class LocalSigma(LocalOperator):
    __slots__ = ['_rep_pair']
    
    def __new__(cls, local_space, rep_lhs, rep_rhs):
        return LocalOperator.__new__(cls, 'sigma_%s%s' % (rep_lhs, rep_rhs), local_space)
    
    def __init__(self, local_space, rep_lhs, rep_rhs):
        LocalOperator.__init__(self, 'sigma_%s%s' % (rep_lhs, rep_rhs), local_space)
        self._rep_pair = rep_lhs, rep_rhs

    @property
    def rep_pair(self):
        return self._rep_pair

    def adjoint(self):
        return LocalSigma(self.local_space, *reversed(self._rep_pair))
    
    def __str__(self):
        return "sigma_%s%s^%s" % (self._rep_pair + (self.space,))
    
    def __repr__(self):
        return "%s(%r, %s, %s)" % ((self.__class__.__name__, self.local_space) + self.rep_pair)

    def tex(self):
        return "\sigma_{%s%s}^{[%s]}" % (self._rep_pair +  (tex(self.space),))
    
    def mathematica(self):
        return "Sigma[%s,%s,%s]" % ((self.local_space,) + self.rep_pair)
        
    def representation_matrix(self):
        states = self.get_local_basis()
        to_rep, from_rep = self._rep_pair
        indices = np_array([[states.index(to_rep)],[states.index(from_rep)]])
        values = np_array([1+0j])
        coo_m = coo_matrix((values, indices), shape = (len(states), len(states)), dtype = complex)
        return coo_m.tolil()
        
    def to_qutip(self, full_space = None):
        if full_space == None or full_space == self.space:
            D = self.space.dimension
            to_rep, from_rep = self._rep_pair
            states = self.get_local_basis()
            nto, nfrom = states.index(to_rep), states.index(from_rep)
            sto, sfrom = qutip.basis(self.space.dimension, nto), qutip.basis(self.space.dimension, nfrom)
            return sto * qutip.dag(sfrom)
        else:
            return self.extend_to_space(full_space).to_qutip()
    
class Destroy(LocalOperator):
    def __new__(cls, local_space):
        return LocalOperator.__new__(cls, 'a', local_space)
    
    def __init__(self, local_space):
        LocalOperator.__init__(self, 'a', local_space)
    
    # def map_local_rep(self, local_rep):
    #     if local_rep == 0:
    #         return {}
    #     return {(local_rep-1): sqrt(local_rep),}
    
    def adjoint(self):
        return Create(self.local_space)
        
    # def __mul__(self, other):
    #     if isinstance(other, Create) and other.local_space == self.local_space: #commutator relation!
    #         return other * self + 1
    #     return Operator.__mul__(self, other)
        
    def representation_matrix(self):
        states = self.get_local_basis()
        
        if list(states) != range(len(states)):
            # print states
            raise NotImplementedError('Only truncated Fock spaces are supported')
        values = np_sqrt(np_array(states[1:]))
        indices = np_array([states[:-1], states[1:]])
        coo_m = coo_matrix((values, indices), shape = (len(states), len(states)), dtype = complex)
        return coo_m.tolil()
        
    def to_qutip(self, full_space = None):
        if full_space == None or full_space == self.space:
            D = self.space.dimension
            return qutip.destroy(D)
        else:
            return self.extend_to_space(full_space).to_qutip()

        
    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.local_space)
    
    def __str__(self):
        return "a^%s" % self.space

    def tex(self):
        return "a^{[%s]}" % (tex(self.space),)
    
    def mathematica(self):
        return "Destroy[%s]" % self.local_space
            

#just for symbolic calculation, does not affect creation of representation matrices
INFINITE_FOCK_SPACES = False 

class Create(LocalOperator):
    def __new__(cls, local_space):
        return LocalOperator.__new__(cls, 'a*', local_space)

    def __init__(self, local_space):
        LocalOperator.__init__(self, 'a*', local_space)
    
    # def map_local_rep(self, local_rep):
    #     if INFINITE_FOCK_SPACES or \
    #         (local_rep + 1) <= HilbertSpace.retrieve_by_sid(self.local_space)[1][-1]:
    #         return {(local_rep + 1): sqrt(local_rep + 1),}
    #     else:
    #         return {}
    
    def adjoint(self):
        return Destroy(self.local_space)

    def representation_matrix(self):
        states = self.get_local_basis()
        if list(states) != range(len(states)):
            raise NotImplementedError('Only truncated Fock spaces are supported')
        values = np_sqrt(np_array(states[1:]))
        indices = np_array([states[1:], states[:-1]])
        coo_m = coo_matrix((values, indices), shape = (len(states), len(states)), dtype = complex)
        return coo_m.tolil()

    def to_qutip(self, full_space = None):
        if full_space == None or full_space == self.space:
            D = self.space.dimension
            return qutip.create(D)
        else:
            return self.extend_to_space(full_space).to_qutip()


    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.local_space)
    
    def __str__(self):
        return "a*^%s" % self.space
        
    def tex(self):
        return "{a^{[%s]}}^*" % (tex(self.space),)
    
    def mathematica(self):
        return "Create[%s]" % self.local_space
    
        

def Z(local_space):
    return LocalProjector(local_space, ('h')) - LocalProjector(local_space, ('g'))

def X(local_space):
    return LocalSigma(local_space, 'h', 'g') + LocalSigma(local_space, 'g', 'h')

def Y(local_space):
    return 1j * (-LocalSigma(local_space, 'h', 'g') + LocalSigma(local_space, 'g', 'h'))




def set_union(*sets):
    return reduce(lambda a, b: a | b, sets)
    
# print set_union(set((1,2,3)), set((3,4,5)), set((5,6,7)))

def space(op):
    try:
        s = op.space
        # print op, s
        assert isinstance(s, HilbertSpace)
        return s
    except AttributeError:
        return HilbertSpace(())


class OperatorOperation(Operator):
    @property
    def space(self):
        return set_union(*map(space, self._operands))

    def n(self):
        return self.__class__.apply_with_rules(*map(n, self.operands))

class OperatorPlus(OperatorOperation, Addition):
    
    @classmethod
    def simplify_binary(cls, lhs, rhs, **rules):
        lcoeff = 1
        rcoeff = 1
        if lhs == 0:
            return rhs
        if rhs == 0:
            return lhs
        if isinstance(lhs, ScalarTimesOperator):
            lcoeff = lhs.coeff
            lhs = lhs.term
        if isinstance(rhs, ScalarTimesOperator):
            rcoeff = rhs.coeff
            rhs = rhs.term
        if lhs == rhs:
            return (lcoeff + rcoeff) * rhs
        
        # if isinstance(lhs, LocalProjector) and isinstance(rhs, LocalProjector):
        #     if lhs.space == rhs.space and lcoeff == rcoeff:
        #         return lcoeff * LocalProjector(lhs.space.pop(), tuple(set(lhs.reps) | set (rhs.reps)))

        raise CannotSimplify()
    
    def representation_matrix(self):
        full_space = self.space
        # print self.space
        # ops = self.expand().operands
        # print self.operands
        lifted_operands = [extend_to_space(op, full_space) for op in self.operands]
        matrices = [op.representation_matrix() for op in lifted_operands]
        # print lifted_operands, matrices
        return sum(matrices[1:], matrices[0])
        
    def to_qutip(self, full_space = None):
        if full_space == None:
            full_space = self.space
        assert self.space <= full_space
        return sum((op.to_qutip(full_space) for op in self.operands), 0)
        
        
    
    def check_operands(self, *operands, **rules):
        if any((algebra(op) != Operator  for op in operands)):
            raise Exception()
        
    
    def map(self, rep_dict):
        return reduce()

def cmp_by_space(a, b):
    if not hasattr(a, 'space'):
        if not hasattr(b ,'space'):
            return 0
        return -1
    if not hasattr(b, 'space'):
        return +1
    if len(a.space) == len(b.space) == 0:
        return cmp(a, b)
    return cmp(a.space, b.space)
    
def factors_sorted_by_space(operands):
    return tuple(sorted(operands, cmp = cmp_by_space))
    
    
class OperatorTimes(OperatorOperation, Multiplication):
    
    @classmethod
    def check_operands(cls, *operands, **rules):
        if any((algebra(op) != Operator for op in operands)):
            raise Exception()
            
        if len(operands) < 2:
            raise Exception()
            
        id_o = IdentityOperator()
        
        if any((op == id_o for op in operands)):
            raise Exception()
        
    
    @classmethod
    def simplify_binary(cls, lhs, rhs, **rules):
        coeff = 1
        if lhs == 0 or rhs == 0:
            return 0
        if isinstance(lhs, ScalarTimesOperator):
            coeff *= lhs.coeff
            lhs = lhs.term
        if isinstance(rhs, ScalarTimesOperator):
            coeff *= rhs.coeff
            rhs = rhs.term
        if not (isinstance(lhs, Operator) and isinstance(rhs, Operator)):
            print lhs, rhs
            raise Exception()
        id_o = IdentityOperator()
        if lhs == id_o:
            return coeff * rhs
        if rhs == id_o:
            return coeff * lhs
        # print lhs, rhs
        if isinstance(lhs, LocalOperator):
            if lhs.space == rhs.space:
                if isinstance(rhs, LocalOperator):
                    if isinstance(lhs, LocalSigma):
                        if isinstance(rhs, LocalSigma):
                            if lhs.rep_pair[1] == rhs.rep_pair[0]:
                                # print 'hm'
                                if lhs.rep_pair[0] == rhs.rep_pair[1]:
                                    # print 'yay'
                                    return coeff * LocalProjector(lhs.local_space, (lhs.rep_pair[0],))
                                return coeff * LocalSigma(lhs.local_space, lhs.rep_pair[0], rhs.rep_pair[1])
                            return 0
                        elif isinstance(rhs, LocalProjector):
                            if lhs.rep_pair[1] in rhs.reps:
                                return coeff * lhs
                            return 0
                    elif isinstance(lhs, LocalProjector):
                        if isinstance(rhs, LocalSigma):
                            if rhs.rep_pair[0] in lhs.reps:
                                return coeff * rhs
                            return 0
                        elif isinstance(rhs, LocalProjector):
                            combined_reps = tuple(set(lhs.reps) & set(rhs.reps))
                            if len(combined_reps) > 0:
                                return coeff * LocalProjector(lhs.local_space, combined_reps)
                            return 0
                    elif isinstance(lhs, Destroy) and isinstance(rhs, Create):
                        return coeff * (rhs * lhs + 1)
            
                if isinstance(rhs, OperatorPlus):
                    return coeff * OperatorPlus.apply_with_rules(*[lhs*rhs_s for rhs_s in rhs.operands])
        elif isinstance(rhs, LocalOperator):
            if lhs.space == rhs.space:
                if isinstance(lhs, OperatorPlus):
                    return coeff * OperatorPlus.apply_with_rules(*[lhs_s * rhs for lhs_s in lhs.operands])
        elif lhs.space == rhs.space and len(lhs.space) == 1:
            if isinstance(lhs, OperatorPlus) and isinstance(rhs, OperatorPlus):
                return coeff * OperatorPlus.apply_with_rules(*[lhs_s*rhs_s for lhs_s in lhs.operands for rhs_s in rhs.operands])
        if coeff != 1:
            return coeff*(lhs * rhs)
        raise CannotSimplify()
        
        

        
    @classmethod
    def filter_operands(cls, operands):
        
        operands = factors_sorted_by_space(operands)
        id_op = IdentityOperator()
        return filter(lambda op: op != 1 and op != id_op, operands)
    
#    def representation_matrix(self):
#        from qsolve.qsolve import sparse_kron
#        if any(isinstance(op, OperatorSymbol) and not isinstance(op, LocalOperator) for op in self.operands):
#            raise Exception('Convert all placeholder symbols to specified operators first')
#            
#        if len(self.space) == len(self.operands) and all((len(op.space) == 1 for op in self.operands)):
#            matrices = [op.representation_matrix() for op in self.operands]
#            return reduce(lambda a, b: sparse_kron(a, b, 'lil'), matrices)
#        else:
#            
#
#            # this code might still produce infinite loops
#            if any(len(op.space) > 1 for op in self.operands):
#                return self.expand().representation_matrix()
#                
#            ops_by_space = operands_by_space(self.operands)
#            result = id_lil_matrix(1)
#            for spc in self.space:
#                hspc = HilbertSpace((spc,)) #convert from sid to actual HilbertSpace obj
#                ops = ops_by_space[hspc]
#                spc_result = id_lil_matrix(hspc.dimension)
#                assert len(ops) >= 1
#                for op in ops:
#                    spc_result = spc_result * op.representation_matrix().tolil()
#                result = sparse_kron(result, spc_result)
#            return result
            #     
            # 
            # #print lifted_operands
            # matrices = [op.representation_matrix().tolil() for op in lifted_operands]
            # return reduce(lambda a, b: a * b, matrices, 1).tolil()        

            # # this code might still produce infinite loops
            # lifted_operands = [op.extend_to_space(self.space).expand() for op in self.operands]
            # #print lifted_operands
            # matrices = [op.representation_matrix().tolil() for op in lifted_operands]
            # return reduce(lambda a, b: a * b, matrices, 1).tolil()        
            
    def to_qutip(self, full_space = None):
        if any(len(op.space) > 1 for op in self.operands):
            return self.expand().to_qutip(full_space)
        if full_space == None or full_space == self.space:
            if len(self.space) == 1: # if only operands from same space
                return product((op.to_qutip(self.space) for op in self.operands), 1)
            else: # otherwise group by space and return tensor product
                ops_by_space = operands_by_space(self.operands)
                return qutip.tensor(*[OperatorTimes(*ops_by_space[HilbertSpace((hspc,))]).to_qutip() for hspc in self.space])
        else:
            return self.extend_to_space(full_space).to_qutip()
    

def operands_by_space(operands):
    ret_dict = {}
    spaces = map(space, operands)
#    full_space = set_union(spaces)
    assert all(len(spc) == 1 for spc in spaces)
    for spc in spaces:
        space_ops = filter(lambda op: op.space == spc, operands)
        if space_ops:
            ret_dict[spc] = space_ops
    return ret_dict
        

class ScalarTimesOperator(OperatorOperation, CoefficientTermProduct):

    
    @property
    def space(self):
        return self.term.space
    
    def n(self):
        # print complex(self.coeff)
        return complex(self.coeff)*self.term
        
    def __str__(self):
        if self.coeff == -1:
            return "(-%s)" % (self.term,)
        if is_number(self.coeff) and not isinstance(self.coeff, complex) and self.coeff < 0:
            return "(%g) * %s" % (self.coeff, self.term)
        return "%s * %s" % (self.coeff, self.term)
        
    def tex(self):
        if self.term == IdentityOperator():
            return tex(self.coeff)
        if self.coeff == -1:
            return "(-%s)" % tex(self.term)
        return "(%s)  %s" % (tex(self.coeff), tex(self.term))
        
    # def substitute(self, var_map):
    #     pass
        
    @classmethod
    def apply_with_rules(cls, coeff, term):
        if coeff == 0:
            # print "0 == ", coeff
            return 0
        if term == 0:
            return 0
        if coeff == 1:
            # print "1 == ", coeff
            return term
        if isinstance(term, ScalarTimesOperator):
            return (coeff * term.coeff) * term.term
        return ScalarTimesOperator(coeff, term)
        
    def representation_matrix(self):
        if self.coeff == 1:
            return self.term.representation_matrix()
        if not is_number(self.coeff):
            try:
                coeff = complex(self.coeff)
            except ValueError:
                raise Exception('Please convert all scalar factors in %s to numbers first' % self)
        else:
            coeff = self.coeff
        return coeff * self.term.representation_matrix()
        
    def to_qutip(self, full_space = None):
        return complex(n(self.coeff)) * self.term.to_qutip(full_space)
    
    def evalf(self):
        return self.coeff * self.term.evalf()
    


scalar_as_operator = lambda n: (ScalarTimesOperator.apply_with_rules(n, IdentityOperator()))
# number_as_operator = lambda n: scalar_as_operator(number_as_scalar(n))


second_arg_as_operator = lambda fn: modify_second_arg(fn, scalar_as_operator)
            
# operator_collect_distributively = lambda cls, ops: collect_distributively(cls, ScalarTimesOperator, ops)

Operator.add_map[Operator] = OperatorPlus.apply_with_rules
# Operator.add_map[Scalar] = second_arg_as_operator(OperatorPlus.apply_with_rules)
Operator.add_map[SympyBasic] = second_arg_as_operator(OperatorPlus.apply_with_rules)
Operator.add_map[Number] = second_arg_as_operator(OperatorPlus.apply_with_rules)
# Operator.add_map[Number] = lambda op, n: op + sympify(n)


def operator_factor_out_coeffs(cls, operands):
    return factor_out_coeffs(cls, ScalarTimesOperator, operands)

def sympify_if_not_pure_complex(obj):
    if isinstance(obj, complex):
        return obj
    return sympify(obj)

Operator.mul_map[Operator] = OperatorTimes.apply_with_rules

# Operator.mul_map[Scalar] = reverse_args(ScalarTimesOperator.apply_with_rules)
Operator.mul_map[Number] = reverse_args(ScalarTimesOperator.apply_with_rules)
# print reverse_args(ScalarTimesOperator.apply_with_rules)
# Operator.mul_map[Number] = lambda op, number: op * sympify_if_not_pure_complex(number)
Operator.mul_map[SympyBasic] = reverse_args(ScalarTimesOperator.apply_with_rules)

# Operator.rmul_map[Scalar] = reverse_args(ScalarTimesOperator.apply_with_rules)
Operator.rmul_map[Number] = reverse_args(ScalarTimesOperator.apply_with_rules)
# Operator.rmul_map[Number] = lambda op, number: op * sympify_if_not_pure_complex(number)
Operator.rmul_map[SympyBasic] = reverse_args(ScalarTimesOperator.apply_with_rules)



def expand_rhs(cls, operands):
    if len(operands) == 2:
        lhs, rhs = operands
        if isinstance(rhs, Addition):
            return (rhs.__class__.operation(*tuple((cls.apply_with_rules(lhs,rhs_summand) for rhs_summand in rhs.operands))),)
            
    return operands


Operator.sub_map[Operator] = subtract
# Operator.sub_map[Scalar] = subtract
Operator.sub_map[Number] = subtract
Operator.sub_map[SympyBasic] = subtract

Operator.rsub_map[Number] = reverse_args(subtract)
# Operator.rsub_map[Scalar] = reverse_args(subtract)
Operator.rsub_map[SympyBasic] = reverse_args(subtract)

# divide_by_scalar = lambda op, s: ScalarFraction.apply_with_rules(1, s) * op
divide_by_sympy_expr = lambda op, se: (sympyOne/se) * op

# Operator.div_map[Scalar] = divide_by_scalar
# Operator.div_map[Number] = lambda op, number: op / sympify(number)
Operator.div_map[Number] = lambda op, number: op * (sympyOne/number)
Operator.div_map[SympyBasic] = divide_by_sympy_expr


def adjoint(obj):
    try:
        return obj.adjoint()
    except:
        return obj.conjugate()


class Adjoint(Operator, UnaryOperation):
    """
    """
    
    def evalf(self):
        return adjoint(evalf(self.operand))

    @classmethod
    def apply_with_rules(cls, operand):
        if is_number(operand):
            return operand.conjugate()
        if isinstance(operand, Adjoint):
            return operand.operand
        if isinstance(operand, (OperatorPlus, ScalarTimesOperator)):
            return operand.__class__(*map(adjoint, operand.operands))
        if isinstance(operand, OperatorTimes):
            return operand.__class__(*map(adjoint, reversed(operand.operands)))
        if isinstance(operand, LocalOperator):
            return operand.adjoint()
        return cls(operand)
        
    def adjoint(self):
        return self.operand

    @property
    def space(self):
        return self.operand.space
    
    def to_qutip(self, full_space = None):
        return qutip.dag(self.operand.to_qutip(full_space))
    
