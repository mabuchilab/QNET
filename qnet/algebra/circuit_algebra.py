#!/usr/bin/env python

from operator_algebra import *

#import abstract_algebra
#from sympy import I

#def cdim(circuit):
#    return circuit.cdim


class CannotConvertToSLH(Exception):
    pass

class CannotConvertToABCD(Exception):
    pass


class CircuitVisualizer(object):

    def __init__(self, circuit):
        self.circuit = circuit

    def _repr_png_(self):
        import tempfile, circuit_visualization
        tmp_dir = tempfile.gettempdir()
        fname = tmp_dir + "/tmp_%s.png" % hash(str(self.circuit))

        if circuit_visualization.draw_circuit(self.circuit, fname):
            with open(fname, "rb") as png_file:
                return png_file.read()


class Circuit(object):

#    @property
#    def algebra(self):
#        return Circuit
    
    # @property
    # def space(self):
    #     raise NotImplementedError('Please implement the space property for the class %s' % self.__class__.__name__)
    
    @property
    def cdim(self):
        raise NotImplementedError(self.__class__.__name__)
    
    @property
    def block_structure(self):
        return (self.cdim,)

    def index_in_block(self, channel_index):
        if channel_index < 0 or channel_index >= self.cdim:
            raise AlgebraError()

        struct = self.block_structure

        if len(struct) == 1:
            return channel_index, 0
        i = 1
        while(sum(struct[:i]) <= channel_index and i < self.cdim):
            i +=1
        block_index = i - 1
        index_in_block = channel_index - sum(struct[:block_index])
        
        return index_in_block, block_index
        
    
    def get_blocks(self, block_structure = None):
        if block_structure is None or block_structure == self.block_structure:
            return (self, )
        raise Exception("Requested incompatible block structure %s" % (block_structure,))
    
    def series_inverse(self):
        return SeriesInverse.create(self)
    
    def feedback(self, out_index = None, in_index = None):
        if out_index is None:
            out_index = self.cdim -1
        if in_index is None:
            in_index = self.cdim -1
        return Feedback.create(self, out_index, in_index)
        
    
#    def show_in_sage(self):
#        try:
#            import warnings
#            with warnings.catch_warnings():
#                warnings.simplefilter("ignore")
#                tmp_file = "visualize.png"
#                import circuit_visualization
#                circuit_visualization.draw_circuit(self, tmp_file, direction = 'rl')
#        except ImportError, e:
#            print e
#            return
        
    def show(self):        
        return CircuitVisualizer(self)

    def reduce(self):
        return self
    
    def toSLH(self):
        raise NotImplementedError(self.__class__.__name__)

    def toABCD(self):
        raise NotImplementedError(self.__class__.__name__)
    

    def __lshift__(self, other):
        if isinstance(other, Circuit):
            return SeriesProduct.create(self, other)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Circuit):
            return Concatenation.create(self, other)
        return NotImplemented


#    mul_map = {}
#    rmul_map = {}
#
#    add_map = {}
#    radd_map = {}
#
#    sub_map = {}
#    rsub_map = {}
#
#    div_map = {}
#    rdiv_map = {}
#
#    pow_map = {}
#    rpow_map = {}
#
#    lshift_map = {}
#    rlshift_map = {}
#
#    rshift_map = {}
#    rrshift_map = {}
#
#    # def scatter(self, index):
#    #     if index < 0:
#    #         raise Exception
#    #     if index >= self.cdim:
#    #         raise Exception
#    #     # print "scattered into %s:%d" %  (self, index)
#    #     return set(range(self.cdim))
        
    
    def coherent_input(self, *input_amps):
        if len(input_amps) != self.cdim:
            raise AlgebraError()
        return self << SLH(IdentityMatrix(self.cdim), OperatorMatrix((input_amps,)).T, 0)
    


class SLH(Circuit, Operation):

    def __init__(self, S, L, H):
        if not isinstance(S, OperatorMatrix):
            S = OperatorMatrix(S)
        if not isinstance(L, OperatorMatrix):
            L = OperatorMatrix(L)
        if S.shape[0] != L.shape[0]:
            raise ValueError('S and L misaligned: S = {!r}, L = {!r}'.format(S, L))
        self._S = S
        self._L = L
        self._H = H
        self._hash = None
        super(SLH, self).__init__(S, L, H)

    @property
    def cdim(self):
        return self._S.shape[0]

    @property
    def S(self):
        return self._S

    @property
    def L(self):
        return self._L

    @property
    def H(self):
        return self._H


    @property
    def space(self):
        return self._S.space * self._L.space * space(self._H)


    
    # def __add__(self, other):
    #     if isinstance(other, SLH):
    #         return self.concatenate_triplets(other)
    #     return Circuit.__add__(self, other)
    # 
    # 
    # def __lshift__(self, other):
    #     if isinstance(other, SLH):
    #         return self.series_with_triplets(other)
    #     return Circuit.__lshift__(self, other)
    
    def series_with_slh(self, other):
        new_S = self.S * other.S
        new_L = self.S * other.L + self.L
        
        delta =  ImAdjoint(self.L.adjoint() * self.S * other.L)

        if isinstance(delta, OperatorMatrix):
            new_H = self.H + other.H + delta[0,0]
        else:
            assert delta == 0
            new_H = self.H + other.H
        
        return SLH(new_S, new_L, new_H)
    
    def concatenate_slh(self, other):
        selfS = self.S if isinstance(self.S, OperatorMatrix) else evalf(self.S)
        otherS = other.S if isinstance(other.S, OperatorMatrix) else evalf(other.S)        
        new_S = block_matrix(selfS, zeros((selfS.shape[0],otherS.shape[1])), zeros((selfS.shape[1],otherS.shape[0])), otherS)
        new_L = vstack((self.L, other.L))
        new_H = self.H + other.H
        
        return SLH(new_S, new_L, new_H)
    
    def __repr__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self._S, self._L, self._H)
    
    def __str__(self):
        return "{%s, %s, %s}" % (self._S, self._L, self._H)
    
#    def tex(self):
#        return "\left\{ %s, %s, %s \\right\}" % (tex(self._S), tex(self._L), tex(self._H))
#
#    def evalf(self):
#        return SLH(evalf(self._S), evalf(self._L), evalf(self._H))
        
    def toSLH(self):
        return self
    
#    def substitute(self, var_map):
#        return SLH(substitute(self._S, var_map), substitute(self._L, var_map), substitute(self._H, var_map))
#
#    def expand(self):
#        return self
    
#    def __hash__(self):
#        if self._hash is None:
#            self._hash = hash((self.__class__, self._S, self._L, self._H))
#        return self._hash
    
#    def __eq__(self, other):
#        if other.__class__ == self.__class__ and (other._S, other._L. other._H) == (self._S, self._L, self._H):
#            return True
#        return False
    
    def series_inverse(self):
        return SLH(self.S.adjoint(), - self.S.adjoint()*self.L, -self.H)

    def feedback(self, out_index = None, in_index = None):
        if not isinstance(self.S, OperatorMatrix) or not isinstance(self.L, OperatorMatrix):
            return Feedback(self, out_index, in_index)

        if out_index is None:
            out_index = self.cdim - 1

        if in_index == None:
            in_index = self.cdim -1
        n = self.cdim - 1

        if out_index != n:
            return (map_signals_circuit({out_index:n}, self.cdim).evalf() << self).feedback(in_index = in_index)
        elif in_index != n:
            return (self << map_signals_circuit({n:in_index}, self.cdim).evalf()).feedback()


        S, L, H = self._S, self._L, self._H

        one_minus_Snn = sympyOne - S[n,n]

        if isinstance(one_minues_Snn, Operator):
            if isinstance(one_minus_Snn, ScalarTimesOperator) and one_minus_Snn.operands[1] == IdentityOperator():
                one_minus_Snn = one_minus_Snn.coeff
            else:
                raise AlgebraError('Inversion not implemented for general operators')

        one_minus_Snn_inv = sympyOne/one_minus_Snn

        new_S = S[:n,:n] + S[0:n , n:] * one_minus_Snn_inv * S[n:, 0 : n]
        new_L = L[:n] + S[0:n, n] * one_minus_Snn_inv * L[n]
        delta_H  = Im((L.adjoint() * S[:,n:]) * one_minus_Snn_inv * L[n])

        if isinstance(delta_H, OperatorMatrix):
            delta_H = delta_H[0,0]
        new_H = H + delta_H

        return SLH(new_S, new_L, new_H)
    
    # def scatter(self, index):
    #     if index < 0:
    #         raise Exception
    #     if index >= self.cdim:
    #         raise Exception
    #     ret_indices = set([])
    #     if isinstance(self._S, OperatorMatrix):
    #         col = self._S.array[:, index]
    #         for i, op in enumerate(col):
    #             if op != 0:
    #                 ret_indices.add(i)
    #     else:
    #         ret_indices = set(range(self.cdim))
    #     return ret_indices
    
#    def get_extended_operators(self, additional_space = HilbertSpace(())):
#        space = self.space | additional_space
#        S = [[extend_to_space(Sik, space) for Sik in Si] for Si in self._S.array]
#        L = [extend_to_space(Lk, space) for Lk in self._L.array.flatten()]
#        H = extend_to_space(self._H, space)
#        return S, L, H
        
    
    def symbolic_liouvillian(self, rho = None):
        L, H = self.L, self.H
        if rho is None:
            rho = OperatorSymbol('rho', self.space)
        return -I*(H*rho - rho*H) + sum( Lk * rho * adjoint(Lk)
                             -  (adjoint(Lk)*Lk * rho + rho * adjoint(Lk)*Lk) / 2
                                                for Lk in L.array.flatten())


    def symbolic_heisenberg_eom(self, M = None, noises = None):
        L, H = self.L, self.H
        
        if M is None:
            M = OperatorSymbol('M', L.space | H.space)
            
        ret =  I*(H*M - M*H) + sum(adjoint(Lk)* M * Lk \
                    -  (adjoint(Lk)*Lk * M + M * adjoint(Lk)*Lk) / 2 \
                                                            for Lk in L.array.flatten())
        if noises is not None:
            if not isinstance(noises, OperatorMatrix):
                noises = OperatorMatrix(noises)
            LambdaT = (noises.conjugate() * noises.transpose()).transpose()
            assert noises.shape == L.shape
            S = self.S
            ret += (adjoint(noises) * S.adjoint() * (M * L - L * M)).evalf()[0,0] \
                    + ((L.adjoint() *M - M * L.adjoint()) * S * noises).evalf()[0,0]
            if len(S.space & M.space):
                comm = (S.adjoint() * M * S - M)
                ret +=  (comm * LambdaT).evalf().trace()        
        return ret


#    def __iter__(self):
#        return iter((self.S, self.L, self.H))
#
#    def __len__(self):
#        return 3
    
    def mathematica(self):
        return "SLH[%s, %s, %s]" % (mathematica(self.S), mathematica(self.L), mathematica(self.H))


#TODO ADD ABCD class and toABCD() methods
class ABCD(Circuit, Operation):
    pass
#
#    @classmethod
#    def create(cls, A, B, C, D):
#        n, m = B.shape
#        if n % 2 or m % 2:
#            raise ValueError("The matrix dimensions must be even")
#        if A.shape != (n,n) \
#            or C.shape != (m,n) \
#            or D.shape != (m, m):
#                raise ValueError("The ABCD matrix dimensions must match")
#
#        return super(ABCD, cls).create(A, B, C, D)



@check_signature
class CSymbol(Circuit, Operation):
    signature = str, int
#    __slots__ = ['_cdim', '_space']
    
#    def __init__(self, identifier, cdim):
#        Symbol.__init__(self, identifier, cdim)
#
#        self._cdim = cdim
        # self._space = space
    
#    def __repr__(self):
#        return "%s('%s', %r)" % (self.__class__.__name__, self.identifier, self.cdim)
    
    def __str__(self):
        return self.identifier
    
    # def __str__(self):
    #     return "%s(%d)" % (self.identifier, self.cdim)
    
    # @property
    # def space(self):
    #     return self._space
    @property
    def identifier(self):
        return self.operands[0]

    @property
    def cdim(self):
        return self.operands[1]


    def toABCD(self):
        raise CannotConvertToABCD()

    def toSLH(self):
        raise CannotConvertToSLH()

@singleton
class CIdentity(Circuit):

    cdim = 1

    def __repr__(self):
        return "CIdentity"
    
    def __str__(self):
        return "cid(1)"
    
#    def tex(self):
#        return "{\\rm id}_1"
    
    def __eq__(self, other):
        if not isinstance(other, Circuit):
            return NotImplemented

        if self.cdim == other.cdim:
            if self is other:
                return True
            try:
                return self.toSLH() == other.toSLH()
            except CannotConvertToSLH:
                return False
        return False

    def toSLH(self):
        return SLH(OperatorMatrix([[1]]), OperatorMatrix([[0]]), 0)

#    def evalf(self):
#        return self.toSLH()

    def series_inverse(self):
        return self
    
#    def mathematica(self):
#        return "CIdentity[1]"
    
@singleton
class CircuitZero(Circuit):
    cdim = 0

    def __repr__(self):
        return "CircuitZero"
    
    def __str__(self):
        return "cid(0)"
    
    def tex(self):
        return "{\\rm id}_0"
    
    def substitute(self, var_map):
        return self
    
    def __eq__(self, other):
        if self is other:
            return True
        if self.cdim == other.cdim:
            try:
                return self.toSLH() == other.toSLH()
            except CannotConvertToSLH:
                return False
        return False
    
    def toSLH(self):
        return SLH(OperatorMatrix([[]]), OperatorMatrix([[]]), 0)
            
#    def evalf(self):
#        return self.toSLH()
#    def series_inverse(self):
#        return self

cid_1 = CIdentity

def circuit_identity(cdim):
    # if cdim < 0:
    #     raise AlgebraError()
    if cdim <= 0:
        return CircuitZero
    if cdim == 1:
        return cid_1
    return Concatenation(*((cid_1,)*cdim))

cid = circuit_identity



def get_common_block_structure(lhs_bs, rhs_bs):
    """
    For two block structures aa = (a1, a2, ..., an), bb = (b1, b2, ..., bm)
    generate the maximal common block structure so that every block from aa and bb
    is contained in exactly one block of the resulting structure.
    This is useful for determining how to apply the distributive law when feeding
    two concatenated Circuit objects into each other.
    
    Examples: 
        (1, 1, 1), (2, 1) -> (2, 1)
        (1, 1, 2, 1), (2, 1, 2) -> (2, 3)
        
    """
    
    # for convenience the arguments may also be Circuit objects
    if isinstance(lhs_bs, Circuit):
        lhs_bs = lhs_bs.block_structure
    if isinstance(rhs_bs, Circuit):
        rhs_bs = rhs_bs.block_structure
    
    if sum(lhs_bs) != sum(rhs_bs):
        raise AlgebraError('Blockstructures have different total channel numbers.')
        
    if len(lhs_bs) == len(rhs_bs) == 0:
        return ()
    
    i = j = 1
    lsum = 0
    while(True):
        lsum = sum(lhs_bs[:i])
        rsum = sum(rhs_bs[:j])
        if(lsum < rsum):
            i +=1
        elif (rsum < lsum):
            j += 1
        else:
            break

    return (lsum, ) + get_common_block_structure(lhs_bs[i:], rhs_bs[j:])




def check_cdims_mtd(dcls, clsmtd, cls, *ops):
    if not len({o.cdim for o in ops}) == 1:
        raise ValueError("Not all operands have the same cdim:" + str(ops))
    return clsmtd(cls, *ops)
check_cdims = preprocess_create_with(check_cdims_mtd)




@flat
@filter_neutral
@check_cdims
@match_replace_binary
@filter_neutral
@check_signature_flat
class SeriesProduct(Circuit, Operation):
    signature = Circuit,

    @singleton
    class neutral_element(object):
        def __eq__(self, other):
#            print "neutral?", other
            return self is other or other == cid(other.cdim)
        def __ne__(self, other):
            return not (self == other)


#    neutral_element = GCIdentity

#    operation_symbol = ' << '
#    tex_symbol = r" \triangleleft "
    
    
#    default_rules = AssociativeOperation.default_rules.copy()
#    default_rules.update(combine_distributively = True, \
#                        combine_distributively_with_lhs_permutation = True,\
#                        combine_permutations = True, \
#                        combine_triplets = True, \
#                        factor_permutations_for_blocks = True)
    
        
    
#    @classmethod
#    def filter_operands(cls, operands):
#        if len(operands):
#
#            nc = operands[0].cdim
#            id_nc = circuit_identity(nc)
#            res =  [op for op in operands if op != id_nc]
#            if len(res) == 0:
#                return (id_nc,)
#            return res
#        return ()
#
    
    
#    @classmethod
#    @insert_default_rules(default_rules)
#    def check_operands(cls, *operands):
#
#        AssociativeOperation.check_operands.im_func(cls, *operands)
#
#        if not len(set([o.cdim for o in operands])) == 1:
#            raise Exception("All operands must have same number of channels: %s" % (operands,) )
#
#        id_n = circuit_identity(operands[0].cdim)
#
#        if any([o == id_n for o in operands]):
#            raise Exception
#
#        return
#        if rules['combine_distributively'] or rules['combine_permutations']:
#            for lhs, rhs in izip(operands[:-1], operands[1:]):
#                if rules['combine_distributively']:
#                    cbs = get_common_block_structure(lhs.block_structure, rhs.block_structure)
#                    if len(cbs) > 1:
#                        # print cbs
#                        raise Exception("%s << %s could have been simplified!" % (lhs, rhs))
#                if rules['combine_permutations']:
#                    if (isinstance(lhs, CPermutation) and isinstance(rhs, CPermutation)):
#                        raise Exception("%s << %s could have been simplified!" % (lhs, rhs))
        
    
    
    @property
    def cdim(self):
        return self.operands[0].cdim
    
    def toSLH(self):
        return reduce(lambda a, b: a.toSLH().series_with_slh(b.toSLH()), self.operands)
    
#    def evalf(self):
#        e_ops = map(evalf, self.operands)
#        return reduce(lambda a, b: a << b, e_ops)

    def reduce(self):
        return SeriesProduct.create(*[op.reduce() for op in self.operands])
    
    def series_inverse(self):
        return SeriesProduct.create(*[o.series_inverse() for o in reversed(self.operands)])

    # @classmethod
    # def apply_with_rules(cls, *operands):
    #     print "X"*100
    #     print " << ".join(str(op) for op in operands)
    #     result = AssociativeOperation.apply_with_rules.im_func(cls, *operands)
    #     print result
    #     print "X"*100
    #     return result

    binary_rules = [
    ]

#    @classmethod
##    @insert_default_rules(default_rules)
#    def simplify_binary(cls, lhs, rhs):
#
#        # if isinstance(lhs, cls) or isinstance(rhs, cls):
#        #     return cls.apply_with_rules(lhs, rhs)
#
#        if not lhs.cdim == rhs.cdim:
#            print lhs, rhs
#            raise AlgebraError('All operands must share the same number of channels')
#
#        nc = lhs.cdim
#
#        id_nc = circuit_identity(nc)
#
#        if lhs == id_nc:
#            return rhs
#
#        if rhs == id_nc:
#            return lhs
#
#        if rules['combine_permutations']:
#            if isinstance(lhs, CPermutation) and isinstance(rhs, CPermutation):
#                return lhs.series_with_permutations(rhs)
#
#        if rules['combine_triplets']:
#            if isinstance(lhs, SLH) and isinstance(rhs, SLH):
#                return lhs.series_with_slh(rhs)
#
#        if nc > 1:
#            if rules['combine_distributively']:
#                if not isinstance(rhs, CPermutation) \
#                        and (rules['combine_distributively_with_lhs_permutation'] \
#                                or not isinstance(lhs, CPermutation)):
#                    res_struct = get_common_block_structure(lhs.block_structure, rhs.block_structure)
#
#                    if len(res_struct) > 1:
#                        blocks, oblocks = lhs.get_blocks(res_struct), rhs.get_blocks(res_struct)
#                        parallel_series = [cls.create(lb, rb)  for (lb, rb) in izip(blocks, oblocks)]
#                        return Concatenation.create(*parallel_series)
#
#
#        if rules['factor_permutations_for_blocks']:
#            rbs = rhs.block_structure
#            if isinstance(lhs, CPermutation) and len(rbs) > 1:
#                residual_lhs, transformed_rhs, carried_through_lhs = lhs.factorize_for_rhs(rhs)
#                if residual_lhs == lhs:
#                    # if not transformed_rhs == rhs and carried_through_lhs == id_nc:
#                        # print "WARNING:", transformed_rhs, "!=", rhs,"and", carried_through_lhs, "!=", id_nc
#                        # raise AssertionError()
#                    raise CannotSimplify()
#
#                # print lhs, " << ", rhs, " --> ", residual_lhs, " << ", transformed_rhs, " << ", carried_through_lhs
#                # display_circuit(SeriesProduct(lhs, rhs))
#                # display_circuit(SeriesProduct(residual_lhs, transformed_rhs, carried_through_lhs))
#                return cls.create(residual_lhs, transformed_rhs, carried_through_lhs)
#
#
#        raise CannotSimplify()
#
#
#    # def scatter(self, index):
#    #     if index < 0 or index >= self.cdim:
#    #         raise Exception
#    #     first_indices = self.operands[-1].scatter(index)
#    #     if len(self.operands) == 2:
#    #         return reduce(lambda s1, s2: s1 | s2, (self.operands[0].scatter(i) for i in first_indices), set([]))
#    #     return reduce(lambda s1, s2: s1 | s2, (SeriesProduct(*self.operands[:-1]).scatter(i) for i in first_indices), set([]))

     
    
                

#Circuit.lshift_map[Circuit] = SeriesProduct.create



@flat
@filter_neutral
@match_replace_binary
@filter_neutral
@check_signature_flat
class Concatenation(Circuit, Operation):
    signature = Circuit,
    operation_symbol = " + "
#    tex_symbol = r" \oplus "
    
#    @classmethod
#    def check_operands(cls, *operands):
#        MultiaryOperation.check_operands(*operands)
#        cls.check_associatively_expanded(*operands)
#        cid_0 = CircuitZero()
#        if any((op == cid_0 for op in operands)):
#            raise Exception
#        if any((isinstance(op, CPermutation) for op in operands)) \
#                and all((isinstance(op, CIdentity) or isinstance(op, CPermutation) for op in operands)):
#            raise Exception

    neutral_element = CircuitZero
    binary_rules = []
#    @classmethod
#    def handle_zero_operands(cls):
#        return CircuitZero()
    
    
#    @classmethod
#    def filter_operands(cls, operands):
#        cid_0 = CircuitZero()
#        operands = filter(lambda op: op is not cid_0, operands)
#        if set(map(type, operands)) == set([CIdentity, CPermutation]):
#            #return permutation object instead
#            p = []
#            id_1 = circuit_identity(1)
#            cnt = 0
#            for op in operands:
#                if op is id_1:
#                    p.append(cnt)
#                    cnt += 1
#                else:
#                    p += [k + cnt for k in op.permutation]
#                    cnt += op.cdim
#            return (CPermutation(tuple(p)),)
#        return tuple(operands)
    
    
    
    
    def __str__(self):
        ops_strs = []
        id_count = 0
        for o in self.operands:
            if o == CIdentity:
                id_count += 1
            else:
                if id_count > 0:
                    ops_strs += ["cid(%d)" % id_count]
                    id_count = 0
                ops_strs += [str(o)]
        if id_count > 0:
            ops_strs += ["cid(%d)" % id_count]
        return "(%s)" % self.operation_symbol.join(ops_strs)
    
    
    @property
    def cdim(self):
        return sum((circuit.cdim for circuit in self.operands))
    
    def toSLH(self):
        return reduce(lambda a, b: a.toSLH().concatenate_slh(b.toSLH()), self.operands)
    
    
#    def evalf(self):
#        # print self.__class__.__name__
#        return reduce(lambda a, b: a + b, map(evalf, self.operands))
    
    def reduce(self):
        return Concatenation.create(*[op.reduce() for op in self.operands])
    
    @property
    def block_structure(self):
        return sum((circuit.block_structure for circuit in self.operands), ())
        
#    @trace
    def get_blocks(self, block_structure = None):
        if block_structure is None:
            return tuple(self.operands)
        
        blocks = []
        block_iter = iter(sum((op.get_blocks() for op in self.operands), ()))
        cbo = []
        current_length = 0
        for bl in block_structure:
            while(current_length < bl):
                next_op = block_iter.next()
                cbo.append(next_op)
                current_length += next_op.cdim
            if current_length != bl:
                raise Exception('requested blocks according to incompatible block_structure')
            blocks.append(Concatenation.create(*cbo))
            cbo = []
            current_length = 0
        return tuple(blocks)

    
    def series_inverse(self):
        return Concatenation.create(*[o.series_inverse() for o in self.operands])
    
    
#    @classmethod
#    def simplify_binary(cls, lhs, rhs):
#        if isinstance(lhs, cls) or isinstance(rhs, cls):
#            raise Exception
#
#        cid_0 = CircuitZero()
#
#        if lhs.cdim == 0:
#            if lhs != cid_0:
#                raise NotImplementedError
#            return rhs
#
#        if rhs.cdim == 0:
#            if rhs != cid_0:
#                raise NotImplementedError
#            return lhs
#
#        if isinstance(lhs, SLH) and isinstance(rhs, SLH):
#            return lhs.concatenate_slh(rhs)
#
#
#        if isinstance(lhs, CPermutation):
#            if isinstance(rhs, CPermutation):
#                nl = lhs.cdim
#                res_perm = lhs.permutation + tuple((p + nl for p in rhs.permutation))
#                return CPermutation(res_perm)
#            elif rhs is cid_1:
#                nl = lhs.cdim
#                return CPermutation(lhs.permutation + (nl,))
#        if lhs is cid_1:
#            if isinstance(rhs, CPermutation):
#                res_perm = (0,) + tuple((p + 1 for p in rhs.permutation))
#                return CPermutation(res_perm)
#
#        if isinstance(lhs, SeriesProduct) and isinstance(lhs.operands[-1], CPermutation):
#            if isinstance(rhs, SeriesProduct) and isinstance(rhs.operands[-1], CPermutation):
#                left_part =  cls.create(SeriesProduct.create(*lhs.operands[:-1]), SeriesProduct.create(*rhs.operands[:-1]))
#                right_part = cls.create(lhs.operands[-1], rhs.operands[-1])
#                factored_out =  SeriesProduct.create(left_part, right_part)
#                assert isinstance(factored_out, SeriesProduct)
#                return factored_out
#            else:
#                left_part =  cls.create(SeriesProduct.create(*lhs.operands[:-1]), rhs)
#                right_part = cls.create(lhs.operands[-1], cid(rhs.cdim))
#                factored_out =  SeriesProduct.create(left_part, right_part)
#                assert isinstance(factored_out, SeriesProduct)
#                return factored_out
#
#        elif isinstance(rhs, SeriesProduct) and isinstance(rhs.operands[-1], CPermutation):
#            left_part =  cls.create(lhs, SeriesProduct.create(*rhs.operands[:-1]))
#            right_part = cls.create(cid(lhs.cdim), rhs.operands[-1])
#            factored_out =  SeriesProduct.create(left_part, right_part)
#            assert isinstance(factored_out, SeriesProduct)
#            return factored_out
#
#
#        raise CannotSimplify()
      
        
    
    def feedback(self, out_index = None, in_index = None):
        if out_index == None:
            out_index = self.cdim - 1
        
        if in_index == None:
            in_index = self.cdim -1
        n = self.cdim
        
        if out_index == n -1 and in_index == n -1:
            return Concatenation.create(*(self.operands[:-1] + (self.operands[-1].feedback(),)))
                
        
        in_index_in_block, in_block = self.index_in_block(in_index)
        out_index_in_block, out_block = self.index_in_block(out_index)
        
        
        blocks = self.get_blocks()
        
        if in_block == out_block:
            
            return Concatenation.create(*blocks[:out_block]) \
                + blocks[out_block].feedback(out_index = out_index_in_block, in_index = in_index_in_block) \
                + Concatenation.create(*blocks[out_block + 1:])
        ### no 'real' feedback loop, just an effective series
        #partition all blocks into just two
        
        
        if in_block < out_block:
            b1 = Concatenation.create(*blocks[:out_block])
            b2 = Concatenation.create(*blocks[out_block:])
            
            return (b1 + circuit_identity(b2.cdim - 1))  \
                    << map_signals_circuit({out_index - 1 :in_index}, n - 1) \
                        << (circuit_identity(b1.cdim - 1) + b2)
        else:
            b1 = Concatenation.create(*blocks[:in_block])
            b2 = Concatenation.create(*blocks[in_block:])
            
            return (circuit_identity(b1.cdim - 1) + b2) \
                    << map_signals_circuit({out_index : in_index - 1}, n - 1) \
                        << (b1 + circuit_identity(b2.cdim - 1))
    
    # def scatter(self, index):
    #     if index <0 or index >= self.cdim:
    #         raise Exception
    #     index_in_block, block_index = self.index_in_block(index)
    #     # print repr(self)
    #     # print self.get_blocks()
    #     # print self.block_structure
    #     # print block_index
    #     block = self.get_blocks()[block_index]
    #     
    #     # print index, index_in_block, block
    #     delta = index - index_in_block
    #     return set([ii + delta for ii in block.scatter(index_in_block)])

            


#Concatenation.filters = [expand_operands_associatively]
#Circuit.add_map[Circuit] = Concatenation.create

class CannotFactorize(Exception):
    pass



@check_signature
class CPermutation(Circuit, Operation):
    signature = tuple,

#    @staticmethod
#    def permute(sequence, permutation):
#        if len(sequence) != len(permutation):
#            raise Exception()
#        check_permutation(permutation)
#        if type(sequence) in (list, tuple, str):
#            constr = type(sequence)
#        else:
#            constr = list
#        return constr((sequence[p] for p in permutation))

    def __init__(self, permutation):
        if not isinstance(permutation, tuple):
            permutation = tuple(permutation)
        super(CPermutation, self).__init__(permutation)

    @classmethod
    def create(cls, permutation):
        if not check_permutation(permutation):
            raise ValueError(str(permutation))
        if list(permutation) == range(len(permutation)):
            return cid(len(permutation))
        return super(CPermutation, cls).create(permutation)

    _block_perms = None

    @property
    def block_perms(self):
        if not self._block_perms:
            self._block_perms = permutation_to_block_permutations(self.permutation)
        return self._block_perms

#    def __init__(self, permutation):
#        if not isinstance(permutation, tuple):
#            permutation = tuple(permutation)
#        if not check_permutation(permutation):
#            raise Exception('Invalid permuation tuple: %r', permutation)
#        self.block_perms = permutation_to_block_permutations(permutation)
#        assert permutation == permutation_from_block_permutations(self.block_perms)
#        # print self.block_permsbl
#        self.permutation = permutation
#        super(CPermutation, self).__init__(permutation)


    @property
    def permutation(self):
        return self.operands[0]
    
    
    def toSLH(self):
        return SLH(permutation_matrix(self.permutation), zeros((self.cdim,1)), 0)
    
#    def evalf(self):
#        return self.toSLH()
    
    def __repr__(self):
        return "CPermutation(%s)" % str(self.permutation)
    
    def __str__(self):
#        if len(self.block_perms) > 2 and not abstract_algebra.CHECK_OPERANDS:
#            return str(Concatenation(*[CPermutation(bp) for bp in self.block_perms]))
        return "P_sigma%r" % (self.permutation,)
    
    @property
    def cdim(self):
        return len(self.permutation)
    
    def tex(self):
        return "P_\sigma \\begin{pmatrix} %s \\\ %s \\end{pmatrix}" % (" & ".join(map(str, range(self.cdim))), " & ".join(map(str, self.permutation)))
    
    def series_with_permutation(self, other):
        combined_permutation = tuple([self.permutation[p] for p in other.permutation])
        return CPermutation.create(combined_permutation)
    
#    def __lshift__(self, other):
#        if isinstance(other, CPermutation):
#            return self.series_with_permutation(other)
#        return Circuit.__lshift__(self, other)
    
    def series_inverse(self):
        return CPermutation(invert_permutation(self.permutation))
    
    @property
    def block_structure(self):
        return tuple(map(len, self.block_perms))
    
    def get_blocks(self, block_structure = None):
        
        if block_structure == None:
            block_structure = self.block_structure
        
        block_perms = []
        
        if block_structure == self.block_structure:
            return tuple(map(CPermutation.create, self.block_perms))
        
        if len(block_structure) > len(self.block_perms):
            raise Exception
        if sum(block_structure) != self.cdim:
            raise Exception
        current_perm = []
        block_perm_iter = iter(self.block_perms)
        for l in block_structure:
            while(len(current_perm) < l):
                offset = len(current_perm)
                current_perm += [p + offset for p in block_perm_iter.next()]
            
            if len(current_perm) != l:
                # print block_structure, self.block_perms, block_perms
                raise Exception
            
            block_perms.append(tuple(current_perm))
            current_perm = []
        return tuple(map(CPermutation.create, block_perms))
    

    
    
    @staticmethod
    def full_block_perm(block_permutation, block_structure):
        full_block_perm = []
        bp_inv = invert_permutation(block_permutation)
        for k, block_length in enumerate(block_structure):
            p_k = block_permutation[k]
            offset = sum([block_structure[bp_inv[j]] for j in range(p_k)])
            full_block_perm += range(offset, offset + block_length)
        
        assert sorted(full_block_perm) == range(sum(block_structure))
        
        return tuple(full_block_perm)
    
    @staticmethod
    def block_perm_and_perms_within_blocks(permutation, block_structure):
        nblocks = len(block_structure)
        cdim = sum(block_structure)
        
        offsets = [sum(block_structure[:k]) for k in range(nblocks)]
        images = [permutation[offset: offset + length] for (offset, length) in izip(offsets, block_structure)]
        
        images_mins = map(min, images)
        
        
        key_block_perm_inv = lambda block_index: images_mins[block_index]
        
        block_perm_inv = tuple(sorted(range(nblocks), key = key_block_perm_inv))
        # print images_mins
        # print permutation, block_structure, "-->", block_perm, invert_permutation(block_perm)
        block_perm = invert_permutation(block_perm_inv)
        
        assert images_mins[block_perm_inv[0]] == min(images_mins)
        assert images_mins[block_perm_inv[-1]] == max(images_mins)
        
        # block_perm = tuple(invert_permutation(block_perm_inv))
        
        perms_within_blocks = []
        for (offset, length, image) in izip(offsets, block_structure, images):
            block_key = lambda elt_index: image[elt_index]
            within_inv = sorted(range(length), key = block_key)
            within = invert_permutation(tuple(within_inv))
            assert permutation[within_inv[0] + offset] == min(image)
            assert permutation[within_inv[-1] + offset] == max(image)
            perms_within_blocks.append(within)
        
        return block_perm, perms_within_blocks
    
    
    def factorize_for_rhs(self, rhs):
        block_structure = rhs.block_structure
        
        block_perm, perms_within_blocks = CPermutation.block_perm_and_perms_within_blocks(self.permutation, block_structure)
        full_block_perm = CPermutation.full_block_perm(block_perm, block_structure)
        
        if not sorted(full_block_perm) == range(self.cdim):
            # print self, block_structure, full_block_perm
            raise Exception()
                            
        
        new_rhs_circuit = CPermutation.create(full_block_perm)
        within_blocks = [CPermutation.create(within_block) for within_block in perms_within_blocks]
        within_perm_circuit = sum(within_blocks, cid(0))
        rhs_blocks = rhs.get_blocks(block_structure)
        
        permuted_rhs_circuit = Concatenation.create(*[SeriesProduct.create(within_blocks[p], rhs_blocks[p]) \
                                                                for p in invert_permutation(block_perm)])
        
        new_lhs_circuit = self << within_perm_circuit.series_inverse() << new_rhs_circuit.series_inverse()
        
        # print new_lhs_circuit, permuted_rhs_circuit, new_rhs_circuit
        
        return new_lhs_circuit, permuted_rhs_circuit, new_rhs_circuit
        
    
    
    
    def feedback(self, out_index = None, in_index = None):
        if out_index == None:
            out_index = self.cdim - 1
        
        if in_index == None:
            in_index = self.cdim -1
        n = self.cdim
        
        
        new_perm_circuit = map_signals_circuit( {out_index: (n-1)}, n) << self << map_signals_circuit({(n-1):in_index}, n)
        if new_perm_circuit == circuit_identity(n):
            return circuit_identity(n-1)
        new_perm = list(new_perm_circuit.permutation)
        n_inv = new_perm.index(n-1)
        new_perm[n_inv] = new_perm[n-1]
        
        return CPermutation.create(tuple(new_perm[:-1]))
    
#    def __eq__(self, other):
#        if isinstance(other, CPermutation):
#            return self.permutation == other.permutation
#        return False
    
    # def scatter(self, index):
    #     if index < 0 or index >= self.cdim:
    #         raise Exception
    #     return set([self.permutation[index]])

    
    def factor_rhs(self, in_index):
        """
        With
        
        n           := self.cdim
        in_im       := self.permutation[in_index]
        m_{k->l}    := map_signals_circuit({k:l}, n)
        
        solve the equation (I) containing 'self'
            
            self << m_{(n-1) -> in_index} == m_{(n-1) -> in_im} << (red_self + cid(1))          (I)
        
        for the (n-1) channel CPermutation 'red_self'.
        Return in_im, red_self.
        
        This is useful when 'self' is the RHS in a SeriesProduct Object that is within a Feedback loop
        as it allows to extract the feedback channel from the permutation and moving the
        remaining part of the permutation (red_self) outside of the feedback loop.
        """
        n = self.cdim
        if not (0 <= in_index < n):
            raise Exception
        in_im = self.permutation[in_index]
        # (I) is equivalent to
        #       m_{in_im -> (n-1)} <<  self << m_{(n-1) -> in_index} == (red_self + cid(1))     (I')
        red_self_plus_cid1 = map_signals_circuit({in_im:(n-1)}, n) << self << map_signals_circuit({(n-1): in_index}, n)
        if isinstance(red_self_plus_cid1, CPermutation):
            
            #make sure we can factor
            assert red_self_plus_cid1.permutation[(n-1)] == (n-1)
            
            #form reduced permutation object
            red_self = CPermutation.create(red_self_plus_cid1.permutation[:-1])
            
            return in_im, red_self
        else:
            # 'red_self_plus_cid1' must be the identity for n channels.
            # Actually, this case can only occur
            # when self == m_{in_index ->  in_im}
            
            return in_im, circuit_identity(n-1)
    
    def factor_lhs(self, out_index):
        """
        With
        
        n           := self.cdim
        out_inv     := invert_permutation(self.permutation)[out_index]
        m_{k->l}    := map_signals_circuit({k:l}, n)
        
        solve the equation (I) containing 'self'
            
            m_{out_index -> (n-1)} << self == (red_self + cid(1)) << m_{out_inv -> (n-1)}           (I)
        
        for the (n-1) channel CPermutation 'red_self'.
        Return out_inv, red_self.
        
        This is useful when 'self' is the LHS in a SeriesProduct Object that is within a Feedback loop
        as it allows to extract the feedback channel from the permutation and moving the
        remaining part of the permutation (red_self) outside of the feedback loop.
        """
        n = self.cdim
        if not (0 <= out_index < n):
            print self, out_index
            raise Exception
        out_inv = self.permutation.index(out_index)
        
        # (I) is equivalent to
        #       m_{out_index -> (n-1)} <<  self << m_{(n-1) -> out_inv} == (red_self + cid(1))     (I')
        
        red_self_plus_cid1 = map_signals_circuit({out_index:(n-1)}, n) << self << map_signals_circuit({(n-1): out_inv}, n)
        
        if isinstance(red_self_plus_cid1, CPermutation):
            
            #make sure we can factor
            assert red_self_plus_cid1.permutation[(n-1)] == (n-1)
            
            #form reduced permutation object
            red_self = CPermutation.create(red_self_plus_cid1.permutation[:-1])
            
            return out_inv, red_self
        else:
            # 'red_self_plus_cid1' must be the identity for n channels.
            # Actually, this case can only occur
            # when self == m_{in_index ->  in_im}
            
            return out_inv, circuit_identity(n-1)
    
    def extract_rhs(self, in_index, length = 1):
        return circuit_identity(self.cdim), circuit_identity(0), self, circuit_identity(0)
    
    def extract_lhs(self, out_index):
        return circuit_identity(0), self, circuit_identity(0), circuit_identity(self.cdim)
    
#    def simplify(self):
#        return self
    
    def mathematica(self):
        return "CPermutation[%s]" % (",". join(map(str,self.permutation)))  

def P_sigma(*permutation):
    return CPermutation.create(permutation)


def extract_signal(k, cdim):
    return tuple(range(k) + [cdim-1] + range(k, cdim-1))


def extract_signal_circuit(k, cdim):
    return CPermutation.create(extract_signal(k, cdim))


def map_signals(mapping, n):
    
    # keys = mapping.keys()
    free_values = range(n)
    
    
    for v in mapping.values():
        if v >= n:
            raise Exception('the mapping cannot take on values larger than cdim - 1')
        free_values.remove(v)
    for k in mapping:
        if k >= n:
            raise Exception('the mapping cannot map keys larger than cdim - 1')
    # sorted(set(range(n)).difference(set(mapping.values())))
    permutation = []
    # print free_values, mapping, n
    for k in range(n):
        if k in mapping:
            permutation.append(mapping[k])
        else:
            permutation.append(free_values.pop(0))
    # print permutation
    return tuple(permutation)

def map_signals_circuit(mapping, n):
    # print mapping, n
    return CPermutation.create(map_signals(mapping, n))

        

def pad_with_identity(circuit, k, n):
    circuit_n = circuit.cdim
    combined_circuit = circuit + circuit_identity(n)
    permutation = range(k) + range(circuit_n, circuit_n + n) + range(k, circuit_n)
    return CPermutation.create(invert_permutation(permutation)) << combined_circuit << CPermutation.create(permutation)


#@check_signature
@match_replace
class Feedback(Circuit, Operation):
    delegate_to_method = (Concatenation, SLH, CPermutation)
    signature = Circuit, int, int
    
#    def __init__(self, operand, out_index = None, in_index = None):
#
#        out_index = out_index if not out_index is None else operand.cdim -1
#        in_index = in_index if not in_index is None else operand.cdim -1
#        MultiaryOperation.__init__(self, operand, out_index, in_index)
    rules = []
    
    @property
    def operand(self):
        return self._operands[0]
    
    @property
    def out_in_pair(self):
        return self._operands[1:]
    
    
#    @classmethod
#    def check_operands(cls, operand, out_index = None, in_index = None):
#
#
#        cdim = operand.cdim
#
#        if not isinstance(operand, Circuit):
#            raise Exception
#        if operand.cdim < 2:
#            raise Exception
#        if isinstance(operand, Concatenation):
#            raise Exception
#        if isinstance(operand, CPermutation):
#            raise Exception
#        if isinstance(operand, SLH) and isinstance(operand.S, OperatorMatrix) and isinstance(operand.L, OperatorMatrix):
#            raise Exception
#        if isinstance(operand, CIdentity):
#            raise Exception
#        # if isinstance(operand, SeriesProduct):
#        #     if isinstance(operand.operands[-1], CPermutation):
#        #         raise Exception
#        #     if isinstance(operand.operands[0], CPermutation):
#        #         raise Exception
#
    
    @property
    def cdim(self):
        return self.operand.cdim - 1
    
    @classmethod
    def create(cls, circuit, out_index, in_index):

        if not isinstance(circuit, Circuit):
            raise ValueError()

        n = circuit.cdim
        if n == 0:
            raise ValueError()

        if n == 1:
            raise ValueError()


#        if out_index is None:
#            out_index = n-1
#
#        if in_index is None:
#            in_index = n-1
#
#        if not (0 <= out_index < n) and (0 <= in_index < n):
#            raise ValueError()

        if isinstance(circuit, cls.delegate_to_method):
            return circuit.feedback(out_index, in_index)

        return super(Feedback, cls).create(circuit, out_index, in_index)




        
##
##
##        # print "out_index -> in_index", out_index, "->", in_index
##
##        if isinstance(circuit, SeriesProduct):
##
##            # Do this to try to factorize permutations both ways!
##            # Probably not very efficient, but good enough for the moment
##            circuit = circuit.series_inverse().series_inverse()
##
##
##        if isinstance(circuit, Concatenation):
##            return circuit.feedback(out_index, in_index)
##
##
##
##        # now, if circuit is STILL as series
##        if isinstance(circuit, SeriesProduct):
##
##
##
##            lhs, rhs = circuit.operands[0], circuit.operands[-1]
##
##            if isinstance(lhs, CPermutation):
##                # print "lhs is CPermutation:", lhs
##
##                out_inv , lhs_red = lhs.factor_lhs(out_index)
##
##                # print "out_index <- out_inv:", out_index, "<-", out_inv
##                # print "lhs_red:", lhs_red
##
##                return lhs_red << cls.create(SeriesProduct.create(*circuit.operands[1:]), out_inv, in_index)
##
##            elif isinstance(lhs, Concatenation):
##                # print "lhs is Concatenation", lhs
##
##                _, block_index = lhs.index_in_block(out_index)
##                # print "index_in_block, block_index", index_in_block, block_index
##
##                bs = lhs.block_structure
##                # print "block_structure", bs
##
##                nbefore, nblock, nafter = sum(bs[:block_index]), bs[block_index], sum(bs[block_index + 1:])
##                before, block, after = lhs.get_blocks((nbefore, nblock, nafter))
##                # print "before, block, after", before, block, after
##
##                if before != cid(nbefore) or after != cid(nafter):
##                    outer_lhs = before + cid(nblock - 1) + after
##                    inner_lhs = cid(nbefore) + block + cid(nafter)
##                    return outer_lhs << cls.create(SeriesProduct.create(inner_lhs, *circuit.operands[1:]), out_index, in_index)
##                elif block == cid(nblock):
##                    outer_lhs = before + cid(nblock - 1) + after
##                    return outer_lhs << cls.create(SeriesProduct.create(*circuit.operands[1:]), out_index, in_index)
##
##
##            if isinstance(rhs, CPermutation):
##                # print "rhs is CPermutation:", rhs
##
##                in_im, rhs_red = rhs.factor_rhs(in_index)
##                # print "in_im <- in_index:", in_im, "<-", in_index
##                # print "rhs_red:", rhs_red
##
##                return cls.create(SeriesProduct.create(*circuit.operands[:-1]), out_index, in_im) << rhs_red
##
##            elif isinstance(rhs, Concatenation):
##                _, block_index = rhs.index_in_block(in_index)
##                bs = rhs.block_structure
##                nbefore, nblock, nafter = sum(bs[:block_index]), bs[block_index], sum(bs[block_index + 1:])
##                before, block, after = rhs.get_blocks((nbefore, nblock, nafter))
##                if before != cid(nbefore) or after != cid(nafter):
##                    outer_rhs = before + cid(nblock - 1) + after
##                    inner_rhs = cid(nbefore) + block + cid(nafter)
##                    return cls.create(SeriesProduct.create(*(circuit.operands[:-1] + (inner_rhs,))), out_index, in_index) << outer_rhs
##                elif block == cid(nblock):
##                    outer_rhs = before + cid(nblock - 1) + after
##                    return cls.create(SeriesProduct.create(*circuit.operands[:-1]), out_index, in_index) << outer_rhs
##
##        if isinstance(circuit, SLH) and isinstance(circuit.S, OperatorMatrix) and isinstance(circuit.L, OperatorMatrix):
##            return circuit.feedback(out_index, in_index)
##
##        if isinstance(circuit, CPermutation):
##            return circuit.feedback(out_index, in_index)
#
#        return cls(circuit, out_index, in_index)
    
#    def evalf(self):
#        # print self.__class__.__name__
#        op = evalf(self.operand)
#        of =  op.feedback(*self.out_in_pair)
#        # print of, repr(of)
#        return of
        
    def toSLH(self):
        return self.operand.toSLH().feedback(*self.out_in_pair)
    
    def reduce(self):
        return self.operand.reduce().feedback(*self.out_in_pair)
    
    
#    def substitute(self, var_map):
#        op = substitute(self.operand, var_map)
#        return op.feedback(*self.out_in_pair)
    
    def __str__(self):
        if self.out_in_pair == (self.operand.cdim - 1, self.operand.cdim - 1):
            return "FB(%s)" % self.operand
        o, i = self.out_in_pair
        return "FB(%s, %d, %d)" % (self.operand, o, i)
    
    def tex(self):
        o, i = self.out_in_pair
        if self.out_in_pair == (self.cdim -1, self.cdim-1):
            return "\left\lfloor%s\\right\\rfloor" % tex(self.operand)
        return "\left\lfloor%s\\right\\rfloor_{%d\\to%d}" % (tex(self.operand), o, i)
    

    def series_inverse(self):
        return Feedback.create(self.operand.series_inverse(), *reversed(self.out_in_pair))
    
        
        


def FB(circuit, out_index = None, in_index = None):
    if out_index is None:
        out_index = circuit.cdim -1
    if in_index is None:
        in_index = circuit.cdim -1
    return Feedback.create(circuit, out_index, in_index)

@check_signature
class SeriesInverse(Circuit, Operation):
    signature = Circuit,

    delegate_to_method = (SeriesProduct, Concatenation, Feedback, SLH, CPermutation, CIdentity.__class__)

    @property
    def operand(self):
        return self.operands[0]


    @classmethod
    def create(cls, circuit):
        if isinstance(circuit, SeriesInverse):
            return circuit.operand
        elif isinstance(circuit, cls.delegate_to_method):
#            print circuit
            return circuit.series_inverse()
#            return SeriesProduct.create(*reversed(SeriesInverse.create(op) for op in circuit.operands))
#        elif isinstance(circuit, Concatenation):
#            return Concatenation.create(*[SeriesInverse.create(op) for op in circuit.operands])
#        elif isinstance(circuit, Feedback):
#            return Feedback(SeriesInverse.create(circuit.operand), *reversed(circuit.out_in_pair))
#        elif isinstance(circuit, CIdentity):
#            return circuit
#        elif isinstance(circuit, CPermutation):
#            return CPermutation(invert_permutation(circuit.permutation))
        return cls(circuit)
    
    @property
    def cdim(self):
        return self.operand.cdim
    
    
#    @classmethod
#    def check_operands(cls, *operands):
#        UnaryOperation.check_operands.im_func(cls, *operands)
#
#        op = operands[0]
#        if isinstance(op, SeriesInverse):
#            raise Exception()
#        n = op.cdim
#        if op == circuit_identity(n):
#            raise Exception
#        if isinstance(op, SeriesProduct):
#            raise Exception
#        if isinstance(op, Concatenation):
#            raise Exception
    
    def toSLH(self):
        return self.operand.toSLH().series_inverse()
    
#    def evalf(self):
#        return evalf(self.operand).series_inverse()
    
    def reduce(self):
        return self.operand.reduce().series_inverse()
    
    def substitute(self, var_map):
        return substitute(self, var_map).series_inverse()
    
    # def scatter(self, index):
    #     return self.operand.scatter(index)
    
    def __str__(self):
        return "[%s]^(-1)" % self.operand
    
    def tex(self):
        return r"\left[ %s \right]^{-1}" % tex(self.operand)

#@trace
def tensor_decompose_series(lhs, rhs):
    if isinstance(rhs, CPermutation):
        raise CannotSimplify
    res_struct = get_common_block_structure(lhs.block_structure, rhs.block_structure)
    if len(res_struct) > 1:
        blocks, oblocks = lhs.get_blocks(res_struct), rhs.get_blocks(res_struct)
        parallel_series = [SeriesProduct.create(lb, rb)  for (lb, rb) in izip(blocks, oblocks)]
        return Concatenation.create(*parallel_series)
    raise CannotSimplify()

#@trace
def factor_permutation_for_blocks(lhs, rhs):
#    print lhs, rhs
    rbs = rhs.block_structure
    if rhs == cid(rhs.cdim):
        return lhs
    if len(rbs) > 1:
        residual_lhs, transformed_rhs, carried_through_lhs = lhs.factorize_for_rhs(rhs)
        if residual_lhs == lhs:
        # if not transformed_rhs == rhs and carried_through_lhs == id_nc:
        # print "WARNING:", transformed_rhs, "!=", rhs,"and", carried_through_lhs, "!=", id_nc
        # raise AssertionError()
            raise CannotSimplify()

        # print lhs, " << ", rhs, " --> ", residual_lhs, " << ", transformed_rhs, " << ", carried_through_lhs
        # display_circuit(SeriesProduct(lhs, rhs))
        # display_circuit(SeriesProduct(residual_lhs, transformed_rhs, carried_through_lhs))
        return SeriesProduct.create(residual_lhs, transformed_rhs, carried_through_lhs)
    raise CannotSimplify()

#def move_perms_through(lhs, rhs):
#    if isinstance(lhs, SeriesProduct) and isinstance(lhs.operands[-1], CPermutation):
#        if isinstance(rhs, SeriesProduct) and isinstance(rhs.operands[-1], CPermutation):
#            left_part =  Concatenation.create(SeriesProduct.create(*lhs.operands[:-1]), SeriesProduct.create(*rhs.operands[:-1]))
#            right_part = Concatenation.create(lhs.operands[-1], rhs.operands[-1])
#            factored_out =  SeriesProduct.create(left_part, right_part)
#            assert isinstance(factored_out, SeriesProduct)
#            return factored_out
#        else:
#            left_part =  Concatenation.create(SeriesProduct.create(*lhs.operands[:-1]), rhs)
#            right_part = Concatenation.create(lhs.operands[-1], cid(rhs.cdim))
#            factored_out =  SeriesProduct.create(left_part, right_part)
#            assert isinstance(factored_out, SeriesProduct)
#            return factored_out
#
#    elif isinstance(rhs, SeriesProduct) and isinstance(rhs.operands[-1], CPermutation):
#        left_part =  Concatenation.create(lhs, SeriesProduct.create(*rhs.operands[:-1]))
#        right_part = Concatenation.create(cid(lhs.cdim), rhs.operands[-1])
#        factored_out =  SeriesProduct.create(left_part, right_part)
#        assert isinstance(factored_out, SeriesProduct)
#        return factored_out
#    raise CannotSimplify()
#@trace
def pull_out_perm_lhs(lhs, rest, out_index, in_index):
#    print lhs, rest
    out_inv , lhs_red = lhs.factor_lhs(out_index)
    return lhs_red << Feedback.create(SeriesProduct.create(*rest), out_inv, in_index)

def pull_out_unaffected_blocks_lhs(lhs, rest, out_index, in_index):
    _, block_index = lhs.index_in_block(out_index)
    # print "index_in_block, block_index", index_in_block, block_index

    bs = lhs.block_structure
    # print "block_structure", bs

    nbefore, nblock, nafter = sum(bs[:block_index]), bs[block_index], sum(bs[block_index + 1:])
    before, block, after = lhs.get_blocks((nbefore, nblock, nafter))
    # print "before, block, after", before, block, after

    if before != cid(nbefore) or after != cid(nafter):
        outer_lhs = before + cid(nblock - 1) + after
        inner_lhs = cid(nbefore) + block + cid(nafter)
        return outer_lhs << Feedback.create(SeriesProduct.create(inner_lhs, *rest), out_index, in_index)
    elif block == cid(nblock):
        outer_lhs = before + cid(nblock - 1) + after
        return outer_lhs << Feedback.create(SeriesProduct.create(*rest), out_index, in_index)
    raise CannotSimplify()
#@trace
def pull_out_perm_rhs(rest, rhs, out_index, in_index):
    in_im, rhs_red = rhs.factor_rhs(in_index)
    # print "in_im <- in_index:", in_im, "<-", in_index
    # print "rhs_red:", rhs_red
    return Feedback.create(SeriesProduct.create(*rest), out_index, in_im) << rhs_red

def pull_out_unaffected_blocks_rhs(rest, rhs, out_index, in_index):
    _, block_index = rhs.index_in_block(in_index)
    bs = rhs.block_structure
    nbefore, nblock, nafter = sum(bs[:block_index]), bs[block_index], sum(bs[block_index + 1:])
    before, block, after = rhs.get_blocks((nbefore, nblock, nafter))
    if before != cid(nbefore) or after != cid(nafter):
        outer_rhs = before + cid(nblock - 1) + after
        inner_rhs = cid(nbefore) + block + cid(nafter)
        return Feedback.create(SeriesProduct.create(*(rest + (inner_rhs,))), out_index, in_index) << outer_rhs
    elif block == cid(nblock):
        outer_rhs = before + cid(nblock - 1) + after
        return Feedback.create(SeriesProduct.create(*rest), out_index, in_index) << outer_rhs
    raise CannotSimplify()


def series_feedback(series, out_index, in_index):
    series_s = series.series_inverse().series_inverse()
#    print series, series_s, out_index, in_index
    if series_s == series:
        raise CannotSimplify()
    return series_s

A_CPermutation = wc("A", head = CPermutation)
B_CPermutation = wc("B", head = CPermutation)
C_CPermutation = wc("C", head = CPermutation)
D_CPermutation = wc("D", head = CPermutation)

A_Concatenation= wc("A", head = Concatenation)
B_Concatenation = wc("B", head = Concatenation)

A_SeriesProduct = wc("A", head = SeriesProduct)

A_Circuit = wc("A", head = Circuit)
B_Circuit = wc("B", head = Circuit)
C_Circuit = wc("C", head = Circuit)

A__Circuit = wc("A__", head = Circuit)
B__Circuit = wc("B__", head = Circuit)
C__Circuit = wc("C__", head = Circuit)

A_SLH = wc("A", head = SLH)
B_SLH = wc("B", head = SLH)

A_ABCD = wc("A", head = ABCD)
B_ABCD = wc("B", head = ABCD)

j_int = wc("j", head = int)
k_int = wc("k", head = int)

SeriesProduct.binary_rules += [
    ((A_CPermutation, B_CPermutation), lambda A, B: A.series_with_permutation(B)),
    ((A_SLH, B_SLH), lambda A, B: A.series_with_slh(B)),
    ((A_ABCD, B_ABCD), lambda A, B: A.series_with_abcd(B)),
    ((A_Circuit, B_Circuit), lambda A, B: tensor_decompose_series(A,B)),
    ((A_CPermutation, B_Circuit), lambda A, B: factor_permutation_for_blocks(A,B))
]

Concatenation.binary_rules += [
    ((A_SLH, B_SLH), lambda A, B: A.concatenate_slh(B)),
    ((A_ABCD, B_ABCD), lambda A, B: A.concatenate_abcd(B)),
    ((A_CPermutation, B_CPermutation), lambda A, B: CPermutation.create(concatenate_permutations(A.operands[0], B.operands[0]))),
    ((A_CPermutation, CIdentity), lambda A: CPermutation.create(concatenate_permutations(A.operands[0], (0,)))),
    ((CIdentity, B_CPermutation ), lambda B: CPermutation.create(concatenate_permutations((0,), B.operands[0]))),
    ((SeriesProduct(A__Circuit, B_CPermutation), SeriesProduct(C__Circuit, D_CPermutation)), lambda A, B, C, D: (SeriesProduct.create(*A) + SeriesProduct.create(*C)) << (B + D)),
    ((SeriesProduct(A__Circuit, B_CPermutation), C_Circuit), lambda A, B, C: (SeriesProduct.create(*A) + C) << (B + cid(C.cdim))),
    ((A_Circuit, SeriesProduct(B__Circuit, C_CPermutation)), lambda A, B, C: (A + SeriesProduct.create(*B)) << (cid(A.cdim) + C)),
#    ((A_Circuit, B_Circuit), lambda A, B: move_perms_through(A,B)),
]

Feedback.rules += [
    ((A_SeriesProduct, j_int, k_int), lambda A, j, k: series_feedback(A, j, k)),
    ((SeriesProduct(A_CPermutation, B__Circuit),j_int, k_int ), lambda A, B, j, k: pull_out_perm_lhs(A,B,j,k)),
    ((SeriesProduct(A_Concatenation, B__Circuit),j_int, k_int ), lambda A, B, j, k: pull_out_unaffected_blocks_lhs(A,B,j,k)),
    ((SeriesProduct(A__Circuit, B_CPermutation),j_int, k_int ), lambda A, B, j, k: pull_out_perm_rhs(A,B,j,k)),
    ((SeriesProduct(A__Circuit, B_Concatenation),j_int, k_int ), lambda A, B, j, k: pull_out_unaffected_blocks_rhs(A,B,j,k)),
]



#if __name__ == "__main__":
#    from testing.test_circuit_algebra import *
#    unittest.main()



