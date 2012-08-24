from abstract_algebra import *

## helpers

def check_permutation(permutation):
    return list(sorted(permutation)) == range(len(permutation))

def invert_permutation(permutation):
    return tuple([permutation.index(p) for p in range(len(permutation))])

def permutation_to_disjoint_cycles(permutation):
    if not check_permutation(permutation):
        raise Exception('Malformed permutation %r' % permutation)

    p_index = 0
    current_cycle = [0]
    permutation_nums = sorted(permutation)
    permutation_nums.remove(0)

    cycles = []
    while(True):
        p_index = permutation[p_index]
        # print p_index
        if p_index == current_cycle[0]:
            # current_cycle.sort()
            cycles.append(current_cycle)
            # print current_cycle
            try:
                p_index = permutation_nums.pop(0)
                current_cycle = [p_index]
            except IndexError:
                break
        else:
            permutation_nums.remove(p_index)
            current_cycle.append(p_index)

    # cycles.sort()
    return cycles

def permutation_from_disjoint_cycles(cycles, offset = 0):
    perm_length = sum(map(len, cycles))
    res_perm = range(perm_length)
    for c in cycles:
        p1 = c[0] - offset
        for p2 in c[1:]:
            p2 = p2 - offset
            res_perm[p1] = p2
            p1 = p2
        res_perm[p1] = c[0] - offset #close cycle
    assert sorted(res_perm) == range(perm_length)
    return tuple(res_perm)

def permutation_to_block_permutations(permutation):

    if len(permutation) == 0:
        raise Exception

    cycles = permutation_to_disjoint_cycles(permutation)

    if len(cycles) == 1:
        return (permutation,)
    current_block_start = cycles[0][0]
    current_block_end = max(cycles[0])
    current_block_cycles = [cycles[0]]
    res_permutations = []
    for c in cycles[1:]:
        if c[0] > current_block_end:
            res_permutations.append(permutation_from_disjoint_cycles(current_block_cycles, current_block_start))
            assert sum(map(len, current_block_cycles)) == current_block_end - current_block_start + 1
            current_block_start = c[0]
            current_block_end = max(c)
            current_block_cycles = [c]
        else:
            current_block_cycles.append(c)
            if max(c) > current_block_end:
                current_block_end = max(c)

    res_permutations.append(permutation_from_disjoint_cycles(current_block_cycles, current_block_start))

    assert sum(map(len, current_block_cycles)) == current_block_end - current_block_start + 1
    assert sum(map(len, res_permutations)) == len(permutation)
    return res_permutations


def permutation_from_block_permutations(permutations):

    offset = 0
    new_perm = []
    for p in permutations:
        new_perm[offset: offset +len(p)] = [p_i + offset for p_i in p]
        offset += len(p)
    return tuple(new_perm)


def compose_permutations(a, b):
#    if len(a) != len(b):
#        raise ValueError(str((a,b)))
#    return tuple([a[p] for p in b])
    return permute(a, b)

def concatenate_permutations(a, b):
    l = len(a)
    return a + tuple(bj + l for bj in b)

def permute(sequence, permutation):
    if len(sequence) != len(permutation):
        raise ValueError((sequence, permutation))
    if not check_permutation(permutation):
        raise ValueError(str(permutation))
    
    if type(sequence) in (list, tuple, str):
        constructor = type(sequence)
    else:
        constructor = list
    return constructor((sequence[p] for p in permutation))


    #
#
#@check_signature_flat
#class Permutation(Operation):
#    signature = int,
#
#    @property
#    def length(self):
#        raise NotImplementedError(self.__class__.__name__)
#
#    def __len__(self):
#        return self.length
#
#    def toPTuple(self):
#        raise NotImplementedError(self.__class__.__name__)
#
#    def getCycles(self):
#
#
#    def asBlocks(self):
#        return permutation_to_block_permutations(self.toPTuple.operands)
#
#    def __add__(self, other):
#        return Permutation(permutation_from_block_permutations((self.toPTuple(), other.toPTuple())))
#
#    def __mul__(self, other):
#        return Permutation(compose_permutations(self.toPTuple(), other.toPTuple()))
#
#
#@check_signature
#class PIdentity(Permutation, Operation):
#    signature = int,
#
#
#class PTuple(Permutation, Operation):
#
#    @classmethod
#    def create(cls,*operands):
#        if not check_permutation(operands):
#            raise ValueError(str(operands))
#
#        if list(operands) == range(len(operands)):
#            return PIdentity(len(operands))
#
#        return super(Permutation, cls).create(*operands)
#
#
#
#    def __init__(self, *t):
#        if not check_permutation(t):
#            raise ValueError(str(t))
#        super(Permutation, self).__init__(*t)
#
#
#@check_signature_flat
#class PSum(Permutation, Operation)
#    signature = Permutation,
#
#
#
## TODO Add test code