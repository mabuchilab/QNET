from abstract_algebra import *



class Singleton(object):
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            return cls._instance
        _instance = object.__new__(cls)
        return _instance

def modify_first_arg(fn, modifyer_fn):
    def decorated_fn(arg1, *other_args):
        return fn(modifyer_fn(arg1), *other_args)
    return decorated_fn

def modify_second_arg(fn, modifyer_fn):
    def decorated_fn(arg1, arg2, *other_args):
        return fn(arg1, modifyer_fn(arg2), *other_args)
    return decorated_fn

def reverse_args(fn):
    def decorated_fn(*args):
        return fn(*reversed(args))
    return decorated_fn
    

def non_zero(n):
    return (not is_number(n)) or n != 0

def not_one(n): 
    return not is_number(n) or n != 1

filter_out_zeros = lambda cls, ops: filter(non_zero, ops)
filter_out_ones = lambda cls, ops: filter(not_one, ops)


def product(sequence, neutral = 1):
    return reduce(lambda a, b: a * b, sequence, neutral)



def subtract(a, b):
    return a + b*(-1)




def expand_operands_associatively(cls, operands):    
    for i, o in enumerate(operands):
        #find first occurance of similar operation among operands
        if type(o) == cls:
            # insert this operation's operands in place of the operation
            # recursively treat remaining operands
            return operands[:i] + o.operands + expand_operands_associatively(cls, operands[i+1:])
    return operands
    

sort_operands = lambda cls, ops: tuple(sorted(ops))


def collect_distributively(cls, coeff_cls, coeff_identity, operands):
    if len(operands) < 2:
        return operands
    
    operands = sort_operands(cls, operands)
    
    last_coeff, last_term = None, None
    
    for i, o in enumerate(operands):
        o_coeff, o_term = (o.coeff, o.term) if isinstance(o, coeff_cls) \
                else (coeff_identity, o)
        if o_term == last_term:
            remaining_operands = ((o_coeff + last_coeff) * o_term,) + operands[i+1:]
            return operands[: i-1 ] + collect_distributively(cls, coeff_cls, coeff_identity, remaining_operands)
        last_coeff, last_term = o_coeff, o_term
    return operands
    
    
    
zero_factor = lambda cls, ops: ops if all(map(non_zero, ops)) else 0

def factor_out_coeffs(cls, coeff_cls, coeff_identity, operands):
    if len(operands) < 2:
        return 1, operands
    coeff = coeff_identity
    n_operands = list(operands)
    for i, o in enumerate(operands):
        if isinstance(o, coeff_cls):
            coeff *= o.coeff
            n_operands[i] = o.term
    return coeff, n_operands
    


    
    
class frozendict(dict):
    __slots__ = ['_cached_hash']
    
    def _blocked_attribute(self, obj):
        raise AttributeError, "A frozendict cannot be modified."
    _blocked_attribute = property(_blocked_attribute)

    __delitem__ = __setitem__ = clear = _blocked_attribute
    pop = popitem = setdefault = update = _blocked_attribute

    def __new__(cls, *args):
        new = dict.__new__(cls)
        dict.__init__(new, *args)
        return new

    def __init__(self, *args):
        self._chached_hash = None
        pass

    def __hash__(self):
        h = self._cached_hash
        if h is None:
            h = self._cached_hash = hash(tuple(sorted(self.items())))    
        return h

    def __repr__(self):
        return "frozendict(%s)" % dict.__repr__(self)
        
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
    
def fixpoints(permutation):
    return [k for k, p in enumerate(permutation) if k == p]
    # return filter(lambda k: permutation[k] == k, range(len(permutation)))
    
if __name__ == '__main__':
    perm = (1,2,3,4,5,7,6,0,10,9,8)
    print check_permutation(perm)
    cycles =  permutation_to_disjoint_cycles(perm)
    print cycles
    print permutation_to_block_permutations(perm)
    print permutation_from_disjoint_cycles(cycles)
    

def conditional_combine_pairwise(binary_condition, binary_operation, lst):
    offset = 0
    reduced = False
    while True:
        for i,(a,b) in enumerate(izip(lst[offset:-1], lst[offset+1:])):
            if binary_condition(a,b):
                reduced = True
                lst[i] = binary_operation(a,b)
                del lst[i+1]
                offset += i
                break
        if not reduced:
            break
        reduced = False
    return lst
    
            
binary_condition = lambda a,b: a % 2 == 0
binary_operation = lambda a,b: a*b

# print reduce_pairwise(binary_condition, binary_operation, [111,1,1,1,1,1,1])
def add_dicts(d1, d2):
    """
     Add two dictionaries like (sparse) vectors, 
     where the values correspond to the scalar amplitude of the basis vector
     component designated by the key.
    """
    ret_dict = dict(filter(lambda (rep, amp): amp != 0, d1.items())) #@IndentOk
    #ret_dict = d1.copy() @IndentOk
    for rep, amp in d2.items():
        p_amp = ret_dict.get(rep, 0)
        n_amp = p_amp + amp
        if n_amp != 0:
            ret_dict[rep] = n_amp
        elif p_amp != 0: # but n_amp == 0
            del ret_dict[rep]
    return ret_dict
