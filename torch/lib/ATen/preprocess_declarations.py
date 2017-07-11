import re
from copy import deepcopy
from function_wrapper import TYPE_FORMAL_GENERIC
import common_with_cwrap

type_map = {
    'floating_point': [
        'Float',
        'Double',
        'Half',
    ],
    'integral': [
        'Byte',
        'Char',
        'Short',
        'Int',
        'Long'
    ],
}

all_types = type_map['floating_point'] + type_map['integral']
type_map['all'] = all_types

all_backends = ['CPU', 'CUDA', 'SparseCPU', 'SparseCUDA']
default_backends = ['CPU', 'CUDA']


def process_types_and_backends(option):
    # if specific pairs were not listed, then enumerate them
    # based on the backend and type attributes
    # if backend or type is not defined, it is assumed to be all of them
    if 'backend_type_pairs' not in option:
        backends = option.get('backends', default_backends)
        types = option.get('types', all_types)
        pairs = [[p, t] for p in backends for t in types]
    else:
        pairs = option['backend_type_pairs']

    # expand type alias (integral, floating_point, all)
    def expand(pair):
        p, t = pair
        assert(p in all_backends)
        if t in type_map:
            return [(p, tt) for tt in type_map[t]]
        assert(t in all_types)
        return [(p, t)]
    pairs = set(p for pair in pairs for p in expand(pair))

    # special case remove Half for cpu unless it is explicitly enabled
    if not option.get('cpu_half', False):
        pairs.discard(('CPU', 'Half'))

    # sort the result for easy reading
    option['backend_type_pairs'] = sorted([p for p in pairs])


def exclude(declaration):
    return 'only_register' in declaration or declaration.get('python_name') == 'ndimension'


def add_variants(option):
    option.setdefault('variants', ['method'])

# if we have 'output' arguments, generate a variant where
# we mark oututs as allocate = True, and where the method variant
# is disabled...


def handle_outputs_taken_as_arguments(options):
    new_options = []
    for option in options:
        if any('output' in arg for arg in option['arguments']):
            allocate_option = deepcopy(option)
            # the allocating option needs to be marked
            for arg in allocate_option['arguments']:
                if 'output' in arg:
                    arg['allocate'] = True

            # the original option, which takes arguments for the results,
            # is no longer a method, and has _out added to indicte it takes
            # output arguments
            if 'function' in option['variants']:
                if 'method' in option['variants']:
                    option['variants'].remove('method')
                option['api_name'] += '_out'
                new_options.append(option)

            new_options.append(allocate_option)
        else:
            new_options.append(option)
    return new_options


def sanitize_return(option):
    ret = option['return']
    m = re.match('argument (\d+(,\d+)*)', ret)
    if m is not None:
        arguments = [int(x) for x in m.group(1).split(',')]
        option['return'] = {'kind': 'arguments', 'arguments': arguments}
    elif ret == 'self':
        option['return'] = {'kind': 'arguments', 'arguments': []}
        for i, x in enumerate(option['arguments']):
            if x['name'] == 'self':
                option['return']['arguments'].append(i)
                break
    else:
        option['return'] = {'kind': 'type', 'type': option['return']}


def set_mode(option):
    option['mode'] = option.get('mode', 'TH')

# To enable 0-dim support in TH operations
# we find all places where a single Scalar replaced with a Tensor
# as an argument is still a valid function
# we then mark the tensor variant with a key zero_dim_dispatch_when_scalar: name
# where 'name' is the name of the argument that should be a scalar
# during dispatch, if that argument is marked internally as holding a scalar
# then the method will dispatch to that function.


def discover_zero_dim_tensor_operations(declaration):
    def exclude(arg):
        return arg.get('ignore_check')

    def signature(option, i=None, value=None):
        elements = [TYPE_FORMAL_GENERIC.get(arg['type'], arg['type'])
                    if i is None or j != i else value
                    for j, arg in enumerate(option['arguments'])
                    if not exclude(arg)]
        return '#'.join(elements)
    signature_to_option = {signature(option): option
                           for option in declaration['options']}

    for option in declaration['options']:
        for i, arg in enumerate(option['arguments']):
            if arg['type'] == 'real':
                signature_of_tensor_version = signature(option, i, 'Tensor &')
                if signature_of_tensor_version in signature_to_option:
                    tensor_version = \
                        signature_to_option[signature_of_tensor_version]
                    names = [arg['name'] for arg in tensor_version['arguments']
                             if not exclude(arg)]
                    tensor_version['zero_dim_dispatch_when_scalar'] = names[i]
                    # print("FOUND "+str(i)   )
                    # print("Scalar Version ===== ")
                    # print(yaml.dump(option))
                    # print("Tensor Version ===== ")
                    # print(yaml.dump(tensor_version))
                    # print("SHARED "+names[i])


def run(declarations):
    declarations = [d for d in declarations if not exclude(d)]
    for declaration in declarations:
        common_with_cwrap.set_declaration_defaults(declaration)
        common_with_cwrap.enumerate_options_due_to_default(
            declaration,
            allow_kwarg=False,
            type_to_signature=TYPE_FORMAL_GENERIC,
            remove_self=True)
        common_with_cwrap.sort_by_number_of_options(declaration)
        discover_zero_dim_tensor_operations(declaration)

        for option in declaration['options']:
            set_mode(option)
            sanitize_return(option)
            process_types_and_backends(option)
            add_variants(option)
        declaration['options'] = handle_outputs_taken_as_arguments(
            declaration['options'])
    return declarations
