import torch
import torch.distributions as dist

from daphne import daphne
from tests import is_tol, run_prob_test,load_truth

from primitives import sample, vector, get, put, first, second, last, rest, append, hashmap, observe, iff, less_than, greater_than, my_discrete, my_sqrt, my_mult

env = {'normal': dist.Normal,
       #'sqrt': torch.sqrt,
       'sqrt':my_sqrt,
       '+': torch.add,
       '/': torch.divide,
       '*': torch.multiply,
       'sample*': sample,
       'beta': dist.Beta,
       'exponential': dist.Exponential,
       'uniform': dist.Uniform,
       'vector': vector,
       'get':get,
       'put':put,
       'first':first,
       'second':second,
       'last':last,
       'rest':rest,
       'append':append,
       'hash-map':hashmap,
       'observe*':observe,
       'if':iff,
       '<':less_than,
       '>':greater_than,
       #'discrete':dist.Categorical,
       'discrete':my_discrete,
       'mat-transpose':torch.transpose,
       'mat-add':torch.add,
       'mat-tanh':torch.tanh
       }

        
def evaluate_program(ast,local_env,func):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    (sampling from the prior in the probabilistic 
    programming context means generating samples of 
    the return value)
    """

    if not ast:
        return ast


    if type(ast) is int or type(ast) is float:
        return torch.tensor(float(ast))
    elif type(ast) is torch.Tensor:
        return ast
    elif type(ast)==str:
        return local_env[ast]
    else:
        exp = ast


    if type(exp) is list:
        while len(exp) == 1:
            exp = exp[0]


    if type(exp) is str:
        return evaluate_program(local_env[exp],local_env,func)


    elif type(exp) is list:
        op = exp[0]
        args = exp[1:]
        
        if op == 'sample':
            return env['sample*'](evaluate_program(args,local_env,func))
        elif op == 'observe':
            return torch.tensor(float(0))
        elif type(op) == int or type(op)== float:
            return op
        elif op == 'let':
            args = args[0]
            varname = args[0]
            c1 = evaluate_program(args[1],local_env,func)
            local_env[varname]=c1 #add to dictionary
            return evaluate_program(exp[2:],local_env,func)
        elif op == 'if':
            if evaluate_program(exp[1],local_env,func):
                return evaluate_program(exp[2],local_env,func)
            else:
                return evaluate_program(exp[3],local_env,func)
        elif op[0] == 'defn':
            fname = op[1]
            numvars = len(op[2])
            body = op[3]
            func[fname] = [ op[2],op[3]]
            return evaluate_program(args,local_env,func)
        else:
            c =[]
            for i in range(0, len(args)):
                c.append(evaluate_program(args[i], local_env,func))
            if type(op) == str:
                if op in list(env.keys()):
                    result = env[op](*c)
                    return result
                elif op in list(func.keys()):
                    variables = func[op][0]
                    body = func[op][1]
                    new_env = {}
                    for i in range(0,len(variables)):
                        print(variables[i])
                        if type(c[i]) is list:
                            new_env[variables[i]] = c[i].copy()
                        else:
                            new_env[variables[i]] = c[i]
                    return evaluate_program(body,new_env,func)
                else:
                    print("current op giving error: ", op)
                    raise("operation type invalid", op)
            else:
                print(op)
                raise("operation type invalid", op)


    ## TO DO
    return None


def get_stream(ast):
    """Return a stream of prior samples"""
    local_env = {}
    func = {}
    while True:
        result = evaluate_program(ast,local_env,func)
        print("sample result is: ", result)
        yield result
        #yield evaluate_program(ast,local_env)
    


def run_deterministic_tests():
    
    for i in range(1,14):
        ast = daphne(['desugar', '-i', '../../cpsc532_hw2/programs/tests/deterministic/test_{}.daphne'.format(i)])

        print("abstract syntax tree is:", ast)
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        local_env = {}
        func = {}
        #ret, sig = evaluate_program(ast,local_env)
        ret = evaluate_program(ast,local_env,func)
        print("return value is:", ret)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Deterministic Test {} passed'.format(i))
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    #num_samples=1e4
    num_samples=100
    max_p_value = 1e-4
    
    for i in range(5,7):
        #note: this path should be with respect to the daphne path!        
        #ast = daphne(['desugar', '-i', '../../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        ast = daphne(['desugar', '-i', '../../cpsc532_hw2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        print("abstract syntax tree is:", ast)
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)

        print('Probabilistic Test {} passed'.format(i))
    
    print('All probabilistic tests passed')    


def plot_tests():
    num_samples = 10

    graph1 = daphne(['desugar', '-i', '../../cpsc532_hw2/programs/tests/deterministic/test_1.daphne'])

    graph2 = daphne(['desugar', '-i', '../../cpsc532_hw2/programs/tests/deterministic/test_2.daphne'])



if __name__ == '__main__':

    run_deterministic_tests()
    
    run_probabilistic_tests()

    local_env = {}
    func = {}

    for i in range(3,4):
        #ast = daphne(['desugar', '-i', '../../CS532-HW2/programs/{}.daphne'.format(i)])
        ast = daphne(['desugar', '-i', '../../cpsc532_hw2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        print("abstract syntax tree is:", ast)
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast,local_env,func))

