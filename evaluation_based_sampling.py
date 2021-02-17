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
        #print("current op is", op)
        args = exp[1:]
        #print("current args:", args)
        
        if op == 'sample':
            #dist = evaluate_program(args,local_env)
            #print("distribution is")
            return env['sample*'](evaluate_program(args,local_env,func))
        elif op == 'observe':
            #ignore?
            #env['observe*'](evaluate_program(args,local_env,func))
            return torch.tensor(float(0))
            #return (evaluate_program(args,local_env,func))
        elif type(op) == int or type(op)== float:
            return op
        elif op == 'let':
            #print("\n let statement!! \n")
            args = args[0]
            varname = args[0]
            #print("varname is", varname)
            #print("args[1] is:", args[1])
            c1 = evaluate_program(args[1],local_env,func)
            #print("\n here is c1", c1)
            local_env[varname]=c1 #add to dictionary
            #print("new local env is", local_env) 
            return evaluate_program(exp[2:],local_env,func)
        elif op == 'if':
            #print("boolean is", exp[1])
            #if exp[1]:
            if evaluate_program(exp[1],local_env,func):
                return evaluate_program(exp[2],local_env,func)
            else:
                return evaluate_program(exp[3],local_env,func)
        #elif type(op) == str and op in local_env.keys()
        #else:
        elif op[0] == 'defn':
            #print("uh oh!!")
            fname = op[1]
            #print(fname)
            numvars = len(op[2])
            #print(numvars)
            body = op[3]
            #print(body)
            func[fname] = [ op[2],op[3]]
            #print(func[fname])
            #print(func)
            #print(args)
            return evaluate_program(args,local_env,func)
        else:
            c =[]
            for i in range(0, len(args)):
                c.append(evaluate_program(args[i], local_env,func))
            #for i in range(1,len(args)):
            #    c[i] = evaluate_program(args[i])
            if type(op) == str:
                #print("current expression is: ",exp)
                #print("op is: ", op)
                #print("args is:",args)
                #print("env[op] is:", env[op])
                #print("c is", c)
                if op in list(env.keys()):
                    result = env[op](*c)
                    #print("result is:", result)
                    return result
                elif op in list(func.keys()):
                    #print("\n \n using local variable: ", op)
                    variables = func[op][0]
                    body = func[op][1]
                    new_env = {}
                    for i in range(0,len(variables)):
                        print(variables[i])
                        #local_env[variables[i]] = c[i]
                        if type(c[i]) is list:
                            new_env[variables[i]] = c[i].copy()
                        else:
                            new_env[variables[i]] = c[i]
                        #print("\n \n local_env[variables[i]]", local_env[variables[i]])
                        #print("\n \n local_env[variables[i]]", new_env[variables[i]])
                        #print("\n \n c[i] is", c[i])
                    #print(local_env)
                    #return evaluate_program(body,local_env,func)
                    #print("\n new env: ", new_env)
                    return evaluate_program(body,new_env,func)
                else:
                    print("current op giving error: ", op)
                    raise("operation type invalid", op)
                #return env[op](*c)
                #return env[op]( *map (lambda a: evaluate_program(a,local_env),args) )
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
        #note: this path should be with respect to the daphne path!
        #ast = daphne(['desugar', '-i', '../../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
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

        
if __name__ == '__main__':

    run_deterministic_tests()
    
    #run_probabilistic_tests()

    local_env = {}
    func = {}

    for i in range(3,4):
        #ast = daphne(['desugar', '-i', '../../CS532-HW2/programs/{}.daphne'.format(i)])
        ast = daphne(['desugar', '-i', '../../cpsc532_hw2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        print("abstract syntax tree is:", ast)
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast,local_env,func))




#{'observe-data': [['_', 'data', 'slope', 'bias'], 
# ['let', ['xn', ['first', 'data']], ['let', ['yn', ['second', 'data']], ['let', ['zn', ['+', ['*', 'slope', 'xn'], 'bias']], ['let', ['dontcare0', ['observe', ['normal', 'zn', 1.0], 'yn']], ['rest', ['rest', 'data']]]]]]]}