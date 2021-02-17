import torch
import torch.distributions as dist

import matplotlib.pyplot as plt

from daphne import daphne
from functools import partial

from primitives import sample, vector, get, put, last, append, observe, iff, less_than, greater_than, my_discrete
from tests import is_tol, run_prob_test,load_truth



# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'normal': dist.Normal,
       'sqrt': torch.sqrt,
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
       'last':last,
       'append':append,
       #'observe*':observe,
       'if':iff,
       '<':less_than,
       '>':greater_than,
       #'discrete':dist.Categorical,
       'discrete':my_discrete,
       'mat-transpose':torch.transpose,
       'mat-add':torch.add,
       'mat-tanh':torch.tanh
       }


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    else:
        raise("Expression type unknown.", exp)


def ancestral_eval(exp,functions):
    #print("functions are: ",functions)
    #print("current expression:  ", exp)
    #print("type of exp is:  ",type(exp))
    #print("type of functions is:  ",type(functions))

    if type(exp) is list:
        op = exp[0]
        #print("operation is:  ",op)
        #print(type(functions[op]))
        args = exp[1:]
        #print("arguments are:  ", args)
        return functions[op](*map(lambda a: ancestral_eval(a,functions),args))
    elif type(exp) is int or type(exp) is float:
        #print("we have a float")
        return torch.tensor(float(exp))
    elif type(exp) is str and exp in list(functions.keys()):
        new_exp = functions[exp]
        return ancestral_eval(new_exp,functions)
    else:
        raise("Expression type unknown.", exp)




def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # TODO insert your code here
    functions = {**env, **graph[1]['P']}

    if type(graph) is list:
        output_exp = graph[2]
        ##print(output_vars)
        if type(output_exp) == list:
            #result_list = [ ancestral_eval(graph[1]['P'][v],functions) for v in output_vars]
            result = ancestral_eval(output_exp,functions)
            print("sample value is:",result)
            return result
        elif type(output_exp) == str:
            #print("evaluating the link function")
            #print(graph[1]['P'][output_exp])
            result = ancestral_eval(graph[1]['P'][output_exp],functions)
            print("sample value is:", result)
            return result
        elif type(output_exp) is int or type(output_exp) is float:
            print("sample value is:", result)
            return torch.tensor(float(output_exp))
        else:
            raise("Expression type unknown.", output_exp)
    elif type(graph) is int or type(graph) is float:
        return torch.tensor(float(graph))
    else:
        raise("Graph expression type unknown.", exp)

    return torch.tensor([0.0, 0.0, 0.0])


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)
        




#Testing:

def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../../cpsc532_hw2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        #print(i)
        #print(graph)
        #print(graph[-1])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        print("truth is:", truth)
        ret = deterministic_eval(graph[-1])
        print("return value is: ", ret)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
    
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    #TODO: 
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(5,7):
        
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../../cpsc532_hw2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        #print(graph)
        #print(graph[0])
        #print(graph[2])
        #print(type(graph[2]))

        #print(sample_from_joint(graph))

        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
        print('Test passed')
    
    print('All probabilistic tests passed')    


def plot_tests():
    num_samples = 10
    samples = []

    num_bins = 10

    graph = daphne(['graph','-i','../../cpsc532_hw2/programs/1.daphne'])
    print('\n\n\nSample of prior of program:')

    for i in range(0,num_samples):
        samples.append(sample_from_joint(graph))
    
    print(samples)

    plt.hist(samples,num_bins,facecolor='blue', alpha=0.5)

    plt.show()



        
        
if __name__ == '__main__':
    
    plot_tests()



    run_deterministic_tests()
    run_probabilistic_tests()


    #graph = daphne(['graph','-i','../../cpsc532_hw2/programs/1.daphne'])
    #print('\n\n\nSample of prior of program:')
    #print(graph)
    #print(sample_from_joint(graph)) 

    #i=2
    #graph = daphne(['graph','-i','../../cpsc532_hw2/programs/{}.daphne'.format(i)])
    #print(graph)
    #print('\n\n\nSample of prior of program {}:'.format(i))
    #print(sample_from_joint(graph)) 

    #for i in range(4,5):
    #    graph = daphne(['graph','-i','../../cpsc532_hw2/programs/{}.daphne'.format(i)])
    #    print(graph)
    #    print('\n\n\nSample of prior of program {}:'.format(i))
    #    print("expression we want to evaluate: ", graph[2])
    #    s = sample_from_joint(graph)

    