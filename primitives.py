import torch
import torch.distributions as dist
import math

 

def vector(*args):

    result = [args[i] for i in range(len(args))]
    return result


def sample(distr):
    return distr.sample()


def get(list,ind):
    print(list)
    print("ind ", ind)
    return list[int(ind)]


def put(list,where,what):
    #where = int(where)
    list[int(where)] = float(what)
    return list

def first(lst):
    new = lst.copy()
    return new[0]

def second(lst):
    return lst[1]

def last(lst):
    return lst[-1]

def rest(lst):
    print("\n \n list is", lst)
    new = lst.copy()
    result = new[1:]
    print(result)
    return result

def append(lst,entry):
    print("list is ", list)
    print("entry is ", entry)
    new = lst.copy()
    new.append(entry)
    #print("a is ", a)
    return new
    #return list.append(entry)

def hashmap(*args):
    D = {}
    for i in range(0,len(args)):
        if i%2 == 0:
            D[int(args[i])]=args[i+1]
    return D



def observe(args):
    return args

def iff(boo,if_true,if_false):
    if boo:
        return float(if_true)
    else:
        return float(if_false)


def less_than(x,y):
    if x < y:
        return True
    else:
        return False

def greater_than(x,y):
    if x > y:
        return True
    else:
        return False


def let(var, arg, exp):
    for i in range(0,len(exp)):
        if type(exp[i]) is str:
            if exp[i] == var:
                exp[i] = arg
        if type(exp[i]) is list:
            let(var,arg,exp[i])


def my_discrete(*params):
    if type(params[0]) != torch.Tensor:
        #print(type(params[0]))
        params = torch.Tensor(params)
    return dist.Categorical(params)


def my_sqrt(x):
    #print(type(x))
    if type(x)== int or type(x)== float:
        #return math.sqrt(x)
        return torch.sqrt(torch.tensor(float(x)))
    else:
        return torch.sqrt(x)
