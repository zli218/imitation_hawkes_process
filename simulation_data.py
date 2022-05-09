import generator
import utils
import random

def random_missing(data,missing_rate):
    for seq in data:
        i=0
        while i<len(seq['nodes']):
            if random.random()<missing_rate:
                del seq['nodes'][i]
                del seq['times'][i]
            else:
                i=i+1
    return data

def type_missing(data,missing_rate):
    for seq in data:
        i=0
        while i<len(seq['nodes']):
            if random.random()<missing_rate[int(seq['nodes'][i])]:
                del seq['nodes'][i]
                del seq['times'][i]
            else:
                i=i+1
    return data

if __name__ == "__main__":


    settings_gen = {
        'dim_process': 5,
        'seed_random': 12345,
        'path_pre_train': True,
        'sum_for_time': 1,#Do we use total intensity for time sampling? 0 -- False; 1 -- True
        'args': None
    }
    settings_gen_seqs = {
        'num_seqs':50,
        'min_len': 100,
        'max_len': 128,
        'max_time':1000
    }
    gen_model = generator.HawkesGen(settings_gen)
    gen_model.gen_seqs(settings_gen_seqs)
    s = gen_model.list_seqs
    data = utils.data_restructure(s)


    data1=data[0:40]
    data2 = data[40:45]
    data3 = data[45:]

    # data1= random_missing(data1,0.5)
    # data2 = random_missing(data2, 0.5)

    data1 = type_missing(data1, [0.5,0.5,0.5,0.5,0.5])
    data2 = random_missing(data2, [0.5,0.5,0.5,0.5,0.5])




