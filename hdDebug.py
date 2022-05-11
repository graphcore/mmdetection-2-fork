assigner_counter = 0
ipu_mode = False
if False:
    skip_loss = False
    # from torch.nn import MultiheadAttention
    skip_assigner = True
    skip_sampler = False
    skip_loss_cls = False
else:
    skip_loss = False
    skip_assigner = False
    skip_sampler = False
    skip_loss_cls = False


import pickle
import copy
def set_func_io_recorder(save_path, save_rounds):
    # get data out of saved pickle
    # args, kwargs, results = [records[i][ele] for ele in ['args', 'kwargs', 'results']]
    def io_recorder(func):
        records = []
        counter = [0]
        def inner(*args, **kwargs):
            if counter[0] < save_rounds:
                inputs = dict(args=copy.deepcopy(args),
                              kwargs=copy.deepcopy(kwargs))
            results = func(*args, **kwargs)
            if counter[0] < save_rounds:
                inputs['results'] = copy.deepcopy(results)
                records.append(inputs)
                counter[0] += 1
                if counter[0] == save_rounds:
                    with open(save_path, 'wb') as f:
                        pickle.dump(records, f)
                    print('all data recorded')
                    exit()
            return results
        return inner
    return io_recorder
