'''
config.py

Object that holds model hyper-parameters
'''
import os
import json

class Config():

    def __init__(self, is_new, path, **kwargs):
        if is_new:
            self.experiment_name = kwargs.get('experiment_name', 'default')
            self.basepath = path
            # TODO: this - self.checkpoints = self.basepath+'/weights/'
            if not os.path.exists(self.basepath):
                os.mkdir(self.basepath)
                #TODO: this - os.mkdir(self.checkpoints)
            self.learning_rate = kwargs.get('learning_rate', 0.01)
            self.n_iters = kwargs.get('n_iters', 5)
            self.write_params(self.basepath+'/params.json')
        else:
            fname = os.path.join(path, 'params.json')
            self.load_params(fname)

    def load_params(self, filename):
        with open(filename) as f:
            data = json.load(f)
            for key in data.keys():
                self.__dict__[key] = data[key]

    def write_params(self, filename):
        dict_to_write = {}
        for key in self.__dict__.keys():
            dict_to_write[key] = self.__dict__[key]

        with open(filename, 'w') as f:
            json.dump(dict_to_write, f)
