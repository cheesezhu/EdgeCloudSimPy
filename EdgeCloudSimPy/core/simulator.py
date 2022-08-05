from email import header
from omegaconf import OmegaConf
from core.controller import Controller
from core.monitor import Monitor
import core.device
import core.job
import time
import os
import csv
import simpy
# import EdgeCloudSimPy
# class Logger(object):

class Simulator(object):
    def __init__(
        self,
        env: simpy.Environment,
        cluster: core.device.Cluster,
        jobs: core.job.Job,
        algo,
        ) -> None:
        self.env = env
        self.cluster = cluster
        self.jobs = jobs
        
        self.algo = algo
        
        self.monitor = Monitor(self.env, self.cluster, self.jobs)
        self.controller = Controller(self.env, self.algo)
        self.cluster.attach(self)
        self.controller.attach(self)

        self.finished = False
        self.results_name = ['Job ID', 'Current Time', 'Computing Time', 'Estimated Duration', 'Finished In Time']
        self.results = [[] for _ in self.results_name]

        
    def call_assign(self):
        self.controller.make_assignment()


    def run(self):
        while True:
            # if self.cluster.
            yield self.env.timeout(0.2)
            

    def process(self):
        self.env.process(self.run())
        self.controller.process()
        # self.cluster.process()

    
    def log(self, res, show=False):
        str2print = '[INFO] | '
        for i, x in enumerate(res):
            self.results[i].append(x)
            str2print += f'{self.results_name[i]} {str(x)} | '
        if show:
            print(str2print)
    

    def save_results(self, date, conf):
        algo_name = type(self.algo).__name__
        root = os.path.join('./exp', date, algo_name)
        os.makedirs(root, exist_ok=True)
        conf_file = os.path.join(root, 'config.yaml')
        OmegaConf.save(config=conf, f=open(conf_file, 'w+'))

        result_file = os.path.join(root, 'results.csv')
        rows = zip(*self.results)
        # print(list(rows))
        headers = self.results_name
        with open(result_file, 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
       
                writer.writerow(row)


    