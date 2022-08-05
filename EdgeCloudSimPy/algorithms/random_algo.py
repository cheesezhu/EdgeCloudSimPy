
# from torch import rand
from algorithms.base import BaseAlgorithm

import random
class RandomAlgo(BaseAlgorithm):
    def __init__(self) -> None:
        super().__init__()

    def make_assignments(self, ready_tasks, cluster):
        # print(ready_tasks)
        # ret_tasks = []
        ret_devices = []
        device_num = cluster.edge_num + 2
        for task in ready_tasks:
            # print(task.id, task.pre_tasks, task.suc_tasks)
            # if len(task.pre_tasks)==1 and task.pre_tasks[0].id==0:
            #     device_id=1
            # elif len(task.suc_tasks)==1 and task.suc_tasks[0].id==-1:
            #     device_id=1
            # if task.id==-1:
            #     device_id=0
            # else:
            #     device_id = random.randint(1, device_num - 1)
            device = self.make_assignment(task, cluster, device_num)

            
            ret_devices.append(device)
        return ready_tasks, ret_devices
    

    def make_assignment(self, ready_task, cluster, device_num):
        if ready_task.id==-1:
            device_id = 0
        else:
            device_id = random.randint(1,device_num-1)
        device = cluster.devices[device_id]
        return device

