from algorithms.base import BaseAlgorithm
class FCFS(BaseAlgorithm):
    def __init__(self) -> None:
        super().__init__()
    
    def make_assignments(self, ready_tasks, cluster):                               
        # ret_tasks = []
        ret_devices = []
        device_condition = [d.lenQ for d in cluster.devices[1:]]
        # device_condition = [d.lenQ for d in cluster.devices[1:]]
        for task in ready_tasks:
            # if len(task.pre_tasks)==1 and task.pre_tasks[0].id==0:
            #     device_id=1
            # elif len(task.suc_tasks)==1 and task.suc_tasks[0].id==-1:
            #     device_id=1
            # if task.id==-1:
            #     device_id=0
            # else:
            #     device_id = device_condition.index(min(device_condition)) + 1
            # if device_id>0:
            #     device_condition[device_id-1] += 1

            device = self.make_assignment(task, cluster, device_condition)
            if device.id>0:
                device_condition[device.id - 1] += 1
        
            ret_devices.append(device)
        return ready_tasks, ret_devices
            
    def make_assignment(self, ready_task, cluster, device_condition):
        
        if ready_task.id==-1:
            device_id = 0
        else:
            device_id = device_condition.index(min(device_condition)) + 1
        return cluster.devices[device_id]