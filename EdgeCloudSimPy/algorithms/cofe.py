from enum import EnumMeta
import enum
from algorithms.base import BaseAlgorithm
inf = 1e28
class COFE(BaseAlgorithm):
    def __init__(self) -> None:
        super().__init__()

    def make_assignment(self, task, cluster):
      
        if task.id == -1:
            return cluster.devices[0]
        else:
            # just_finished_task = cluster.just_finished_task
            # print(just_finished_task.id)
            D = []                                                                      # 待选devices    
            minFT = inf                                                                 # 最小finish time            
            for device in cluster.devices[1:]:
                FT = self.cal_FT(task, cluster, device)                                 # 计算finish time 
                if FT < minFT:
                    D = [device]
                    minFT = FT
                elif FT == minFT:
                    D.append(device)
            if len(D) == 1:
                return D[0]
            else:
                
                if task.is_critical:                                                    # 如果task处在critical path上
                    cid = task.job.critical_path.index(task.id)
                    pre_critical = task.job.tasks[task.job.critical_path[cid-1]]
                    pre_critical_device = pre_critical.device                           # 找到关键路径上前一个任务所在的device
                    if pre_critical_device in D:
                        return pre_critical_device
                
                if D[-1].type=='cloud':
                    D = D[:-1]
                minIT = inf
                result = None                                                           # 找闲置时间最短的device
                for device in D:
                    if device.idle_time < minIT:
                        result = device
                        minIT = device.idle_time
                return result
        
   

    def cal_FT(self, task, cluster, device):
        # H = device.
        transfer_duration = 0
        for pre in task.pre_tasks:
            transfer_duration = max(task.job.dag.feature_graph[pre.id, task.id] * cluster.tr(pre.device, device) / 1e9, transfer_duration)
        estimated_data_ready_time = max(task.estimated_data_ready_time, cluster.env.now + transfer_duration)
        estimated_start_time = max(estimated_data_ready_time, device.EST)
        return estimated_start_time + task.workload / device.capacity             




        


        