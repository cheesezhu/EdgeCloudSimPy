from enum import Enum
from platform import machine
import numpy as np

# inf = 1e28
class MachineFlag(Enum):                                            # 用Enum类实现枚举
    IDLE = 0
    BUSY = 1
    DONE = 2



class TransferFlag(Enum):
    IDLE = 0
    BUSY = 1
    DONE = 2


class Machine(object):
    '''
    Very base mode machine
    Can only process one task each time
    '''
    def __init__(
        self,
        device,
        id,
        machine_config,
        ) -> None:

        self._capacity = machine_config.capacity                    # 算力
        self.device = device                                        # 所在device
        self._id = id

        self.machine_flag = MachineFlag.IDLE                        # 初始化为空闲
        # self.tasks = []
    

    def run_task(self, task):
        # self.tasks.append(task)
        self.machine_flag = MachineFlag.BUSY                        # 变成BUSY状态


    def stop_task(self,task):
        # self.tasks.remove(task)
        # if self.tasks == []:
            # self.machine_flag = MachineFlag.IDLE
        self.machine_flag = MachineFlag.DONE                        # 变成DONE状态
        self.device.postprocess(task)                               # 让所在device进行后处理
    

    def reset(self):
        self.machine_flag = MachineFlag.IDLE                        # 重置为空闲状态

    @property
    def is_running(self):                                           # 看看在没在跑
        return self.machine_flag==MachineFlag.BUSY


    @property
    def id(self):
        return self._id
    
    # @property
    # def running_task(self):
    #     return self.tasks.copy()


    @property
    def capacity(self):
        return self._capacity



    @property 
    def state(self):
        return self.machine_flag


class DataIOPort(object):                                               # 可以别看了，暂时没用上
    def __init__(
        self,
        device,
        dataioport_config
        ) -> None:

        self.limited = dataioport_config.limited
        self.transfering_task = []
        self.transfer_flag = TransferFlag.IDLE
        self.device = device

    
    def start_transfer(self, task):
        if not self.limited:
            self.transfering_task.append(task)
            self.transfer_flag = TransferFlag.BUSY
        else:
            raise NotImplementedError('Not implemented yet, i just want to be laaaaaaazy...')
            # TODO

    def stop_transfer(self, task):
        if self.limited:
            raise NotImplementedError('Not implemented yet, i just want to be laaaaaaazy...')
            # TODO

        else:
            self.transfering_task.remove(task)
            if self.transfering_task == []:
                self.transfer_flag = TransferFlag.IDLE



class Device(object):
    def __init__(
        self,
        id,
        device_config,
        ) -> None:
        
        self.type = device_config.type                                                  # mobile/edge/cloud
        self.device_config = device_config                                              # 设置
        self.machines = []                                                                  
        self._id = id

        self.dataioport = DataIOPort(self, device_config.dataioport_config)

        for i in range(device_config.machine_num):
            machine = Machine(self, i, device_config.machine_config)
            self.machines.append(machine)                                               # 向自己的machine列表里添加
        
        self.ubQ = []                                                                   # task queue not begin
        self.ufQ = []                                                                   # unfinished task queue
        self.fQ = []                                                                    # finished task queue
        self.Q = []                                                                     # task queue
        self.ftQ = []                                                                   # finish time queue
        self.iocache = []                                                               # 没用上
        self.cluster = None                                                             # 所在的集群

        self.idle_time = 0                                                              # 空闲时间
        self.last_finished_time = 0                                                     # 上次跑的时间


    @property
    def capacity(self):
        return self.machines[0].capacity
    
    @property
    def machine_num(self):
        return self.device_config.machine_num


    @property
    def machine_running_states(self):
        ret = []
        for machine in self.machines:
            ret.append(machine.is_running)
        return ret
    

    @property
    def is_running(self):                                                               # 有一个machine在跑就算跑
        mrs = self.machine_running_states
        return(max(mrs))

    @property
    def lenQ(self):
        ret = 0
        mrs = self.machine_running_states
        ret -= sum(mrs)
        ret += len(self.ubQ)
        return ret                                                                      # FCFS里用的，看还有多少任务没开始

    @property
    def is_full(self):
        mrs = self.machine_running_states
        return(min(mrs))                                                                # 是不是跑满了

    def attach(self, cluster):
        self.cluster = cluster
    

    def add_task(self, task):
        self.ftQ.append(task.estimated_finish_time)
        self.ftQ.sort()
        self.Q.append(task)
        self.ufQ.append(task)
        self.ubQ.append(task)
        
    

    def run_the_next(self):                                                             # 跑队列里的下一个任务
        if self.is_full:
            return
        if self.ubQ == []:
            return
        task = self.ubQ[0]
        if not (task.ready and task.data_ready):
            return
        for machine in self.machines:
            if not machine.is_running:
                if self.type=='edge':
                    self.idle_time += self.cluster.env.now - self.last_finished_time
                
                task.start(machine)
                self.ubQ.remove(task)
                break
                

    def update_state(self):                                                             # 没跑的machine从DONE变成IDLE
        for machine in self.machines:
            if not machine.is_running:
                machine.reset()
    
    def call_assign(self):                                                              # 申请assign
        self.cluster.call_assign()
    # def run(self):
    #     while True:
    #         if self.ufQ == []:
    #             yield self.cluster.env.timeout(0.001)
    #         else:
    #             if not self.is_full:
    #                 self.run_the_next()
   

    @property
    def EST(self):                                                                      # 新任务来的期望开始时间
        if self.type == 'mobile':
            return self.cluster.env.now
        elif self.type == 'edge':
            if self.ufQ == []:
                return self.cluster.env.now
            else:
                return self.ftQ[-1]
        else:
            if len(self.ufQ) < 4:
                return self.cluster.env.now
            else:
                return self.ftQ[-4]


    def getEFT(self, task):                                                             # 新任务来的期望结束时间
        if self.type == 'mobile':
            return self.cluster.env.now
        else:
            estimated_start_time = max(self.EST, task.estimated_data_ready_time)
    
            return estimated_start_time + task.duration



    

    @property
    def id(self) -> int:
        return self._id


 


    def postprocess(self, task):
        # print(task, self.ufQ, self.id)
        if self.type=='edge':
            self.last_finished_time = self.cluster.env.now
        self.fQ.append(task)
        self.ufQ.remove(task)
        # print(self.Q, self.ufQ, self.fQ)
        # self.iocache.append(task)
        self.call_assign()

        self.run_the_next()
        self.update_state()



class Cluster(object):
    def __init__(
        self,
        cluster_config,
        ) -> None:
        self.sim = None
        self.env = None
        self.cluster_config = cluster_config
        self.edge_num = cluster_config.edge_num
        self.rho = cluster_config.rho
        self.device_num = self.edge_num+2

        self.devices = []
        mobile = Device(0, cluster_config.mobile)
        mobile.attach(self)
        self.devices.append(mobile)
        for i in range(self.edge_num):
            ed = Device(i+1, cluster_config.edge)
            ed.attach(self)
            self.devices.append(ed)
        cloud = Device(self.edge_num+1, cluster_config.cloud)
        cloud.attach(self)
        self.devices.append(cloud)


        connection_config = cluster_config.connection_config
        self.connections = [[None for _ in range(self.device_num)] for _ in range(self.device_num)]
        self.tr_me = connection_config.tr_me
        self.tr_ee = connection_config.tr_ee
        self.tr_ce = connection_config.tr_ce

        self.ts_me = 1e9 / self.tr_me                                                   # 没用上
        self.ts_ee = 1e9 / self.tr_ee
        self.ts_ce = 1e9 / self.tr_ce

        for i in range(self.device_num):
            self.connections[i][i] = 0
        self.connections[0][1] = self.connections[1][0] = self.tr_me
        for i in range(1, self.device_num-1):
            self.connections[0][i] = self.connections[i][0] = self.tr_me + self.tr_ee
            for j in range(i+1, self.device_num-1):
                self.connections[i][j] = self.connections[j][i] = self.tr_ee
            self.connections[i][self.device_num-1] = self.connections[self.device_num-1][i] = self.tr_ce
        self.connections[0][self.device_num-1] = self.connections[self.device_num-1][0] = self.tr_me + self.tr_ee
        sum_tr = 0
        sum_lines = 0
        for i in range(self.device_num):                                                # 计算平均transfer rate
            for j in range(i, self.device_num):
                sum_lines+=1
                sum_tr += self.connections[i][j]
        self._avg_tr = sum_tr / sum_lines 

        print(self._avg_tr)
        
        # self._avg_tr = (self.ts_me + self.ts_ee * self.edge_num * (self.edge_num - 1) + self.ts_ce * self.edge_num) / (1 + self.edge_num * self.edge_num)
        
        self.jobs =[]
        
        self._ready_tasks = []
        # self.just_finished_task = None
        


    def tr(self, device1:Device, device2:Device) -> float:                              # 两个device之间的transfer rate
        id1, id2 = device1.id, device2.id
        rate = self.connections[id1][id2]

        # print(id1, id2, rate)
        return rate 

    def add_job(self, job):                                                             # 向集群添加job
        job.set_duration(self)
        job.arrive()
        job.cal_priority_and_get_critical_path(self)
        job.attach(self.sim)
        
        self.jobs.append(job)
    
        job.tasks[0].allocate(self.devices[0])
        self.devices[0].run_the_next()

    def attach(self, sim):
        self.sim = sim
        self.env = sim.env

    def call_assign(self):
        # print('check')
        # print([str(task.job.id) + '-' + str(task.id) for task in self.ready_tasks])
        self.sim.call_assign()
        

    def assign(self, tasks, devices):                                                   # assign一堆任务
        
        for i, task in enumerate(tasks):
            # print(task.job.id, task.id, devices[i].id, task.pre_tasks)
            task.allocate(devices[i])
            for pre_task in task.pre_tasks:
                # print(pre_task.id)
                pre_task.start_transfer(task, devices[i])
            
    def assign_single(self, task, device):                                              # assign一个任务
        task.allocate(device)
        for pre_task in task.pre_tasks:
            # print(pre_task.id)
            pre_task.start_transfer(task, device)

    def reset_ready_task(self):                                                         # 重置ready task
        self._ready_tasks = []
        # self.just_finished_task = None

    @property
    def ready_tasks(self):
        return self._ready_tasks

    def add_ready_tasks(self, task):
        # ls = []
        # for job in self.jobs:
        #     for task in job.tasks:
        #         if task.isready
        self._ready_tasks.append(task)
        
    
    @property
    def avg_tr(self):
        return self._avg_tr
    
    @property
    def avg_capacity(self):
        return self.sum_capacity / self.machine_num
    
    @property
    def sum_capacity(self):
        sc = 0 
        for device in self.devices:
            sc += device.capacity * device.machine_num
    
        return sc
    
    @property
    def machine_num(self):
        mn = 0
        for device in self.devices:
            mn += device.machine_num
        return mn
    


    @property
    def idle_time(self):
        ls =[]
        for device in self.devices[1:-1]:
            ls.append(device.idle_time)
        return ls
    # def run(self):
    #     while not self.sim.finished:
    #         for device in self.devices:
    #             device.run()
    
    # def process(self):
    #     for device in self.devices:
    #         self.env.process(device.run())