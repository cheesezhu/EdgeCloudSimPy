import numpy as np
import random
import math


import core.device


def generate_dag(
    total_num: int = 10,                                # Nodes num except dummy nodes
    max_od: int = 3,                                    # max out degree
    alpha: float = 1.0,                                 # shape
    beta: float = 1.0,                                  # regularity                    
) -> dict:
    depth = math.floor(math.sqrt(total_num) / alpha)                                    # dag 层数
    mean_width = total_num / depth                                                      # 平均宽度
    widths = np.ceil(np.random.normal(mean_width, beta, size = depth)).astype(int)      # 每层的宽度

    nodes = []                                                                          # 节点 
    generate_num = 0                                                                    # 节点数量
    for i, width in enumerate(widths):
        
        layer = list(range(width))
        nodes.append(layer)
        generate_num += width


    if generate_num != total_num:                                                       # 如果已生成节点与设定总节点数量不一致
        if generate_num < total_num:
            for i in range(total_num - generate_num):
                index = random.sample(range(depth), 1)[0]
                nodes[index].append(len(nodes[index]))
    
        else:
            i = 0
            while i < generate_num - total_num:
                index = random.sample(range(depth), 1)[0]
                if len(nodes[index]) == 1:
                    pass
                else:
                    nodes[index].pop()
                    i+=1

    generate_num = 0
    for i in range(depth):                                                              # 逐层生成节点
        width = len(nodes[i])
        nodes[i] = [nodes[i][_] + generate_num + 1 for _ in range(width)]
        generate_num += width

    # print(nodes)
    nodes = [[0]] + nodes                                                               # 加空首尾
    nodes.append([-1])
    # print(nodes)

    position = {}
    max_pos = 0
    for i, layer in enumerate(nodes):
        pos = 1
        for node in layer:
            position[node] = (3*i, pos)
            pos += 5
        max_pos = max(max_pos, pos)
    position[0] = (0, max_pos/2)
    position[-1] = (3 * (depth+1), max_pos/2)                                           # 用来画图的，目前没用上
    # print(position)

    # -----------------------------------------------------------------------------------------------------
    pre_nodes = [None]+[ [] for _ in range(total_num+1)]                                # 前序节点列表
    suc_nodes = [[] for _ in range(total_num+1)]+[None]                                 # 后序节点列表
    edges = []                                                                          # 边
   
    for i in range(depth-1):
        layer = nodes[i+2]
        for pre_node in nodes[i+1]:
            od = random.randrange(1,max_od+1,1)                                         # 出度
            od = min(od, len(layer))
            sns = random.sample(layer, od)                                              # suc nodes
            sns.sort()
        
            for sn in sns:
              
                edges.append((pre_node, sn))                                            # 连边，添加前序/后序节点列表
                pre_nodes[sn].append(pre_node)
                suc_nodes[pre_node].append(sn)

    for i in range(total_num):                                                          # 没有入度/出度的节点连接空首/尾
        id = len(pre_nodes[i+1])
        od = len(suc_nodes[i+1])
        if  id == 0:    
            edges.append((0, i+1))
            pre_nodes[i+1].append(0)
            suc_nodes[0].append(i+1)
        if od == 0:
            edges.append((i+1, -1))
            suc_nodes[i+1].append(-1)
            pre_nodes[-1].append(i+1)
    
    # print(edges)
    # print(pre_nodes)
    # print(suc_nodes)    
    ret = {                                                                             # 保存dag的节点/边/前序/后序列表
        'nodes': nodes,
        'edges': edges,
        'pre_nodes': pre_nodes,
        'suc_nodes': suc_nodes,
        'position': position
    }
    return ret

def generate_feature(                                                                   # 生成workload和data
    graph: dict,
    total_num: int,
    ccr: float = 0.4,
    bt: float = 0.05,
    bt_std: float = 0.025,
    
    ):
    nodes = graph['nodes']                                                              
    
    edges = graph['edges']

    feature_graph = -np.ones((total_num+2, total_num+2))
    feature_graph[0,0] = feature_graph[-1, -1] = 0
    tct = 0
    for i in range(total_num):
        ct =  abs(np.random.normal(bt, bt_std))
        workload = ct * 1e4
        feature_graph[i+1,i+1] = workload
        tct+=ct
    
    edge_num = len(edges)
    atd = 1.28e8 * tct * ccr / edge_num
    print(tct, atd)

    for edge in edges:
        data = abs(np.random.normal(atd, 0.5*atd))
        feature_graph[edge[0], edge[1]] = feature_graph[edge[1], edge[0]] =data
    
    return feature_graph



    



class DAG(object):
    def __init__(
        self,
        DAG_config,
        ) -> None:
        self.DAG_config =DAG_config
        self.feature_graph, self.pre_nodes, self.suc_nodes = self.generateDAG(DAG_config)
        self.num_tasks = DAG_config.total_num +2

    def generateDAG(self, DAG_config):

        graph = generate_dag(DAG_config.total_num, DAG_config.max_od, DAG_config.alpha, DAG_config.beta)
        feature_graph = generate_feature(graph, DAG_config.total_num, DAG_config.ccr, DAG_config.bt, DAG_config.bt_std)
        return feature_graph, graph['pre_nodes'], graph['suc_nodes']




class Job(object):
    def __init__(
        self,
        jid: int,                                                                           # id
        arrive_time: float,                                                                 # 到达集群的时间
        dag: DAG,                                                                           # DAG
        job_config,                                                                         # config
        ) -> None:
        

        self.env = None                                                                     # simpy 环境
        self.sim = None                                                                     # 模拟实验
        

        self._id = jid                                                                      # 这里用_id作为变量名，外界调用时会使用后面的id()method（是一种规范，但是我没有完全遵照规范）
        self.job_config = job_config
        self.dag = dag
        
        
        self.num_tasks = dag.num_tasks
        self.tasks = self.generate_tasks()                                                  # 生成任务

        self.arrive_time = arrive_time
        self.sum_workload = np.trace(self.dag.feature_graph)                                # 总workload
        self.finish_time = None                                                 
        
        self.arrived = False
        self.finied = False
        self.duration_scalar = self.job_config.duration_scalar                              # 计算ddl时的尺度
        self.duration = None
        self.ddl = None

        self.priority = None
        self._critical_path = None


    @property
    def id(self):
        return self._id

    def attach(self, sim):                                                                  # 将任务连接到仿真中
        self.sim = sim
        self.env = sim.env
        # print(self.num_tasks)
        for task in self.tasks:
            task.attach(sim)

    def generate_tasks(self):
        tasks = []
        for i in range(self.num_tasks):
            id = i if i!=(self.num_tasks-1) else -1
            workload = self.dag.feature_graph[i,i]
            task = Task(self, id, workload)
            tasks.append(task)
            pre_ids = self.dag.pre_nodes[i]
            # print(pre_ids)
            if pre_ids is not None:
                for pre_id in pre_ids:
                    pre = tasks[pre_id]
                    pre.add_suc(task)
                    task.add_pre(pre)
            
        tasks[0].get_ready()
        tasks[0].data_ready = True
        return tasks

    def set_duration(self, cluster):                                                        # 这里设置的是ddl
        self.duration = self.duration_scalar * self.sum_workload / cluster.avg_capacity
        self.ddl = self.arrive_time + self.duration
        
    def cal_priority_and_get_critical_path(self, cluster):                                  # 计算job的优先级列表，得到关键路径
        priority = [0] * self.num_tasks
        for task in self.tasks[::-1]:
            if task.id == -1:
                task.set_priority(0)
                continue
            else:
                P = 0
                ect = task.workload / cluster.avg_capacity
                for suc_task in task.suc_tasks:
                    ett = self.dag.feature_graph[task.id, suc_task.id] * cluster.avg_tr / 1e9

                    theta = ect / ett
                    # print(theta)
                    eta = 0 if 1 - cluster.rho ** (-theta) < random.uniform(0,1) else 1
                    tmp = priority[suc_task.id] + eta * ett + ect
                    if tmp>=P:
                        task.critical_suc = suc_task
                        P = tmp
                priority[task.id] = P
                task.set_priority(P)
        self.priority = priority
        critical_path = []
        cur = self.tasks[0]
        while cur:
            critical_path.append(cur.id)
            cur.set_critical()
            cur = cur.critical_suc
        self._critical_path = critical_path

    @property
    def critical_path(self):
        return self._critical_path

    def done(self):                                                                     # 完成时，保存该job的运行时间
        self.finished = True
        self.finish_time = self.env.now
        self.computing_time = self.finish_time - self.arrive_time
        self.sim.log(
            [
                self.id,
                self.env.now,
                self.computing_time,
                self.duration,
                self.computing_time < self.duration
            ],
            show=False
        )
        # print(f'job {self.id} done. current time {self.env.now}, comsuming time {self.computing_time}, estimated duration {self.duration}')
        # print(self.sim.cluster.idle_time)
    

    def arrive(self):
        self.arrived = True
    
    # def set_deadline(self, deadline):
    #     self.deadline = deadline




    

        
class Task(object):
    def __init__(
        self,
        job: Job,
        task_index: int,
        workload: float,
        ) -> None:
        
        self.job = job
        self._id = task_index
        
        

        self.env = None
        self.sim = None
        self.cluster = None

        self.duration = None                                    # 这里是运行时间
        self.transfer_duration = None                           # 向后继任务的传输时间
        self.device = None                                      # 所在的device
        self.machine = None                                     # 所在的machien
        self.process = None                                     # 运行时的进程
        self.transfer_process = None                            # 向后继任务传data的进程            （这块写得不太好
        self.ready = False                                      # 前序任务全部结束为1
        self.data_ready = False                                 # 前序数据传输全部结束为1
        self.started = False                                    # 到machine上run了为1
        self._finished = False

        self.started_timestamp = None
        self.finished_timestamp = None

        self.pre_tasks = []
        self.pre_data = {}                                      # 前序数据
    
        self.suc_tasks = []

        self.workload = workload
        self.critical_suc = None                                # critical后继任务
        self._priority = None                               
        self._is_critical = False
        self.estimated_data_ready_time = 0
        self.estimated_finish_time = None

    
    

    def add_pre(self, task):
        self.pre_tasks.append(task)
        self.pre_data[task] = False

    def attach(self, sim):
        self.sim = sim
        self.env = sim.env
        self.cluster = sim.cluster
    
    def add_suc(self, task):
        self.suc_tasks.append(task)

    @property
    def finished(self):
        return self._finished
    
    def finish(self):                                                       # 在machine上跑完时的工作
        self._finished = True
        self.cluster.just_finished_task = self
        for task in self.suc_tasks:
            # if task.ready:
            #     continue
            new_ready = True
            for pre in task.pre_tasks:
                new_ready &= pre.finished
            if new_ready:
                # print(f'task {task.job.id} {task.id} is ready!!!')
                task.get_ready()
                self.sim.cluster.add_ready_tasks(task)



    def get_ready(self):
        self.ready = True
        

    def get_data_ready(self, task):                                         # 检测数据是否全部就位
        self.pre_data[task] = True
        data_ready = True
        for val in self.pre_data.values():
            data_ready &= val
        self.data_ready = data_ready
        return self.data_ready

    def allocate(self, device):                                             # 安排自己到device上
        if not self.ready:
            raise ValueError('Task is not ready')
        self.duration = self.workload / device.capacity
        self.estimated_finish_time = device.getEFT(self)
        device.add_task(self)
        
        self.device = device
        
    
    def set_priority(self, P):                                              # 设置自己的优先度
        self._priority = P
    
    def set_critical(self):
        self._is_critical = True

    @property
    def is_critical(self):
        return self._is_critical



    @property
    def priority(self):
        return self._priority
        
            
    @property
    def id(self):                                                           
        return self._id



    def start(self, machine):
        if not self.data_ready:
            raise ValueError('Task data is not ready')
        
        self.started = True
        self.started_timestamp = self.env.now

        self.machine = machine
        
        # print(f'start running:{self.job.id, self.id, self.env.now, self.duration, self.machine.device.id, self.machine.is_running}')
        # print(f'[INFO] | Start running | Task: {self.job.id}-{self.id} | CT: {self.env.now} | Duration: {self.duration} | machine: {self.machine.device.id}-{self.machine.id}-{self.machine.is_running}')
        machine.run_task(self)                                              # 改变machine的状态
        self.process = self.env.process(self.run())                         # 在env开启一个run自己的进程

    def run(self):
        
        yield self.env.timeout(self.duration)
        
        # print(f'tets-{self.duration}-{self.env.now}')
        self.finish()
        self.finished_timestamp = self.env.now
        if self.id==-1:
            self.job.done()
        self.machine.stop_task(self)

    def start_transfer(self, suc_task, suc_device):
        if not self.finished:
            raise ValueError('Pre Task is not done')
        if not suc_task.ready:
            raise ValueError('Suc Task is not ready')
        # print(f'[INFO] | Start transfer | task: {self.job.id}-{self.id}-{suc_task.id} | device: {self.device.id}-{suc_device.id} | current time: {self.env.now}')
        transfer_rate = self.cluster.tr(self.device, suc_device)
        self.transfer_duration = self.job.dag.feature_graph[self.id, suc_task.id] * transfer_rate / 1e9
        suc_task.estimated_data_ready_time = max(suc_task.estimated_data_ready_time, self.env.now + self.transfer_duration)
        # print(transfer_rate, self.transfer_duration, self.duration)
        self.transfer_process = self.env.process(self.transfer(suc_task))

    def transfer(self, suc_task):
        yield self.env.timeout(self.transfer_duration)

        # print('transfer done')

        ret = suc_task.get_data_ready(self)
        if ret:
            suc_task.device.run_the_next()








