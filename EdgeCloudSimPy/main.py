# from EdgeCloudSimPy.algorithms.random_algo import RandomAlgo
from core.controller import Controller
from core.job import *
from omegaconf import OmegaConf
import numpy as np
from core.device import *
from core.simulator import Simulator
from core.monitor import Monitor
from algorithms.cofe import COFE
from algorithms.fcfs import FCFS
from algorithms.random_algo import RandomAlgo
import simpy
import copy
import time

def generate_jobs(
    job_conf,                                                                   # 任务的设置，在job.yaml里定义
    save_dag = False                                                            # TODO: 增加保存DAG到文件的功能
    ):

    i = 0
    t = 0

    timetable = {}
    while i < job_conf.num_jobs:                                                # 按Possion分布设定每秒到达的app数量
        n = np.random.poisson(job_conf.lambda_scalar)
        # print(f'at time {t}, {n} jobs arrvied.')
        if i+n > job_conf.num_jobs:
            timetable[t] = job_conf.num_jobs - i
        else:
            timetable[t] = n
        i += n
        t += 1


    DAG_conf = job_conf.DAG_config                        
    DAGs = []
    dag_id = 0
    for t in timetable.keys():
        # print(t, timetable[t])
        n = timetable[t]
        for i in range(n):
            
            dag = DAG(DAG_conf)                                                 # 生成DAG
            DAGs.append((dag_id, t, dag))                       
            dag_id +=1

    Jobs = []
    for dag_tuple in DAGs:
        job = Job(dag_tuple[0], dag_tuple[1], dag_tuple[2], job_conf)           # 由DAG和config生成job
        # print(job.id, job.arrive_time, job.sum_workload)
        Jobs.append(job)
    return Jobs

def run_episode(
    raw_jobs,                                                                   # 生成的原始jobs列表
    cluster_conf,                                                               # 集群的设定
    conf,                                                                       # 整体设定
    algo,                                                                       # 算法
    dt                                                                          # 当前时间
    ):
    
    jobs = copy.deepcopy(raw_jobs)                                              # 深拷贝以保证后续过程不影响原始jobs
    env = simpy.Environment()                                                   # simpy环境初始化
    cluster = Cluster(cluster_conf)                                             # 根据config生成集群
    sim = Simulator(env, cluster, jobs, algo)                                   # 搭建模拟

    print(sim.cluster.avg_capacity)                                             # 集群的平均算力

    sim.process()                                                               # 启动模拟进程
    env.run(until = cluster_conf.sim_time)                                      # 一直运行300s，这里可自由定义

    avg_makespan = sum(sim.results[2]) / len(sim.results[3])                    # 平均makespan
    avg_fit = sum(sim.results[-1]) / len(sim.results[-1])                       # 准时率
    print(f'Simulation Done. Algorithm={type(algo).__name__}, Average makespan={avg_makespan}, avf fit rate={avg_fit}')

    sim.save_results(dt, conf)


def main():
    dt = time.strftime("%Y-%m-%d/%H%M%S")
    # generate Jobs
    # job_conf = OmegaConf.load('/home/manson/Workspace/ECSP-master/EdgeCloudSimPy/job.yaml')  
    job_conf = OmegaConf.load('E:\PythonCode\ECSP-master\EdgeCloudSimPy\job.yaml')
                                     # 利用OmegaConf库从文件中生成config。运行的时候需要把这里改成相应的绝对路径
    raw_jobs = generate_jobs(job_conf)
    # cluster_conf = OmegaConf.load('/home/manson/Workspace/ECSP-master/EdgeCloudSimPy/cluster.yaml')
    cluster_conf = OmegaConf.load('E:\PythonCode\ECSP-master\EdgeCloudSimPy\cluster.yaml')
                                     # 运行的时候需要把这里改成相应的绝对路径
    conf = OmegaConf.merge(job_conf, cluster_conf)                              # 合并config

    algo = RandomAlgo()
    run_episode(raw_jobs, cluster_conf, conf, algo, dt)

    algo = FCFS()
    run_episode(raw_jobs, cluster_conf, conf, algo, dt)
    
    algo = COFE()
    run_episode(raw_jobs, cluster_conf, conf, algo, dt)
    
    

    
if __name__ == '__main__':
    main()