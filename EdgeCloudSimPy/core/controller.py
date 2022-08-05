
from ast import Assign


class RRController(object):
    def __init__(self, controller) -> None:
        self.controller = controller
        self.env = controller.env
        self.destroyed = False
        self.simulator = None
        self.cluster = None


    def attach(self, simulator):
        self.simulator = simulator
        self.cluster = simulator.cluster

    def run(self):
        for job in self.simulator.jobs:
            assert job.arrive_time >= self.env.now
            yield self.env.timeout(job.arrive_time - self.env.now)
            

            self.cluster.add_job(job)
            # print(job.id, self.env.now, job.duration)
            # print(job.priority)
        self.destroyed = True


class AssignmentMaker(object):
    def __init__(
        self,
        controller,
        ) -> None:
        self.controller = controller
        self.env = controller.env
        self.algo = controller.algo
        self.destroyed = False
        self.simulator = None
        self.cluster = None


    def attach(self, simulator):
        self.simulator = simulator
        self.cluster = simulator.cluster

    
    def make_assignments(self):
        new_ready_tasks = self.cluster.ready_tasks

        tasks, devices = self.algo.make_assignments(new_ready_tasks, self.cluster)

        self.cluster.assign(tasks, devices)

        self.cluster.reset_ready_task()


    def sort_and_make_assignment(self):
        ready_tasks = self.cluster.ready_tasks
        Ps = [task.priority for task in ready_tasks]
        # print([task.id for task in ready_tasks])
        # print(Ps)
        Ps_sorted = sorted(enumerate(Ps), key=lambda x:x[1])
        ids  =[x[0] for x in Ps_sorted]
        sorted_ready_tasks = [ready_tasks[id] for id in ids][::-1]
        for task in sorted_ready_tasks:
            device = self.algo.make_assignment(task, self.cluster)
            self.cluster.assign_single(task, device)
        self.cluster.reset_ready_task()



class Controller(object):
    def __init__(
        self,
        env,
        algo,
        ) -> None:
        self.env = env
        self.algo = algo

        self.destroyed = False

        self.cluster = None
        self.simulator = None

        self.rr_controller = RRController(self)
        self.assignment_maker = AssignmentMaker(self)

    
    def attach(self, simulator):
        self.simulator = simulator
        self.cluster = simulator.cluster
        self.rr_controller.attach(simulator)
        self.assignment_maker.attach(simulator)


    
    def process(self):
        self.env.process(self.rr_controller.run())

    
    def make_assignment(self):
        if type(self.algo).__name__=='COFE':
            self.assignment_maker.sort_and_make_assignment()
        else:
            self.assignment_maker.make_assignments()

    
        
       