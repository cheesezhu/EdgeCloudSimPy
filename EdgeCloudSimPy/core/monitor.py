# 目前这个文件里的东西没啥用，但是需要保留方便以后修改调用AssignmentMaker的机制。


class DeviceMonitor(object):
    def __init__(
        self,
        env,
        cluster
        ) -> None:
        self.cluster = cluster
        self.env = env

    def run(self):
        pass


class JobMonitor(object):
    def __init__(
        self,
        env,
        jobs,
        ) -> None:
        self.jobs = jobs
        self.env = env



class RequestRaiser(object):
    def __init__(self) -> None:
        pass
        


class Monitor(object):
    def __init__(
        self,
        env,
        cluster,
        jobs
        ) -> None:
        self.env = env
        self.device_monitor = DeviceMonitor(env, cluster)
        self.job_monitor = JobMonitor(env, jobs)
        self.request_raiser = RequestRaiser()

