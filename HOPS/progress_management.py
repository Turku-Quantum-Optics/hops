import multiprocessing
import threading

from rich import progress
from rich.progress import TaskID

class ProgressBar:
    total: float
    completed: float = 0
    taskID: TaskID

    _nextUpdate: float = 0

    def __init__(self, total, taskID, _progress, _queue):
        self.total = total
        self.taskID = taskID
        self._progress = _progress
        self._queue = _queue

    def update(self, completed):
        self.completed = completed
        if self.completed >= self._nextUpdate:
            self._nextUpdate = self.completed + self.total * 0.01
            if self._nextUpdate >= self.total:
                self._nextUpdate = self.total
            #self._queue.put_nowait([self.taskID, self.completed])
            self._progress[self.taskID] = self.completed
    
    def complete(self):
        self.completed = self.total
        #self._queue.put([self.taskID, -1])
        self._progress[self.taskID] = -1

class ProgressManager:
    def update_progress_with_queue(self):
        while not self._sem.acquire(timeout=0.1):
            while not self._queue.empty():
                item = self._queue.get_nowait()
                task_id = item[0]
                latest = item[1]
                isVisible = latest >= 0
                self._progress.update(task_id, completed=latest, visible=isVisible)
        while not self._queue.empty():
            item = self._queue.get_nowait()
            task_id = item[0]
            latest = item[1]
            isVisible = latest >= 0
            self._progress.update(task_id, completed=latest, visible=isVisible)

        for task in self._progress.tasks:
            if not task.visible:
                self._progress.remove_task(task.id)

    def update_progress_with_dict(self):
        idsToRemove = []
        while not self._sem.acquire(timeout=0.1):
            for task_id, completed in self._progressDict.items():
                isVisible = completed >= 0
                self._progress.update(task_id, completed=completed, visible=isVisible)
                if not isVisible:
                    idsToRemove.append(task_id)
            while len(idsToRemove) > 0:
                self._progressDict.pop(idsToRemove.pop())
        for task_id, completed in self._progressDict.items():
                isVisible = completed >= 0
                self._progress.update(task_id, completed=completed, visible=isVisible)
        
        for task in self._progress.tasks:
            if not task.visible:
                self._progress.remove_task(task.id)

    def addProgress(self, description: str, total: float | None = 100, completed: int = 0, visible: bool = True, start: bool = True, **fields) -> ProgressBar:
        assert(self._processManager is not None)
        taskID = self._progress.add_task(description, start=start, total=total, completed=completed, visible=visible, fields=fields)
        return ProgressBar(total, taskID, self._progressDict, self._queue)

    def __enter__(self):
        self._progress = progress.Progress("[progress.description]{task.description}",
                                           progress.BarColumn(),
                                           "[progress.percentage]{task.percentage:>3.0f}%",
                                           progress.TimeRemainingColumn(),
                                           progress.TimeElapsedColumn(),
                                           refresh_per_second=10).__enter__()
        self._processManager = multiprocessing.Manager().__enter__()
        self._progressDict = self._processManager.dict()
        self._sem = threading.Semaphore(value=0)
        self._queue = self._processManager.Queue(0)
        self._progressUpdateThread = threading.Thread(target=self.update_progress_with_dict)
        self._progressUpdateThread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._sem.release()
        self._progressUpdateThread.join()
        self._sem = None
        self._processManager.__exit__(exc_type, exc_value, exc_tb)
        self._progress.__exit__(exc_type, exc_value, exc_tb)
        self._processManager = None
        self._progressDict = None
        self._queue = None
        self._progressUpdateThread = None

def long_running_fn_2(progress: ProgressBar):
        for n in range(0, 100):
            progress.update(n)
        progress.complete()
        return progress.taskID, "string"

def _testWithProgressManager():

    trajectories = 1024
    with ProgressManager() as manager:
        mainProgress = manager.addProgress("[green]Simulating:", total=trajectories)
        args = []
        for n in range(0, trajectories):
            subProgress = manager.addProgress(f"Trajectory {n + 1}", total=100, visible=False)
            args.append(subProgress)
            
        results = []
        with multiprocessing.Pool() as p:
            for index, res in enumerate(p.imap(long_running_fn_2, args, 1)):
                trajectory = index + 1
                mainProgress.update(trajectory)
                results.append(res)
    return results

if __name__ == "__main__":
   res = _testWithProgressManager()