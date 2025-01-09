import psutil

def physicalCoreCount():
    return psutil.cpu_count(logical=False)

def logicalCoreCount():
    return psutil.cpu_count(logical=True)
