import platform
from logging import Logger

import cpuinfo
import psutil
import subprocess


def get_installed_ram():
    try:
        output = subprocess.check_output(
            ["wmic", "MemoryChip", "get", "Capacity"],
            universal_newlines=True
        )
        lines = output.strip().split("\n")[1:]
        total_bytes = sum(int(line.strip()) for line in lines if line.strip().isdigit())
        return round(total_bytes / (1024 ** 3), 2)
    except Exception:
        return "Unknown"


def get_os_version():
    return f"{platform.system()} {platform.release()}"


def log_machine_spec(logger: Logger):
    cpu_info = cpuinfo.get_cpu_info()
    cpu_name = cpu_info['brand_raw']
    cores = psutil.cpu_count(logical=False)
    threads = psutil.cpu_count(logical=True)
    freq = psutil.cpu_freq()
    freq_max = round(freq.max, 0)  # in MHz
    installed_ram = get_installed_ram()
    usable_ram = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    arch = platform.machine()
    os_version = get_os_version()
    python_version = platform.python_version()
    interpreter = platform.python_implementation()

    logger.info("Machine Specifications:")
    logger.info(f"CPU: {cpu_name} ({cores} cores / {threads} threads)")
    logger.info(f"Max CPU frequency: {int(freq_max)} MHz")
    logger.info(f"Installed RAM: {installed_ram} GB")
    logger.info(f"Usable RAM: ~{usable_ram} GB")
    logger.info(f"Architecture: {arch}")
    logger.info(f"Operating System: {os_version}")
    logger.info(f"Python Version: {python_version}")
    logger.info(f"Interpreter: {interpreter}")
