import datetime
import time
import subprocess


class WaitTimeProcessor:

    def __init__(self, wait_time):
        self.wait_time = wait_time

    def process(self, data):
        time.sleep(self.wait_time)
        return data


class WaitIfGpuHot:
    def __init__(self, max_temp=80, check_time=1, wait_if_hot_time=5):
        self.wait_if_hot_time = wait_if_hot_time
        self.check_time = check_time
        self.max_temp = max_temp
        self.last_checked_time = datetime.datetime.now()

    def process(self, data):
        if (datetime.datetime.now() - self.last_checked_time).total_seconds() > self.check_time:
            self.last_checked_time = datetime.datetime.now()
            temp = self.get_gpu_temperature()
            if temp >= self.max_temp:
                time.sleep(self.wait_if_hot_time)
        return data

    def get_gpu_temperature(self):
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if result.returncode == 0:
                temperature = int(result.stdout.strip())
                return temperature
            else:
                print("Error running nvidia-smi:")
                print(result.stderr)
                return None
        except Exception as e:
            print("An error occurred:", str(e))
            return None


