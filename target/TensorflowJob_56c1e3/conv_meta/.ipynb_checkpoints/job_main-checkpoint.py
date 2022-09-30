from aibrain_job.job.spark_job_builder import SparkJobBuilder
from aibrain_job.job.tensorflow_job_builder import TensorflowJobBuilder
import logging
logging.basicConfig(level=logging.INFO)
import sys

def main():
    master = {"cpu": 0.4, "memory": 61440, "num": 1, "gpu_memory": 10}
    ps = {"cpu": 0.4, "memory": 20480, "num": 1, "gpu_memory": 10}
    worker = {"cpu": 0.4, "memory": 20480, "num": 4, "gpu_memory": 10}
    tf_job_on_k8s = TensorflowJobBuilder(source_root="./",
                                         main_file="./remote_main.py",
                                         tensorboard=True,
                                         master=master,
                                         ps=ps,
                                         worker=worker,
                                         job_name="conv_meta_model_train"
                                   )
    tf_job_on_k8s.run()


if __name__ == '__main__':
    main()