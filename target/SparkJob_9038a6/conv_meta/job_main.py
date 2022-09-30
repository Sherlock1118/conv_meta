from aibrain_job.job.spark_job_builder import SparkJobBuilder
import logging
logging.basicConfig(level=logging.INFO)
import sys

def main():
    master = {"cpu": 0.4, "memory": 2048, "num": 1}
#     ps = {"cpu": 0.4, "memory": 2048, "num": 1}
#     worker = {"cpu": 0.4, "memory": 2048, "num": 4}
    tf_job_on_k8s = SparkJobBuilder(source_root="./",
                                         main_file="./remote_main.py",
                                         tensorboard=True,
                                         master=master,
                                         job_name="conv_meta_model_train"
                                   )
    tf_job_on_k8s.run()


if __name__ == '__main__':
    main()