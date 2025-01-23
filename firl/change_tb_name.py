import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.core.util import event_pb2

# 输入和输出日志文件路径
input_log_path = "/Users/chengping/Desktop/Code/f-IRL-TRRL/TB/AntFH-v0_fkl/events.out.tfevents.1733203006.Broad-AI-2.1599005.0"
output_log_path = "/Users/chengping/Desktop/Code/f-IRL-TRRL/TB/AntFH-v0_fkl/events.out.tfevents.1733203006.Broad-AI-2.1599005.1"

# 打开原始日志文件并创建新的日志文件
with tf.io.TFRecordWriter(output_log_path) as writer:
    for event in summary_iterator(input_log_path):
        for value in event.summary.value:
            # 修改变量名，例如，将 "old_name" 替换为 "new_name"
            if value.tag == "/distance":
                value.tag = "new_name"
        writer.write(event.SerializeToString())
