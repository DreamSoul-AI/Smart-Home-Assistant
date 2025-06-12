# Smart-Home-Assistant

修复 config.py

读取 config.yml 时改为绝对路径，解决找不到文件的问题。

训练 tokenizer

复制 env.csv 到 data/SmartHome/raw/hh105/ 并修正 env_path 的解析逻辑。

取消对原始 .txt 预处理的依赖

注释掉 dataset.Preprocess(...)，直接用已有的 preprocessed 数据。

SmartHome 数据集类（smarthome.py）

新增 batchify/process_chunk 逻辑，生成序列数据并保存。

加载数据时允许自动重新生成缺失样本。

解决输入维度错误
hyper.py 将
  'SmartHome': [cfg['embedding_size'], cfg['max_length']]
改为
  'SmartHome': [cfg['batch_size'], 384, 3]。
LSTM 取 data_shape[-1] = 3 作为 input_size。
只预测传感器值
forward 中 target_steps = input['data'][:, seq_len:seq_len+pred_len, 1:2]。
decoder 输出维度由 hidden→1。

修复评估指标
metric.py 中 “MSE” 改为直接比较 predicted_steps 与 target。

清理大量调试打印，只保留形状打印或全部注释。
