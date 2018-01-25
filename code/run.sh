export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
python3.4 main.py -embedding_file ../data/glove.840B.300d.txt  -train_file=../data/data/train -dev_file=../data/data/dev -log_file=../log/log -batch_size=100 -num_epoches=40 -para_limit=400 -learning_rate=0.1
