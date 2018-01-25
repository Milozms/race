import os
import logging
import pickle
import sys
import os
import time
import config
import numpy as np
import utils
from model import Model
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def gen_examples(articles, questions, options, answers, labels, batch_size, concat=False):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = utils.get_minibatches(len(articles), batch_size)
    all_ex = []
    for minibatch in minibatches:
        if len(minibatch) < batch_size:
            break
        mb_art = [articles[t] for t in minibatch for k in range(4)]
        mb_que = [questions[t] for t in minibatch for k in range(4)]
        mb_opt = [options[t * 4 + k] for t in minibatch for k in range(4)]
        mb_ans = [answers[t] for t in minibatch for k in range(4)]
        mb_labels = [labels[t * 4 + k] for t in minibatch for k in range(4)]
        all_ex.append((mb_art, mb_que, mb_opt, mb_ans, mb_labels))
    return all_ex


def eval_acc(model, sess, sample):
    sample_num = 0
    truecount = 0
    for idx, minibatch in tqdm(enumerate(sample)):
        arti_batch = np.array(minibatch[0], dtype=np.int32)
        ques_batch = np.array(minibatch[1], dtype=np.int32)
        opt_batch = np.array(minibatch[2], dtype=np.int32)
        ans_batch = np.array(minibatch[3], dtype=np.int32)
        label_batch = np.array(minibatch[4], dtype=np.int32)
        scores = sess.run(model.get_score(),
                      feed_dict={model.article_input: arti_batch,
                                 model.question_input: ques_batch,
                                 model.option_input: opt_batch,
                                 model.labels_input: label_batch})
        # print(arti_batch.shape, ques_batch.shape, opt_batch.shape, ans_batch.shape, label_batch.shape)
        cur_batch_size = int(len(minibatch[0])/4)
        sample_num += cur_batch_size
        for i in range(cur_batch_size):
            max_score = 0
            max_choice = 0
            for k in range(4):
                if scores[i * 4 + k] > max_score:
                    max_score = scores[i * 4 + k]
                    max_choice = k
            if max_choice == ans_batch[i * 4]:
                truecount += 1
    acc = float(truecount/sample_num)
    logging.info('Accuracy: %f\n' % acc)
    print('Accuracy: %f\n' % acc)
    return acc





def main(args):
    logging.info('-' * 50)
    logging.info('Load data files..')
    question_belong = []
    if args.debug:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_data(args.train_file, 100, relabeling=args.relabeling)
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, 100, relabeling=args.relabeling, question_belong=question_belong)
    else:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_data(args.train_file, relabeling=args.relabeling)
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, args.max_dev, relabeling=args.relabeling, question_belong=question_belong)

    args.num_train = len(train_examples[0])
    args.num_dev = len(dev_examples[0])

    logging.info('-' * 50)
    logging.info('Build dictionary..')
    # word_dict = utils.build_dict(train_examples[0] + train_examples[1] + train_examples[2] + dev_examples[0] + dev_examples[1] + dev_examples[2], args.max_vocab_size)
    word_dict = pickle.load(open("../obj/dict.pkl", "rb"))
    # pickle.dump(word_dict, open("../obj/dict.pkl", "wb"))
    logging.info('-' * 50)
    # embeddings = utils.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
    embeddings = pickle.load(open("../obj/embedding.pkl", "rb"))
    # pickle.dump(embeddings, open("../obj/embedding.pkl", "wb"))
    (args.vocab_size, args.embedding_size) = embeddings.shape
    logging.info('Define computation graph..')
    model_train = Model(config=args, word_mat=embeddings)
    # config: batch_size, hidden_size, emb_keep_prob, keep_prob, para_limit, ques_limit, grad_clip
    logging.info('Done.')
    logging.info('-' * 50)
    logging.info(args)

    logging.info('-' * 50)
    '''
    logging.info('Intial test..')
    dev_x1, dev_x2, dev_x3, dev_y = utils.vectorize(dev_examples, word_dict, sort_by_len=not args.test_only, concat=args.concat)
    word_dict_r = {}
    word_dict_r[0] = "unk"
    assert len(dev_x1) == args.num_dev
    all_dev = gen_examples(dev_x1, dev_x2, dev_x3, dev_y, args.batch_size, args.concat)
    # dev_acc, pred = eval_acc(test_fn, all_dev)
    logging.info('Dev accuracy: %.2f %%' % dev_acc)
    best_acc = dev_acc
    if args.test_only:
        return
    utils.save_params(args.model_file, all_params, epoch=0, n_updates=0)
    '''
    # Training
    logging.info('-' * 50)
    logging.info('Start training..')
    train_x1, train_x2, train_x3, train_y, train_label = utils.vectorize(train_examples, word_dict, concat=args.concat, config=args)
    dev_arti, dev_ques, dev_opt, dev_ans, dev_label = utils.vectorize(dev_examples, word_dict, concat=args.concat, config=args)
    assert len(train_x1) == args.num_train
    start_time = time.time()
    n_updates = 0

    all_train = gen_examples(train_x1, train_x2, train_x3, train_y, train_label, args.batch_size, args.concat)
    all_dev = gen_examples(dev_arti, dev_ques, dev_opt, dev_ans, dev_label, args.batch_size, args.concat)

    with tf.Session() as sess:
        maxacc = 0.2
        # saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
        writer = tf.summary.FileWriter(args.graph_dir)
        writer.add_graph(sess.graph)
        lr = args.learning_rate
        sess.run(tf.assign(model_train.is_train, tf.constant(True, dtype=tf.bool)))
        # sess.run(tf.assign(model_train.lr, tf.constant(lr, dtype=tf.float32)))
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.num_epoches):
            logging.info('Epoch: %d\n' % epoch)
            # np.random.shuffle(all_train)
            for idx in tqdm(range(len(all_train))):
                minibatch = all_train[idx]
                arti_batch = np.array(minibatch[0], dtype=np.int32)
                ques_batch = np.array(minibatch[1], dtype=np.int32)
                opt_batch = np.array(minibatch[2], dtype=np.int32)
                label_batch = np.array(minibatch[4], dtype=np.int32)
                # print(arti_batch.shape, ques_batch.shape, opt_batch.shape, label_batch.shape)

                global_step = sess.run(model_train.global_step) + 1
                loss, train_op, summary_str, debug_output = sess.run([model_train.loss, model_train.train_op,
                                                                              model_train.merged_summary_op,
                                                                              model_train.debug_output],
                                          feed_dict={model_train.article_input: arti_batch,
                                                     model_train.question_input: ques_batch,
                                                     model_train.option_input: opt_batch,
                                                     model_train.labels_input: label_batch})
                for i in range(len(debug_output)):
                    # print(model_train.debug_output_name[i], debug_output[i])
                    with open('../npsave/epoch'+str(epoch)+model_train.debug_output_name[i], 'wb') as f:
                        np.save(f, debug_output[i])
                logging.info('batch %d, loss = %f' % (idx, loss))
                print('\nbatch %d, loss = %f' % (idx, loss))
                writer.add_summary(summary_str, epoch*len(all_train)+idx)

            acc = eval_acc(model_train, sess, all_dev)
            if acc>maxacc:
                maxacc = acc
                saver.save(sess, '../obj/models/model0.ckpt')


            '''
            if idx % 100 == 0:
                logging.info('#Examples = %d, max_len = %d' % (len(mb_x1), mb_x1.shape[1]))
                logging.info('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' % (epoch, idx, len(all_train), train_loss, time.time() - start_time))
            n_updates += 1

            if n_updates % args.eval_iter == 0:
                samples = sorted(np.random.choice(args.num_train, min(args.num_train, args.num_dev),
                                                  replace=False))
                sample_train = gen_examples([train_x1[k] for k in samples],
                                            [train_x2[k] for k in samples],
                                            [train_x3[k * 4 + o] for k in samples for o in range(4)],
                                            [train_y[k] for k in samples],
                                            args.batch_size, args.concat)
                
                acc, pred = eval_acc(test_fn, sample_train)
                logging.info('Train accuracy: %.2f %%' % acc)
                dev_acc, pred = eval_acc(test_fn, all_dev)
                logging.info('Dev accuracy: %.2f %%' % dev_acc)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    logging.info('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                 % (epoch, n_updates, dev_acc))
                    utils.save_params(args.model_file, all_params, epoch=epoch, n_updates=n_updates)
            '''
    try:
        writer.close()
    except:
        pass

if __name__ == '__main__':
    # os.system('export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64')
    os.environ["CUDA_VISIBLE_DEVICES"] = ''  # 指定第一块GPU可用
    tfconfig = tf.ConfigProto()
    # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
    tfconfig.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.Session(config=tfconfig)
    args = config.get_args()
    np.random.seed(args.random_seed)

    if args.train_file is None:
        raise ValueError('train_file is not specified.')

    if args.dev_file is None:
        raise ValueError('dev_file is not specified.')

    # if args.rnn_type == 'lstm':
    #     args.rnn_layer = lasagne.layers.LSTMLayer
    # elif args.rnn_type == 'gru':
    #     args.rnn_layer = lasagne.layers.GRULayer
    # else:
    #     raise NotImplementedError('rnn_type = %s' % args.rnn_type)

    if args.embedding_file is not None:
        # dim = utils.get_dim(args.embedding_file)
        dim = 300
        # print('dim = %d' % dim)
        if (args.embedding_size is not None) and (args.embedding_size != dim):
            raise ValueError('embedding_size = %d, but %s has %d dims.' %
                             (args.embedding_size, args.embedding_file, dim))
        args.embedding_size = dim
    elif args.embedding_size is None:
        raise RuntimeError('Either embedding_file or embedding_size needs to be specified.')

    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    logging.info(' '.join(sys.argv))
    main(args)
