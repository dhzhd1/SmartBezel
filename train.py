import argparse
from model.model_yolo2 import get_model
import mxnet as mx


def load_rec_file(train_rec, val_rec):
    train_iter = mx.io.ImageRecordIter(
        path_imgrec='./datasets/train.rec',
        label_width=5,
        data_name='data',
        label_name='predict_label',
        data_shape=(3, 224, 224),
        batch_size=4,
        shuffle=True
    )

    val_iter = mx.io.ImageRecordIter(
        path_imgrec='./datasets/train.rec',
        label_width=5,
        data_name='data',
        label_name='predict_label',
        data_shape=(3, 224, 224),
        batch_size=4,
        shuffle=True
    )

    # for image_batch in train_iter:
    #     print(image_batch)


    return train_iter, val_iter

def parser_args():
    parser = argparse.ArgumentParser('Training the network')
    parser.add_argument('--gpus', help='Specify GPUs for Training, eg 0,1,2,4,5,6,7', required=False, type=str)
    parser.add_argument('--classes', help='Class number of predictation', required=True, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parser_args()
    ctx = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    sym = get_model(args.classes, train=True)
    train_iter, val_iter = load_rec_file('./datasets/train.rec', './datasets/train.rec')
    mod = mx.mod.Module(context=ctx, symbol=sym, data_names=['data'], label_names=['predict_label'])
    mod.bind(train_iter.provide_data, train_iter.provide_label)
    mod.init_params(mx.init.Xavier(magnitude=2.0))
    mod.optimizer_initialized('sgd', optimizer_params=(('learning_rate', 0.1),('momentum', 0.9), ('wd', 0.0005), ))
    train_metric = mx.metric.create('train acc')
    val_metric = mx.metric.create('val acc')

    mod.fit(train_iter, val_iter, 'acc', num_epoch=100)


if __name__ == "__main__":
    main()
