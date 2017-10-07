import argparse
from model.model_vgg16 import vgg16_symbol
import mxnet as mx


def parser_args():
    parser = argparse.ArgumentParser('Training the network')
    parser.add_argument('--gpus', help='Specify GPUs for Training, eg 0,1,2,4,5,6,7', required=False, type=str)
    parser.add_argument('--classes', help='Class number of predictation', required=True, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parser_args()
    ctx = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    net = vgg16_symbol()
    mod = mx.mod.Module(symbol=net, context=ctx, data_names=['data'], label_names=['softmax_label'])



if __name__ == "__main__":
    main()
