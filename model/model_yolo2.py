import mxnet as mx

def yolo2_symbol(data, num_classes):
    # data = mx.sym.Variable('data')

    # conv1
    conv1_1 = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=32, name="conv1_1")
    relu1_1 = mx.sym.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    pool1 = mx.sym.Pooling(data=relu1_1, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool1")

    # conv2
    conv2_1 = mx.sym.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv2_1")
    relu2_1 = mx.sym.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    pool2 = mx.sym.Pooling(data=relu2_1, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool2")

    # conv3
    conv3_1 = mx.sym.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv3_1")
    relu3_1 = mx.sym.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.sym.Convolution(data=relu3_1, kernel=(1, 1), num_filter=64, name="conv3_2")
    relu3_2 = mx.sym.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.sym.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv3_3")
    relu3_3 = mx.sym.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.sym.Pooling(data=relu3_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool3")

    # conv4
    conv4_1 = mx.sym.Convolution(data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv4_1")
    relu4_1 = mx.sym.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.sym.Convolution(data=relu4_1, kernel=(1, 1), num_filter=128, name="conv4_2")
    relu4_2 = mx.sym.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.sym.Convolution(data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv4_3")
    relu4_3 = mx.sym.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.sym.Pooling(data=relu4_3, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool4")

    # conv5
    conv5_1 = mx.sym.Convolution(data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.sym.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.sym.Convolution(data=relu5_1, kernel=(1, 1), num_filter=256, name="conv5_2")
    relu5_2 = mx.sym.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.sym.Convolution(data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    relu5_3 = mx.sym.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    conv5_4 = mx.sym.Convolution(data=relu5_3, kernel=(1, 1), num_filter=256, name="conv5_4")
    relu5_4 = mx.sym.Activation(data=conv5_4, act_type="relu", name="relu5_4")
    conv5_5 = mx.sym.Convolution(data=relu5_4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_5")
    relu5_5 = mx.sym.Activation(data=conv5_5, act_type="relu", name="relu5_5")
    pool5 = mx.sym.Pooling(data=relu5_5, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool5")

    # conv6
    conv6_1 = mx.sym.Convolution(data=pool5, kernel=(3, 3), pad=(1, 1), num_filter=1024, name="conv6_1")
    relu6_1 = mx.sym.Activation(data=conv6_1, act_type="relu", name="relu6_1")
    conv6_2 = mx.sym.Convolution(data=relu6_1, kernel=(1, 1), num_filter=512, name="conv6_2")
    relu6_2 = mx.sym.Activation(data=conv6_2, act_type="relu", name="relu6_2")
    conv6_3 = mx.sym.Convolution(data=relu6_2, kernel=(3, 3), pad=(1, 1), num_filter=1024, name="conv6_3")
    relu6_3 = mx.sym.Activation(data=conv6_3, act_type="relu", name="relu6_3")
    conv6_4 = mx.sym.Convolution(data=relu6_3, kernel=(1, 1), num_filter=512, name="conv6_4")
    relu6_4 = mx.sym.Activation(data=conv6_4, act_type="relu", name="relu6_4")
    conv6_5 = mx.sym.Convolution(data=relu6_4, kernel=(3, 3), pad=(1, 1), num_filter=1024, name="conv6_5")
    relu6_5 = mx.sym.Activation(data=conv6_5, act_type="relu", name="relu6_5")

    # conv7
    conv7_1 = mx.sym.Convolution(data=relu6_5, kernel=(3, 3), pad=(1, 1), num_filter=num_classes, name="conv7_1")
    relu7_1 = mx.sym.Activation(data=conv7_1, act_type="relu", name="relu7_1")
    pool7 = mx.sym.Pooling(data=relu7_1, global_pool=True, pool_type='avg', kernel=(7,7), name='pool7')

    # softmax
    softmax = mx.sym.SoftmaxOutput(data=pool7, name="softmax")

    return softmax


def get_model(num_class, train=True):
    # extract the last conv layer from the network
    # num_class in training is training class, in feature extract is 2, (face or not)
    if train:
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('predict_label')
        rois = mx.sym.Variable('rois')
        bbox_target = mx.sym.Variable('bbox_target')
        bbox_weight = mx.sym.Variable('bbox_weight_1')

        #reshape input
        rois = mx.sym.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')
        label = mx.sym.Reshape(data=label, shape=(-1, 5), name='label_reshape')
        bbox_target = mx.sym.Reshape(data=bbox_target, shape=(-1, 4 * num_class), name='bbox_target_reshape')
        bbox_weight = mx.sym.Reshape(data=bbox_weight, shape=(-1, 4 * num_class), name='bbox_weight_reshape')
    else:
        data = mx.sym.Variable('data')
        rois = mx.sym.Variable('rois')
        rois = mx.sym.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')

    # Get shared Conv Layer
    model = yolo2_symbol(data, num_class)
    feature_layer = model.get_internals()['relu6_5_output']
    # per yolo v2, it says polynomial rate decay with a power of 4.0
    conv_new_1 = mx.sym.Convolution(data=feature_layer, kernel=(1, 1), num_filter=1024, name="conv_new_1", lr_mult=4.0)
    relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu_new_1')

    # cls/bbox
    cls = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7 * 7 * num_class, name="cls")
    bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7 * 7 * 4 * num_class,
                                   name="bbox")
    roipooled_cls_rois = mx.sym.ROIPooling(name='roipooled_cls_rois', data=cls, rois=rois,
                                           pooled_size=(7, 7), spatial_scale=0.0625)
    roipooled_loc_rois = mx.sym.ROIPooling(name='roipooled_loc_rois', data=bbox, rois=rois,
                                           pooled_size=(7, 7), spatial_scale=0.0625)
    cls_score = mx.sym.Pooling(name='ave_cls_scores_rois', data=roipooled_cls_rois, pool_type='avg', global_pool=True,
                               kernel=(7, 7))
    bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=roipooled_loc_rois, pool_type='avg', global_pool=True,
                               kernel=(7, 7))
    cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_class))
    bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_class))


    if train:
        cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid',
                                        grad_scale=1.0)
        bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
        bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / 128)

        # The '2' in below, is TRAIN.BATCH_IMAGES number
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(2, -1, num_class),
                                  name='cls_prob_reshape')
        bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(2, -1, 4 * num_class),
                                   name='bbox_loss_reshape')
        group = mx.sym.Group([cls_prob, bbox_loss])

    else:
        cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
        # 1 IN BELOW IS TEST.BATCH_IMAGES
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(1, -1, num_class),
                                  name='cls_prob_reshape')
        bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(1, -1, 4 * num_class),
                                   name='bbox_pred_reshape_2')
        group = mx.sym.Group([cls_prob, bbox_pred, feature_layer])

    return group



if __name__ == '__main__':
    model = get_model(10, True)
    print(model)
    print(model.get_internals())
    print("Network Argument:")
    print(model.list_arguments())
    print("")
    print("Network Output:")
    print(model.list_outputs())


    model2 = get_model(10, False)
    print (model2)
    print(model2.get_internals())
    print("Network Argument:")
    print(model2.list_arguments())
    print("")
    print("Network Output:")
    print(model2.list_outputs())