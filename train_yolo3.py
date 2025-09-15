import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from YOLOv3 import yolo_body
from PIL import Image
import numpy as np
import cv2

# 定义a-b之间的随机数生成器
def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

# 实时数据增强的随机预处理
def get_random_data(annotation_line, input_shape, random=True, max_boxes=20,
                    jitter=.3, hue=.1, sat=1.5, val=1.5,proc_img=True):
    line = annotation_line.split()  # 以空格分隔标注行
    image = Image.open(line[0])  # 加载图像
    iw, ih = image.size  # 图像宽高
    h, w = input_shape  # 网络输入高宽
    # 分割标注框信息，并转成int后组成ndarray
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    # 数据增强
    # 随机偏移
    # 调整图像大小
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # 计算偏移量
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    # 创建一个新图像，并在偏移位置放置数据图像
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 图像随机左右翻转
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 随机调整HSV
    # 随机产生HSV
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    # 图像转HSV色彩空间
    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    # 调整图像HSV
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val

    # HSV不超过限定范围
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    # 转回RGB色彩空间
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1

    # 校正真实框
    box_data = np.zeros((max_boxes, 5))

    if len(box) > 0:
        np.random.shuffle(box)   # 随机打乱
        # 依据图像resize和随机偏移，校正真实框
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy

        # 依据图像左右翻转，校正真实框
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]

        # 限定真实框不超出图像边界
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h

        # 计算真实框的宽和高
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]

        # 去除无效框
        box = box[np.logical_and(box_w > 1, box_h > 1)]
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data

# ---------------------------------------------------#
#   将预测值的每个特征层调成真实值
# ---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    # [1, 1, 1, num_anchors, 2]
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # 获得x，y的网格
    # (13, 13, 1, 2)
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # (batch_size,13,13,3,85)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 将预测值调成真实值
    # box_xy对应框的中心点
    # box_wh对应框的宽和高
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # 在计算loss的时候返回如下参数
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


# ---------------------------------------------------#
#   用于计算每个预测框与真实框的iou
# ---------------------------------------------------#
def box_iou(b1, b2):
    # 13,13,3,1,4
    # 计算左上角的坐标和右下角的坐标
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 1,n,4
    # 计算左上角和右下角的坐标
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 计算重合面积
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]

    # 计算交并比IOU
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


# ---------------------------------------------------#
#   loss值计算
# ---------------------------------------------------#
def yolo_loss(args, anchors, num_classes, ignore_thresh=.5):
    # 一共有三层
    num_layers = len(anchors) // 3

    # 将预测结果和实际ground truth分开，args是[*model_body.output, *y_true]
    # y_true是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    # yolo_outputs是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    y_true = args[num_layers:]
    yolo_outputs = args[:num_layers]

    # 先验框
    # 678为116,90,  156,198,  373,326
    # 345为30,61,  62,45,  59,119
    # 012为10,13,  16,30,  33,23,
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    # 得到input_shpae为416,416
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))

    # 得到网格的shape为13,13;26,26;52,52
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0

    # 取出每一张图片
    # m的值就是batch_size
    m = K.shape(yolo_outputs[0])[0]
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    # y_true是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    # yolo_outputs是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    for l in range(num_layers):
        # 以第一个特征层(m,13,13,3,85)为例子
        # 取出该特征层中存在目标的点的位置。(m,13,13,3,1)
        object_mask = y_true[l][..., 4:5]
        # 取出其对应的种类(m,13,13,3,80)
        true_class_probs = y_true[l][..., 5:]

        # 将yolo_outputs的特征层输出进行处理
        # grid为网格结构(13,13,1,2)，raw_pred为尚未处理的预测结果(m,13,13,3,85)
        # 还有解码后的xy，wh，(m,13,13,3,2)
        grid, raw_pred, pred_xy, pred_wh = yolo_head(
            yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)

        # 这个是解码后的预测的box的位置
        # (m,13,13,3,4)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # 找到负样本群组，第一步是创建一个数组，[]
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        # 对每一张图片计算ignore_mask
        def loop_body(b, ignore_mask):
            # 取出第b副图内，真实存在的所有的box的参数
            # n,4
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            # 计算预测结果与真实情况的iou
            # pred_box为13,13,3,4
            # 计算的结果是每个pred_box和其它所有真实框的iou
            # 13,13,3,n
            iou = box_iou(pred_box[b], true_box)

            # 13,13,3,1
            best_iou = K.max(iou, axis=-1)

            # 判断预测框的iou小于ignore_thresh则认为该预测框没有与之对应的真实框
            # 则被认为是这幅图的负样本
            ignore_mask = ignore_mask.write(
                b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        # 遍历所有的图片
        _, ignore_mask = tf.while_loop(
            lambda b, *args: b < m, loop_body, [0, ignore_mask])

        # 将每幅图的内容压缩，进行处理
        ignore_mask = ignore_mask.stack()
        # (m,13,13,3,1,1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # 将真实框进行编码，使其格式与预测的相同，后面用于计算loss
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][:] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])

        # object_mask如果真实存在目标则保存其wh值
        # switch接口，就是一个if/else条件判断语句
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(
            raw_true_xy, raw_pred[..., 0:2],from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])

        # 如果该位置本来有框，那么计算1与置信度的交叉熵
        # 如果该位置本来没有框，而且满足best_iou<ignore_thresh，则被认定为负样本
        # best_iou<ignore_thresh用于限制负样本数量
        # 计算是否存在目标的置信度损失
        confidence_loss = object_mask * K.binary_crossentropy(
            object_mask, raw_pred[..., 4:5], from_logits=True) \
                + (1 - object_mask) * K.binary_crossentropy(
            object_mask, raw_pred[..., 4:5],from_logits=True) * ignore_mask
        # 计算类别损失
        class_loss = object_mask * K.binary_crossentropy(
            true_class_probs, raw_pred[..., 5:], from_logits=True)

        # 计算中心点位置偏移平均损失
        xy_loss = K.sum(xy_loss) / mf
        # 计算宽高平均损失
        wh_loss = K.sum(wh_loss) / mf
        # 计算是否存在目标的置信度平均损失
        confidence_loss = K.sum(confidence_loss) / mf
        # 计算类别平均损失
        class_loss = K.sum(class_loss) / mf
        # 总损失求和
        loss += xy_loss + wh_loss + confidence_loss + class_loss

    return loss

# ---------------------------------------------------#
#   获得类别名称
# ---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

# ---------------------------------------------------#
#   获得先验框
# ---------------------------------------------------#
def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


# ---------------------------------------------------#
#   数据生成器
# ---------------------------------------------------#
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)   # 获取标注数据的大小
    i = 0
    while True:     # 循环获取数据
        image_data = []
        box_data = []
        for b in range(batch_size):   # 每次获取一个batch数据
            if i == 0:
                np.random.shuffle(annotation_lines)   # 数据顺序打乱

            # 随机获取一组数据
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            # 图像数据和标注框信息加入列表中
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        # 转成ndarray格式
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        # 预处理真实框
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


# ---------------------------------------------------#
#   读入xml文件，并输出y_true
# ---------------------------------------------------#
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    # 一共有三个特征层数
    num_layers = len(anchors) // 3
    # 先验框
    # 678为116,90,  156,198,  373,326
    # 345为30,61,  62,45,  59,119
    # 012为10,13,  16,30,  33,23,
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')  # 416,416
    # 读出xy轴，读出长宽
    # 中心点(m,n,2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # 计算比例
    true_boxes[..., 0:2] = boxes_xy / input_shape[:]
    true_boxes[..., 2:4] = boxes_wh / input_shape[:]

    # m张图
    m = true_boxes.shape[0]
    # 得到网格的shape为13,13;26,26;52,52
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    # y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1],
                        len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]
    # [1,9,2]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    # 长宽要大于0才有效
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # [n,1,2]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # 计算真实框和哪个先验框最契合
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # 维度是(n)
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    # floor用于向下取整
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    # 找到真实框在特征层l中第b副图像对应的位置
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true

# ----------------------------------------------------#
if __name__ == "__main__":
    # 标签的位置
    annotation_path = '../data/2007_train.txt'
    # 获取classes和anchor的位置
    classes_path = '../data/model_data/voc_classes.txt'
    anchors_path = '../data/model_data/yolo_anchors.txt'
    weights_path = '../data/model_data/yolo_weights.h5'
    # 获得classes和anchor
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    # 一共有多少类
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # 训练后的模型保存的位置
    log_dir = '../tmp/logs'
    # 输入的shape大小
    input_shape = (320, 320)

    # 输入的图像为
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape

    # 创建yolo模型
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)

    # # 载入预训练权重
    # print('Load weights {}.'.format(weights_path))
    # model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # y_true为13,13,3,85
    # 26,26,3,85
    # 52,52,3,85
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    # 输入为*model_body.input, *y_true
    # 输出为model_loss
    loss_input = [*model_body.output, *y_true]
    # 构建模型Loss损失
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes,
                      'ignore_thresh': 0.5})(loss_input)
    # 构建网络模型
    model = Model([model_body.input, *y_true], model_loss)

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    # ------------------------------------------------------#
    freeze_layers = 184
    for i in range(freeze_layers): model_body.layers[i].trainable = False

    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + '/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                monitor='val_loss', save_weights_only=True, save_best_only=False, period=2)
    # 定义学习率衰减策略
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    # 定义训练早停
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    if True:
        # 定义adam优化器
        model.compile(optimizer=Adam(lr=1e-3), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 8  # 定义batch_size大小
        print('Train on {} samples, val on {} samples, '
              'with batch size {}.'.format(num_train, num_val, batch_size))

        # 模型训练
        model.fit_generator(
            data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:],
                            batch_size, input_shape, anchors,num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs=20,
            initial_epoch=0,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + '/trained_weights_stage_1.h5')

    # 对冻结参数进行解冻
    for i in range(freeze_layers): model_body.layers[i].trainable = True

    # 解冻后训练
    if True:
        model.compile(optimizer=Adam(lr=1e-4), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 4  # 定义batch_size大小
        print('Train on {} samples, val on {} samples, '
              'with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:],
                            batch_size, input_shape, anchors,num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs=200,
            initial_epoch=20,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])

        model.save_weights(log_dir + '/last.h5')  # 保存模型