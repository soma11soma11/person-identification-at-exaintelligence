# import
import os
from PIL import Image
import matplotlib
import numpy as np
import glob
import scipy.misc as scm
import h5py
from PIL import Image
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input,Dense,Convolution2D,Activation,MaxPooling2D,Flatten,merge
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.preprocessing import image as pre_image

# Initialization
tf.python.control_flow_ops = tf
np.random.seed(1217)

cwd = os.getcwd()

#data_dir = "/Users/Soma/Desktop/exaintelligence/person-identification/implementation/gen_data/data/"
data_dir = os.path.join(cwd, "data")

which_game=[]
for f in os.listdir(data_dir):
    if not f.startswith('.'):
        which_game.append(f)


def which_court_fc(game):
    #data_dir = "/Users/Soma/Desktop/exaintelligence/person-identification/implementation/gen_data/data/"
    data_dir = os.path.join(cwd, "data")
    which_court=[]
    for f in os.listdir((data_dir + '/' + game)):
        if not f.startswith('.'):
            which_court.append(f)
    return which_court
player_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

def which_frame_fc(game, court):
    #data_dir = "/Users/Soma/Desktop/exaintelligence/person-identification/implementation/gen_data/data/"
    data_dir = os.path.join(cwd, "data")
    which_frame=[]
    for f in os.listdir((data_dir + '/' + game + '/' + court)):
        if not f.startswith('.'):
            which_frame.append(f)
    return which_frame

def which_player_fc(game, court, frame):
    #data_dir = "/Users/Soma/Desktop/exaintelligence/person-identification/implementation/gen_data/data/"
    data_dir = os.path.join(cwd, "data")
    which_player=[]
    for f in os.listdir((data_dir + '/' + game + '/' + court + '/' + frame)):
        if not f.startswith('.'):
            if os.path.isdir((data_dir + game + '/' + court + '/' + frame + '/' + f)):
                which_player.append(f)
    return which_player
which_player_fc('Data01', 'LEFT_COURT', '46')

#file名の変更
for game in which_game:
    for court in which_court_fc(game):
        for frame in which_frame_fc(game, court):
            data_dir = os.path.join(cwd, "data")
            dirdir = data_dir + '/' + game + '/' + court + '/' + frame + '/'
            files = os.listdir(dirdir)
            for file in files:
                if os.path.isdir(dirdir + file):
                    int_file=int(file)
                    new_name=str(int_file)
                    os.rename(dirdir + file, dirdir + new_name)





def array_maker(game, player, x, y):
    marker = 0
    #content_list = []
    for i, court in enumerate(which_court_fc(game)):
        for j, frame in enumerate(which_frame_fc(game, court)):
            try:

                directory = glob.glob(data_dir + "/" + game + "/" + court + "/" + frame + "/" + player + "/*.png")
                data_image = Image.open(directory[0]).resize((x,y))
                data_array = np.array(data_image)
                data_array = data_array.reshape((1,x,y,3))
                #content_list.append("/" + game + "/" + court + "/" + frame + "/" + player + "/")
                if marker == 0:
                     dataset = data_array

                else:
                     dataset = np.concatenate((dataset,data_array), axis=0)

                marker = 1
                print('YES')
            except:
                print('NOO')

    permutation = np.random.permutation(dataset.shape[0])
    dataset = dataset[permutation,:,:,:]
    dataset = dataset / float(255)


# #test dataにbaslettballのデータを全て入れる。
#     a_validation_data = dataset[0:5] #0,1,2,3,4
#     b_validation_data = dataset[5:10] #5,6,7,8,9
#     a_train_data = dataset[10:15] #10,11,12,13,14
#     b_train_data = dataset[15:20] #15,16,17,18,19
#     a_test_data = dataset[20:20+((dataset.shape[0]-20)/2)]
#     hoge = 20+((dataset.shape[0]-20)/2)
#     b_test_data = dataset[hoge :]


    a_validation_data = dataset[0:2] #0,1
    b_validation_data = dataset[2:4] #2,3
    a_test_data = dataset[4:8] #4,5,6,7
    b_test_data = dataset[8:12] #8,9,10,11
    a_train_data = dataset[11:11+((dataset.shape[0]-11)/2)] #11,12,13,14,15,16,17
    hoge = 11+((dataset.shape[0]-11)/2)
    b_train_data = dataset[hoge :]





    f.create_dataset('c/test/' + player, data=a_test_data, dtype=float)
    f.create_dataset('d/test/' + player, data=b_test_data, dtype=float)
    p.create_dataset('a/train/' + player, data=a_train_data, dtype=float)
    p.create_dataset('b/train/' + player, data=b_train_data, dtype=float)
    p.create_dataset('a/validation/' + player, data=a_validation_data, dtype=float)
    p.create_dataset('b/validation/' + player, data=b_validation_data, dtype=float)

#     print(dataset.shape)
#     print(dataset.shape[0]/10)
#
#     a_validation_data = dataset[0:2] #0,1
#     b_validation_data = dataset[2:4] #2,3
#     a_test_data = dataset[4:8] #4,5,6,7
#     b_test_data = dataset[8:12] #8,9,10,11
#     a_train_data = dataset[11:11+((dataset.shape[0]-11)/2)] #11,12,13,14,15,16,17
#     hoge = 11+((dataset.shape[0]-11)/2)
#     b_train_data = dataset[hoge :]
#
# #トレーニングに歩行者全部。weight作って、バスケの画像は、テストだけ。va
# #
#
#     print(a_validation_data.shape)
#     print(b_validation_data.shape)
#     print(a_test_data.shape)
#     print(b_test_data.shape)
#     print(a_train_data.shape)
#     print(b_train_data.shape)
#
#
#     a_validation.create_dataset( player, data=a_validation_data)
#     b_validation.create_dataset( player, data=b_validation_data)
#     a_test.create_dataset( player, data=a_test_data)
#     b_test.create_dataset( player, data=b_test_data)
#     a_train.create_dataset( player, data=a_train_data)
#     b_train.create_dataset( player, data=b_train_data)




##################################train data##################################################################



def model_def(flag=0, weight_decay=0.0005):
    '''
    define the model structure
    ---------------------------------------------------------------------------
    INPUT:
        flag: used to decide which model structure you want to use
                the default value is 0, which refers to the same structure as paper in the reference
        weight_decay: all the weights in the layer would be decayed by this factor

    OUTPUT:
        model: the model structure after being defined

        # References
        - [An Improved Deep Learning Architecture for Person Re-Identification]
    ---------------------------------------------------------------------------
    '''
    K._IMAGE_DIM_ORDERING = 'tf'
    def concat_iterat(input_tensor):
        input_expand = K.expand_dims(K.expand_dims(input_tensor, -2), -2)
        x_axis = []
        y_axis = []
        for x_i in range(5):
            for y_i in range(5):
                y_axis.append(input_expand)
            x_axis.append(K.concatenate(y_axis, axis=2))
            y_axis = []
        return K.concatenate(x_axis, axis=1)

    def cross_input_sym(X):
        tensor_left = X[0]
        tensor_right = X[1]
        x_length = K.int_shape(tensor_left)[1]
        y_length = K.int_shape(tensor_left)[2]
        cross_y = []
        cross_x = []
        tensor_left_padding = K.spatial_2d_padding(tensor_left,padding=(2,2))
        tensor_right_padding = K.spatial_2d_padding(tensor_right,padding=(2,2))
        for i_x in range(2, x_length + 2):
            for i_y in range(2, y_length + 2):
                cross_y.append(tensor_left_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:]
                             - tensor_right_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:])
            cross_x.append(K.concatenate(cross_y,axis=2))
            cross_y = []
        cross_out = K.concatenate(cross_x,axis=1)
        return K.abs(cross_out)

    def cross_input_asym(X):
        tensor_left = X[0]
        tensor_right = X[1]
        x_length = K.int_shape(tensor_left)[1]
        y_length = K.int_shape(tensor_left)[2]
        cross_y = []
        cross_x = []
        tensor_left_padding = K.spatial_2d_padding(tensor_left,padding=(2,2))
        tensor_right_padding = K.spatial_2d_padding(tensor_right,padding=(2,2))
        for i_x in range(2, x_length + 2):
            for i_y in range(2, y_length + 2):
                cross_y.append(tensor_left_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:]
                             - concat_iterat(tensor_right_padding[:,i_x,i_y,:]))
            cross_x.append(K.concatenate(cross_y,axis=2))
            cross_y = []
        cross_out = K.concatenate(cross_x,axis=1)
        return K.abs(cross_out)

    def cross_input_shape(input_shapes):
        input_shape = input_shapes[0]
        return (input_shape[0],input_shape[1] * 5,input_shape[2] * 5,input_shape[3])

    '''
    model definition begin
    -------------------------------------------------------------------------------
    '''
    if flag == 0:
        print ('now begin to compile the model with the difference between ones and neighbour matrixs.')

        a1 = Input(shape=(128,64,3))
        b1 = Input(shape=(128,64,3))
        share = Convolution2D(20,5,5,dim_ordering='tf', W_regularizer=l2(l=weight_decay))
        a2 = share(a1)
        b2 = share(b1)
        a3 = Activation('relu')(a2)
        b3 = Activation('relu')(b2)
        a4 = MaxPooling2D(dim_ordering='tf')(a3)
        b4 = MaxPooling2D(dim_ordering='tf')(b3)
        share2 = Convolution2D(25,5,5,dim_ordering='tf', W_regularizer=l2(l=weight_decay))
        a5 = share2(a4)
        b5 = share2(b4)
        a6 = Activation('relu')(a5)
        b6 = Activation('relu')(b5)
        a7 = MaxPooling2D(dim_ordering='tf')(a6)
        b7 = MaxPooling2D(dim_ordering='tf')(b6)
        a8 = merge([a7,b7],mode=cross_input_asym,output_shape=cross_input_shape)
        b8 = merge([b7,a7],mode=cross_input_asym,output_shape=cross_input_shape)
        a9 = Convolution2D(25,5,5, subsample=(5,5), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(a8)
        b9 = Convolution2D(25,5,5, subsample=(5,5), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(b8)
        a10 = Convolution2D(25,3,3, subsample=(1,1), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(a9)
        b10 = Convolution2D(25,3,3, subsample=(1,1), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(b9)
        a11 = MaxPooling2D((2,2),dim_ordering='tf')(a10)
        b11 = MaxPooling2D((2,2),dim_ordering='tf')(b10)
        c1 = merge([a11, b11], mode='concat', concat_axis=-1)
        c2 = Flatten()(c1)
        c3 = Dense(500,activation='relu', W_regularizer=l2(l=weight_decay))(c2)
        c4 = Dense(2,activation='softmax', W_regularizer=l2(l=weight_decay))(c3)

        model = Model(input=[a1,b1],output=c4)
        model.summary()

    if flag == 1:
        print ('now begin to compile the model with the difference between both neighbour matrixs.')

        a1 = Input(shape=(128,64,3))
        b1 = Input(shape=(128,64,3))
        share = Convolution2D(20,5,5,dim_ordering='tf', W_regularizer=l2(l=weight_decay))
        a2 = share(a1)
        b2 = share(b1)
        a3 = Activation('relu')(a2)
        b3 = Activation('relu')(b2)
        a4 = MaxPooling2D(dim_ordering='tf')(a3)
        b4 = MaxPooling2D(dim_ordering='tf')(b3)
        share2 = Convolution2D(25,5,5,dim_ordering='tf', W_regularizer=l2(l=weight_decay))
        a5 = share2(a4)
        b5 = share2(b4)
        a6 = Activation('relu')(a5)
        b6 = Activation('relu')(b5)
        a7 = MaxPooling2D(dim_ordering='tf')(a6)
        b7 = MaxPooling2D(dim_ordering='tf')(b6)
        c1 = merge([a7,b7],mode=cross_input_sym,output_shape=cross_input_shape)
        c2 = Convolution2D(25,5,5, subsample=(5,5), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(c1)
        c3 = Convolution2D(25,3,3, subsample=(1,1), dim_ordering='tf',activation='relu', W_regularizer=l2(l=weight_decay))(c2)
        c4 = MaxPooling2D((2,2),dim_ordering='tf')(c3)
        c5 = Flatten()(c4)
        c6 = Dense(10,activation='relu', W_regularizer=l2(l=weight_decay))(c5)
        c7 = Dense(2,activation='softmax', W_regularizer=l2(l=weight_decay))(c6)

        model = Model(input=[a1,b1],output=c7)
        model.summary()

    print ('model definition complete')
    return model


def compiler_def(model, *args, **kw):
    '''
    compile the model after defined
    ---------------------------------------------------------------------------
    INPUT:
        model: model before compiled
        all the other inputs should be organized as the form
                loss='categorical_crossentropy'
        # Example
                model = compiler_def(model_def,
                                     sgd='SGD_new(lr=0.01, momentum=0.9)',
                                     loss='categorical_crossentropy',
                                     metrics='accuracy')
        # Default
                if your don't give other arguments other than model, the default
                config is the example showed above (SGD_new is the identical
                optimizer to the one in reference paper)
    OUTPUT:
        model: model after compiled

        # References
        - [An Improved Deep Learning Architecture for Person Re-Identification]
    ---------------------------------------------------------------------------
    '''

    class SGD_new(SGD):
        '''
        redefinition of the original SGD
        '''
        def __init__(self, lr=0.01, momentum=0., decay=0.,
                     nesterov=False, **kwargs):
            super(SGD, self).__init__(**kwargs)
            self.__dict__.update(locals())
            self.iterations = K.variable(0.)
            self.lr = K.variable(lr)
            self.momentum = K.variable(momentum)
            self.decay = K.variable(decay)
            self.inital_decay = decay

        def get_updates(self, params, constraints, loss):
            grads = self.get_gradients(loss, params)
            self.updates = []

            lr = self.lr
            if self.inital_decay > 0:
                lr *= (1. / (1. + self.decay * self.iterations)) ** 0.75
                self.updates .append(K.update_add(self.iterations, 1))

            # momentum
            shapes = [K.get_variable_shape(p) for p in params]
            moments = [K.zeros(shape) for shape in shapes]
            self.weights = [self.iterations] + moments
            for p, g, m in zip(params, grads, moments):
                v = self.momentum * m - lr * g  # velocity
                self.updates.append(K.update(m, v))

                if self.nesterov:
                    new_p = p + self.momentum * v - lr * g
                else:
                    new_p = p + v

                # apply constraints
                if p in constraints:
                    c = constraints[p]
                    new_p = c(new_p)

                self.updates.append(K.update(p, new_p))
            return self.updates
    all_classes = {
        'sgd_new': 'SGD_new(lr=0.01, momentum=0.9)',
        'sgd': 'SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)',
        'rmsprop': 'RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)',
        'adagrad': 'Adagrad(lr=0.01, epsilon=1e-06)',
        'adadelta': 'Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)',
        'adam': 'Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)',
        'adamax': 'Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)',
        'nadam': 'Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)',
    }
    param = {'optimizer': 'sgd_new', 'loss': 'categorical_crossentropy', 'metrics': 'accuracy'}
    config = ''
    if len(kw):
        for (key, value) in kw.items():
            if key in param:
                param[key] = kw[key]
            elif key in all_classes:
                config = kw[key]
            else:
                print ('error')
    if not len(config):
        config = all_classes[param['optimizer']]
    optimiz = eval(config)
    model.compile(optimizer=optimiz,
              loss=param['loss'],
              metrics=[param['metrics']])
    return model



class NumpyArrayIterator_for_CUHK03(pre_image.Iterator):

    def __init__(self, f, train_or_validation = 'train', flag = 1, image_data_generator = None,
                 batch_size=150, shuffle=True, seed=1217,
                 dim_ordering='default'):

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        self.f = f
        self.length = len(f['a'][train_or_validation].keys())
        self.train_or_validation = train_or_validation
        self.flag = flag
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        super(NumpyArrayIterator_for_CUHK03, self).__init__(3000000, batch_size / 2, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x1 = np.zeros(tuple([current_batch_size * 2] + [128,64,3]))
        batch_x2 = np.zeros(tuple([current_batch_size * 2] + [128,64,3]))
        batch_y  = np.zeros([current_batch_size * 2, 2])

        for i, j in enumerate(index_array):

            k = np.random.randint(self.length)
            while k == 1155:
                k = np.random.randint(self.length)
            ja = np.random.randint(self.f['a'][self.train_or_validation][str(k)].shape[0])
            jb = np.random.randint(self.f['b'][self.train_or_validation][str(k)].shape[0])

            x1 = self.f['a'][self.train_or_validation][str(k)][ja]
            x2 = self.f['b'][self.train_or_validation][str(k)][jb]
            if np.random.rand() > self.flag:
                x1 = self.image_data_generator.random_transform(x1.astype('float32'))
            if np.random.rand() > self.flag:
                x2 = self.image_data_generator.random_transform(x2.astype('float32'))

            batch_x1[2*i] = x1
            batch_x2[2*i] = x2
            batch_y[2*i][1] = 1

            ka,kb = np.random.choice(range(self.length),2)
            while ka == 1155 or kb == 1155:
                ka,kb = np.random.choice(range(self.length),2)

            ja = np.random.randint(self.f['a'][self.train_or_validation][str(ka)].shape[0])
            jb = np.random.randint(self.f['b'][self.train_or_validation][str(kb)].shape[0])

            x1 = self.f['a'][self.train_or_validation][str(ka)][ja]
            x2 = self.f['b'][self.train_or_validation][str(kb)][jb]

            batch_x1[2*i+1] = x1
            batch_x2[2*i+1] = x2
            batch_y[2*i+1][0] = 1
        return [batch_x1,batch_x2], batch_y


class ImageDataGenerator_for_multiinput(pre_image.ImageDataGenerator):

    def flow(self, f, train_or_validation = 'train', flag = 0, batch_size=150, shuffle=True, seed=1217):
        return NumpyArrayIterator_for_CUHK03(f, train_or_validation, flag, self, batch_size=batch_size, shuffle=shuffle, seed=seed)


    def agumentation(self, X, rounds=1, seed=None):

        if seed is not None:
            np.random.seed(seed)

        X = np.copy(X)
        aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
        for r in range(rounds):
            for i in range(X.shape[0]):
                aX[i + r * X.shape[0]] = self.random_transform(X[i])
        X = aX
        print("this is X")
        print(X)
        return X

def test(model,val_or_test='test'):
    c,d = _get_test_data(val_or_test)
    return model.predict_on_batch([c,d])

def cmc(model, val_or_test='test'):

        c,d = _get_test_data(val_or_test)

        def _cmc_curve(model, camera1, camera2, rank_max=50):
            num = camera1.shape[0]
            rank = []
            score = []
            camera_batch1 = np.zeros(camera1.shape)
            for i in range(num):
                for j in range(num):
                    camera_batch1[j] = camera1[i]
                similarity_batch = model.predict_on_batch([camera_batch1, camera2])
                sim_trans = similarity_batch.transpose()
                similarity_rate_sorted = np.argsort(sim_trans[0])
                for k in range(num):
                    if similarity_rate_sorted[k] == i:
                        rank.append(k+1)
                        break
            rank_val = 0
            for i in range(rank_max):
                rank_val = rank_val + len([j for j in rank if i == j-1])
                score.append(rank_val / float(num))
            return np.array(score)

        return _cmc_curve(model,c,d)

def _get_test_data(val_or_test='test'):
    with h5py.File('basketball.h5','r') as ff:
        c = np.array([ff['c'][val_or_test][str(i)][0] for i in player_list])
        d = np.array([ff['d'][val_or_test][str(i)][0] for i in player_list])
        return c,d
        #((100, 128, 64, 3) a,bそれぞれ



#one_epoch=30000
def train(model,weights_name='weights_on_basketball',train_num=100,one_epoch=30000,epoch_num=1,flag_random=None,random_pattern=lambda x:x/2+0.4,flag_train=0,flag_val=1,which_val_data='validation',nb_val_samples=1000):
    with h5py.File('cuhk-03.h5','r') as f:
        Data_Generator = ImageDataGenerator_for_multiinput(width_shift_range=0.05,height_shift_range=0.05)
        Rank1s = []
        for i in xrange(train_num):
            print 'number',i,'in',train_num
            if flag_random:
                rand_x = np.random.rand()
                flag_train = random_pattern(rand_x)
            model.fit_generator(Data_Generator.flow(f,flag = flag_train),one_epoch,epoch_num,validation_data=Data_Generator.flow(f,train_or_validation=which_val_data,flag=flag_val),nb_val_samples=nb_val_samples)
            Rank1s.append(round(cmc(model)[0],2))
            print Rank1s
            model.save_weights('weights/'+weights_name+'_'+str(i)+'.h5')
        return Rank1s



# #one_epoch=30000
# def train(model,weights_name='weights_on_basketball',train_num=100,one_epoch=30000,epoch_num=1,flag_random=None,random_pattern=lambda x:x/2+0.4,flag_train=0,flag_val=1,which_val_data='validation',nb_val_samples=1000):
#     with h5py.File('basketgame_final_kkk.h5','r') as f:
#         Data_Generator = ImageDataGenerator_for_multiinput(width_shift_range=0.05,height_shift_range=0.05)
#         Rank1s = []
#         for i in xrange(train_num):
#             print 'number',i,'in',train_num
#             print('soune')
#             if flag_random:
#                 rand_x = np.random.rand()
#                 flag_train = random_pattern(rand_x)
#             print('huh?')
#             model.fit_generator(Data_Generator.flow(f,flag = flag_train),one_epoch,epoch_num,validation_data=Data_Generator.flow(f,train_or_validation=which_val_data,flag=flag_val),nb_val_samples=nb_val_samples)
#             Rank1s.append(round(cmc(model)[0],2))
#             print Rank1s
#             model.save_weights('weights/'+weights_name+'_'+str(i)+'.h5')
#         return Rank1s





# def train(model,weights_name='weights_on_basket_0_0',train_num=100,one_epoch=30000,epoch_num=1,flag_random=None,random_pattern=lambda x:x/2+0.4,flag_train=0,flag_val=1,which_val_data='validation',nb_val_samples=1000):
#     with h5py.File('basketgame_final_kkk.h5','r') as f:
#         Data_Generator = ImageDataGenerator_for_multiinput(width_shift_range=0.05,height_shift_range=0.05)
#         Rank1s = []
#         for i in range(train_num):
#             print ('number',i,'in',train_num)
#             if flag_random:
#                 print('doko')
#                 rand_x = np.random.rand()
#                 print('koko?')
#                 flag_train = random_pattern(rand_x)
#                 print('hehe')
#             model.fit_generator(tuple(Data_Generator.flow(f,flag=flag_train)),one_epoch,epoch_num,validation_data=Data_Generator.flow(f,train_or_validation=which_val_data,flag=flag_val),nb_val_samples=nb_val_samples)
#             print('syaa')
#             Rank1s.append(round(cmc(model)[0],2))
#             print (Rank1s)
#             model.save_weights('weights/'+weights_name+'_'+str(i)+'.h5')
#         return Rank1s




#trainingデータの個数
number_train = 10
#testデータの個数
number_test = 10
#validationデータの個数
number_validation = 10
#trainingデータにおけるpositiveデータの割合
positive_ratio_train = 0.3
#testデータにおけるpositiveデータの割合
positive_ratio_test = 0.3
#validationデータにおけるpositiveデータの割合
positive_ratio_validation = 0.3
#画像サイズ縦横
x, y = 128, 64
#このディレクトリ下に全試合のデータをおく
#data_dir = "/Users/Soma/Desktop/exaintelligence/person-identification/implementation/gen_data/data/" #それぞれの関数のdata_dirも変えてください
#画像名のテキストファイルが出力されるディレクトリ
#output_dir = "/Users/Soma/Desktop/exaintelligence/person-identification/implementation/gen_data/data/"




output_file = "basketball.h5"
f = h5py.File(output_file,'w')
f.create_group('/c/test')
f.create_group('/d/test')


cuhk_file = "cuhk-03.h5"

with h5py.File(cuhk_file,  "a") as pp:
    for player in player_list:
        pp.__delitem__('/a/train/' + player)
        pp.__delitem__('/b/train/' + player)
        pp.__delitem__('/a/validation/' + player)
        pp.__delitem__('/b/validation/' + player)



p = h5py.File(cuhk_file,'a')

for player in player_list:
    array_maker('Data01', player, 128, 64)

f.close()
p.close()






#array_maker('Data01', '00', '30','10')




#
model = model_def()
model = compiler_def(model)
train(model)
