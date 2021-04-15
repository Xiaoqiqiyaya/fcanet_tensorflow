import math 
import  tensorflow as tf
import math

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y



def  MultiSpectralAttentionLayer(x ,channel , dct_h, dct_w ,reduction=16 , freq_sel_method = 'top2'):
    print("------MultiSpectralAttentionLayer----start")
    n,h,w,c = x.shape
    x_pooled = x
    mapper_x, mapper_y = get_freq_indices(freq_sel_method)
    num_split = len(mapper_x)
    mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
    mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
    y =  MultiSpectralDCTLayer(x_pooled, dct_h, dct_w, mapper_x, mapper_y, channel)
    y = tf.layers.dense(y ,channel//reduction ,activation=tf.nn.relu)
    y = tf.layers.dense(y ,channel)
    y = tf.math.sigmoid(y)
    y = tf.reshape(y,[n,1,1,c])
    y = tf.transpose(y,(0,3,1,2))
    y  = tf.tile(y,(1,1,h,w))
    print("------MultiSpectralAttentionLayer----end")
    y = tf.transpose(y,(0,2,3,1))
    return  x*y



def MultiSpectralDCTLayer(x , height ,width ,mapper_x ,mapper_y ,channel):
    print("------MutilSpectralDCTLaer----start")
    # assert len(mapper_x)==(mapper_y)
    assert channel % len(mapper_x)==0
    num_freq = len(mapper_x)
    weight = get_dct_filter(height ,width ,mapper_x ,mapper_y ,channel)
    print(height)
    print(width)
    x = x*weight
    result = tf.reduce_sum(x, [1,2])
    print("------MutilSpectralDCTLaer----end")
    return result 

def build_filter(pos ,freq ,POS):
    # print("------build_filter----statr")
    pi = tf.constant(math.pi)
    POS = tf.cast(pos,tf.float32)
    freq = tf.cast(freq,tf.float32)
    POS = tf.cast(POS,tf.float32)
    result = tf.math.cos(pi*freq*(pos+0.5)/POS)/tf.math.sqrt(POS)
    # print("------build_filter----end")
    if freq==0:
        return result
    else :
        return result*tf.math.sqrt( tf.cast(2,tf.float32))
    

def get_dct_filter(tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
    print("------get_dct_filter----statr")
    dct_filter =tf.Variable(tf.zeros([channel, tile_size_x, tile_size_y]))
    c_part = channel // len(mapper_x)

    for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[i * c_part: (i+1)*c_part, t_x, t_y]. assign(build_filter(t_x, u_x, tile_size_x) * build_filter(t_y, v_y, tile_size_y))
    dct_filter = tf.transpose(dct_filter,[1,2,0])
    print("------get_dct_filter----end")
    return dct_filter

