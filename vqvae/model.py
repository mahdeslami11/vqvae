import sys
import os
import tensorflow as tf
import numpy as np
import librosa
import math

sys.path.append('../wavenet')
from wavenet.model import WaveNetModel
from utils import inv_mu_law_numpy, mu_law_numpy, mu_law, inv_mu_law


def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer()
    variable = tf.get_variable(name, shape, initializer=initializer)
    return variable

def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    variable = tf.get_variable(name, shape, initializer=initializer)
    return variable

def create_embedding_table(name, shape):
    if shape[0] == shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=shape[0], dtype=np.float32)
        variable = tf.get_variable(name, initializer=initial_val)
        return variable
    else:
        return create_variable(name, shape)

    
class VQVAE:
    def __init__(self,
                 batch_size=None, sample_size=None, q_factor=1, n_stack=2, max_dilation=10, K=512, D=128,
                 lr=0.001, use_gc=False, gc_cardinality=None, is_training=True, global_step=None,
                 scope='params', residual_channels=256, dilation_channels=512, skip_channels=256, use_biases=False,
                upsampling_method='deconv'):

        assert sample_size is not None
        assert q_factor == 1 or (q_factor % 2) == 0

        self.filter_width = 2
        self.dilations = [2 ** i for j in range(n_stack) for i in range(max_dilation)]
        self.receptive_field = (self.filter_width - 1) * sum(self.dilations) + 1
        self.receptive_field += self.filter_width - 1

        self.q_factor = q_factor
        self.quantization_channels = 256 * q_factor

        self.K = K
        self.D = D  
        self.use_gc = use_gc
        self.gc_cardinality = gc_cardinality
        self.use_biases = use_biases

        # encoding spec
        self.encode_level = 6

        # model spec
        self.upsampling_method = upsampling_method
        self.is_training = is_training
        self.train_op = None
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.reduced_timestep = None
        self.initialized = False
        if batch_size is not None and sample_size is not None:
            self.reduced_timestep = int(np.ceil(self.sample_size / 2 ** self.encode_level))
            self.initialized = True

        # etc
        self.global_step = global_step
        self.lr = lr

        
        with tf.variable_scope(scope) as params:
            self.enc_var, self.enc_scope = self._encoder()
            with tf.variable_scope('decoder') as dec_param_scope:
                
                self.deconv_var = self.create_deconv_variables()
                self.wavenet = WaveNetModel(batch_size=batch_size,
                        dilations=self.dilations,
                        filter_width=self.filter_width,
                        residual_channels=residual_channels,
                        dilation_channels=dilation_channels,
                        quantization_channels=self.quantization_channels,
                        skip_channels=skip_channels,
                        global_condition_channels=gc_cardinality,
                        global_condition_cardinality=gc_cardinality,
                        use_biases=use_biases)
                
                self.dec_scope = dec_param_scope
                
            with tf.variable_scope('embed'):
                self.embeds = tf.get_variable(
                    'embedding', [self.K, self.D], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.1))        
            
        self.param_scope = params
        self.saver = None
        
        self.set_saver()
        
    def create_deconv_variables(self):
        var = None
        if self.upsampling_method.startswith('deconv'):
            var = list()
            
            tokens = self.upsampling_method.split('-')
            n_step = tokens[0].split('deconv')[1]

            out_channel = int(tokens[1]) if len(tokens) > 1 else 1

            if not n_step:
                n_step = 1
            else:
                n_step = int(n_step)

            assert n_step < 4

            height, width = self.reduced_timestep, self.D
            upscale_factor = 2 ** self.encode_level

            if n_step == 1:
                upscale_per_step = upscale_factor
            elif n_step == 2:
                upscale_per_step = int(np.sqrt(upscale_factor))
            elif n_step == 3:
                upscale_per_step = int(np.cbrt(upscale_factor))

            h = height
            in_channel = 1
            for step in range(n_step):
                with tf.variable_scope('deconv_layer_{}'.format(step)):
                    layer = dict()

                    h *= upscale_per_step

                    kernel_size = 2*upscale_per_step - upscale_per_step%2

                    layer['filter'] = create_variable('deconv_layer_filter', [kernel_size, 1, out_channel, in_channel])
                    layer['strides'] = [1, upscale_per_step, 1, 1]
                    layer['shape'] = [1, h, width, out_channel]
                    if self.use_biases:
                        layer['bias'] = create_bias_variable('deconv_bias', [n_channel])
                    var.append(layer)  
                    
                    in_channel = out_channel
                    out_channel = out_channel * 2
        return var
                    
    def initialize(self, input_batch, sample_size=40960):
        # TODO
        self.batch_size = tf.shape(input_batch)[0]
        self.sample_size = sample_size
        self.reduced_timestep = int(np.ceil(self.sample_size / 2 ** self.encode_level))
        self.initialized = True

    def set_saver(self):
        if self.saver is None:
            save_vars = {('train/' + '/'.join(var.name.split('/')[1:])).split(':')[0]: var for var in
                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.param_scope.name)}
#             for name,var in save_vars.items():
#                 print(name)
            self.saver = tf.train.Saver(var_list=save_vars, max_to_keep=10)
            
    def one_hot(self, ml_encoded):
        return self.wavenet._one_hot(ml_encoded)
    
    def one_hot_gen(self, ml_encoded):
        encoded = tf.one_hot(ml_encoded, self.quantization_channels)
        encoded = tf.reshape(encoded, [-1, self.quantization_channels])
        return encoded

    def _gc_embedding(self):
        return create_embedding_table('gc_embedding', [self.gc_cardinality, self.gc_cardinality])
    
    def _encoder(self):
        with tf.variable_scope('enc') as enc_param_scope:           
            var = dict()

            input_channel = 1
            output_channel = 32
            
            var['enc_conv_stack'] = list()
            for i in range(self.encode_level):
                with tf.variable_scope('encoder_conv_{}'.format(i)):
                    current = dict()
                    current['filter'] = create_variable('filter', [4, 4, input_channel, output_channel])   
                    input_channel = output_channel
                    output_channel = output_channel // 2
                    var['enc_conv_stack'].append(current)
        return var, enc_param_scope

    def upsampling(self, z_q):
        dec_input = tf.expand_dims(z_q, -1)
        
        if self.deconv_var is not None:
            for i in range(len(self.deconv_var)):
                layer = self.deconv_var[i]
                dec_input = tf.nn.conv2d_transpose(
                    dec_input,
                    layer['filter'],
                    layer['shape'],
                    layer['strides'],
                    padding='SAME',
                    data_format='NHWC',
                    name=None
                )
                if self.use_biases:
                    dec_input += layer['bias']
                
                if i < len(self.deconv_var)-1:
                    dec_input = tf.nn.relu(dec_input)
                
            dec_input = tf.reduce_sum(dec_input, -1)
        else:
            dec_input = tf.image.resize_nearest_neighbor(dec_input, [self.sample_size, self.D])
            dec_input = tf.squeeze(dec_input, axis=-1, name='dec_input_squeeze')
        return dec_input

    def encode(self, encoded_input_batch, dtype=tf.float32):
        encoded_input_batch = tf.expand_dims(encoded_input_batch, -1)
     
        out = encoded_input_batch

        for i in range(self.encode_level):
            kernel = self.enc_var['enc_conv_stack'][i]['filter']

            if i < self.q_factor:
                out = tf.nn.conv2d(out, kernel, [1, 2, 2, 1], padding='SAME')
            else:
                out = tf.nn.conv2d(out, kernel, [1, 2, 1, 1], padding='SAME')

            if i < (self.encode_level-1):
                out = tf.nn.relu(out)
        
#         out = tf.layers.batch_normalization(out, training=self.is_training)
        out = tf.nn.sigmoid(out)
        # enc_shape = [self.batch_size, self.reduced_timestep, self.D]
        z_e = tf.squeeze(out, axis=-1, name='encode_squeeze')       

        return z_e

    def vq(self, z_e):
        _e = tf.reshape(self.embeds, [1, self.K, self.D])
        _e = tf.tile(_e, [self.batch_size, self.reduced_timestep, 1])

        _t = tf.tile(z_e, [1, 1, self.K])
        _t = tf.reshape(_t, [self.batch_size, self.reduced_timestep * self.K, self.D])

        dist = tf.norm(_t - _e, axis=-1)
        dist = tf.reshape(dist, [self.batch_size, -1, self.K])
        k = tf.argmin(dist, axis=-1)
        z_q = tf.gather(self.embeds, k)

        return z_q
    
    def get_condition(self, input_batch, gc=None):
        with tf.variable_scope('forward'):
            encoded_input_batch, gc = self.preprocess(input_batch, gc=gc)
            self.encoded_input_batch = encoded_input_batch
            self.gc = gc

            # encoding
            z_e = self.encode(encoded_input_batch)

            # VQ-embedding
            z_q = self.vq(z_e)

            # decoding
            lc = self.upsampling(z_q)          
        return lc, gc

    def create_model(self, padded_input, gc=None, fid=None):
        if fid is not None:
            fid = tf.Print(fid, [fid], message="fid:")
        with tf.variable_scope('forward'):
            
            padded_encoded_input, gc = self.preprocess(padded_input, gc=gc)
            self.gc = gc
            
            # Cut off the last sample of network input to preserve causality.
            wavenet_input_width = tf.shape(padded_encoded_input)[1] - 1
            wavenet_input = tf.slice(padded_encoded_input, [0, 0, 0],
                                     [-1, wavenet_input_width, -1])            
            
            encoded_input = tf.slice(padded_encoded_input, 
                                     [0, self.receptive_field, 0], 
                                     [-1, -1, -1], name="remove_pad")
        
            self.encoded_input = encoded_input
            
            # encoding
            self.z_e = self.encode(encoded_input)

            # VQ-embedding
            self.z_q = self.vq(self.z_e)

            # decoding
            lc = self.upsampling(self.z_q)
            self.lc = lc
            
            paddings = tf.constant([[0, 0], [self.receptive_field - 1, 0], [0, 0]])
            lc = tf.pad(lc, paddings, "CONSTANT")
            
            output = self.wavenet._create_network(wavenet_input, lc, gc)

        return output
    
    def generate_waveform(self, sess, n_samples, lc, gc, seed=None, use_randomness=True):
        sample_placeholder = tf.placeholder(tf.int32)
        lc_placeholder = tf.placeholder(tf.float32)
        gc_placeholder = tf.placeholder(tf.float32)
        next_sample_probs = self.wavenet.predict_proba_incremental(sample_placeholder,
                                                                         lc_placeholder,
                                                                         gc_placeholder)
        sess.run(self.wavenet.init_ops)

        operations = [next_sample_probs]
        operations.extend(self.wavenet.push_ops)
        
        waveform = [128] * (self.receptive_field - 2)
        if seed is None:
            seed = 128
            
        waveform.append(seed)

        for sample in waveform[:-1]:
            lc_sample = np.zeros((1, 128))
            sess.run(operations, feed_dict={sample_placeholder: [sample],
                                                    lc_placeholder: lc_sample,
                                                    gc_placeholder: gc})
        lc = lc.reshape(n_samples, -1)
        for i in range(n_samples):
            if i > 0 and i % 10000 == 0:
                print("Generating {} of {}.".format(i, n_samples))
                sys.stdout.flush()

            sample = waveform[-1]
            lc_sample = lc[i, :].reshape(1, -1)
            results = sess.run(operations, feed_dict={sample_placeholder: sample,
                                                              lc_placeholder: lc_sample,
                                                              gc_placeholder: gc})
            if use_randomness:
                sample = np.random.choice(np.arange(self.quantization_channels), p=results[0])
            else:
                sample = np.argmax(results[0]).reshape(-1)

            waveform.append(sample)

        waveform = np.array(waveform[self.receptive_field:])      
        return waveform

    def _one_hot_encode(self, input_batch, mode="force"):
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(input_batch, depth=self.quantization_channels) 
            if mode == "force":
                encoded = tf.reshape(encoded, [self.batch_size, -1, self.quantization_channels])
            else:
                encoded = tf.reshape(encoded, [-1, self.quantization_channels])

        return encoded
    
    def preprocess(self, input_batch, gc=None):
        if not self.initialized:
            self.initialize(input_batch)
        
        encoded = mu_law(input_batch, quantization_channels=self.quantization_channels)
        encoded = self._one_hot_encode(encoded)
        
        # gc-embedding
        if self.use_gc and gc is not None:
            gc_embedding_table = self._gc_embedding()
            gc = tf.nn.embedding_lookup(gc_embedding_table, gc)
            gc = tf.reshape(gc, [self.batch_size, 1, self.gc_cardinality], name="gc_embbedding_resize")

        return encoded, gc
    
    def loss_recon(self, mu_law_output, encoded_target, beta=0.25):
        encoded_output = self._one_hot_encode(mu_law_output)
        
        output = encoded_output
        target = encoded_target
        
        target = tf.slice(target, [0, 1, 0], [-1, -1, -1], name="loss_recon_slice_target")
        recon = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target)
        recon = tf.reduce_mean(recon)
        
        return recon
        
    def loss(self, output, beta=0.25):
        recon = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.encoded_input)
        recon = tf.reduce_mean(recon)
        
        z_q = self.z_q
        z_e = self.z_e

        vq = tf.reduce_mean(tf.norm(tf.stop_gradient(z_e) - z_q, axis=-1) ** 2, axis=[0, 1])
        commit = tf.reduce_mean(tf.norm(z_e - tf.stop_gradient(z_q), axis=-1) ** 2, axis=[0, 1])
        
        loss = (recon + vq + beta * commit)
        
        if self.is_training:
            with tf.variable_scope('backward'):
                # Decoder Grads
                decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.dec_scope.name)
                decoder_grads = list(zip(tf.gradients(loss, decoder_vars), decoder_vars))

                # Encoder Grads
                encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.enc_scope.name)
                grad_z = tf.gradients(recon, z_q)
                encoder_grads = [(tf.gradients(z_e, _var, grad_z)[0] + beta * tf.gradients(commit, _var)[0], _var)
                                 for _var in encoder_vars]

                # Embedding Grads
                embed_grads = list(zip(tf.gradients(vq, self.embeds), [self.embeds]))

                optimizer = tf.train.AdamOptimizer(self.lr)
                self.train_op = optimizer.apply_gradients(decoder_grads + encoder_grads + embed_grads, global_step=self.global_step)

        
        return loss, recon

        
    def load(self, sess, model):
        self.saver.restore(sess, model)

    def save(self, sess, logdir, step):
        model_name = 'model.ckpt'
        checkpoint_path = os.path.join(logdir, model_name)
        print('Storing checkpoint to {} ...'.format(logdir), end="")
        sys.stdout.flush()

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self.saver.save(sess, checkpoint_path, global_step=step)
        print(' Done.')


class VoiceConverter:
    def __init__(self, model, checkpoint_path=None,
                 batch_size=1, sample_size=40960, sample_rate=16000, session_config=None):

        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.silence_threshold = 0.0

        self.model = model

        self.input_batch = tf.placeholder(tf.float32)
        self.gc_batch = tf.placeholder(tf.int32)

        self.lc, self.gc = model.get_condition(self.input_batch, self.gc_batch)

        self.session = tf.Session(config=session_config)

        if checkpoint_path is not None:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.session.run(init_op)
            print('Loading checkpoint: %s' % checkpoint_path)
            self.model.load(self.session, checkpoint_path)

    def close(self):
        tf.reset_default_graph()
        self.sess.close()

    def get_condition(self, src, gc):
        n_sample = len(src)
        if n_sample <= self.sample_size:
            n_frame = 1
            n_padding = self.sample_size - n_sample
        else:
            n_frame = int(math.ceil(float(n_sample) / self.sample_size))
            n_padding = self.sample_size * n_frame - n_sample

        src = np.pad(src, (0, n_padding), 'constant', constant_values=(0, 0))
        src = src.reshape(-1, 1)

        gcs = [gc] * self.batch_size

        inputs = np.split(src, n_frame)

        inputs = np.hstack(inputs)

        padded_n_frame = np.ceil(float(n_frame) / self.batch_size).astype(np.int32) * self.batch_size
        n_padding = padded_n_frame - n_frame
        padding = np.zeros((self.sample_size, n_padding))
        inputs = np.hstack([inputs, padding])
        inputs = inputs.T

        result = []
        for input in inputs:
            lc = self.session.run(self.lc, feed_dict={self.input_batch: input, self.gc_batch: gcs})
            result.append(lc.reshape(self.sample_size, -1))

        lc = np.vstack(result)

        return lc

    def convert(self, gc, src=None, file=None, use_randomness=True):
        assert src is not None or file is not None
        if src is None:
            src, _ = librosa.load(file, sr=self.sample_rate, mono=True)
        
        src = mu_law_numpy(src.reshape(-1))

        n_samples = len(src)
        src = src[:n_samples]

        a_lc = self.get_condition(src, gc)

        a_gc = self.session.run(self.gc, feed_dict={self.gc_batch: gc})
        a_gc = a_gc.reshape(1, -1)

        waveform = self.model.generate_waveform(self.session, n_samples, a_lc, a_gc, 
                                                seed=src[0], use_randomness=use_randomness)
        result = inv_mu_law_numpy(waveform, quantization_channels=self.model.quantization_channels)
        result = result.reshape(-1)
        return result
    
        
