import random
import fnmatch
import tensorflow as tf
import numpy as np
import math
import os
import librosa
import threading
from sklearn.externals import joblib


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directories, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for i, directory in enumerate(directories):
        category_id = i + 1
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                files.append((os.path.join(root, filename), category_id))
    return files


def load_generic_audio(directories, sample_rate, table_path):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directories, pattern='*.wav')

#     randomized_files = randomize_files(files)
    random.shuffle(files)
    print(len(files))

    if table_path is not None:
        table = dict()
        for id, (src_file, category_id) in enumerate(files):
            table[id] = src_file

        joblib.dump(table, table_path)
    
    for file_id, (src_file, category_id) in enumerate(files):
        src_audio, _ = librosa.load(src_file, sr=sample_rate, mono=True)
        src_audio = src_audio.reshape(-1, 1)

        yield src_audio, file_id, category_id


def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 dirs,
                 coord,
                 sample_rate,
                 gc_enabled,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=32,
                receptive_field=None,
                fid_enabled=False,
                table_path=None):

        self.dirs = dirs
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.fid_enabled = fid_enabled
        self.threads = []
        self.receptive_field = receptive_field
        self.table_path = table_path

        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size, ['float32'], shapes=[(self.receptive_field + self.sample_size, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])
        
        if fid_enabled:
            self.id_file = tf.placeholder(dtype=tf.int32, shape=())
            self.file_id_queue = tf.FIFOQueue(queue_size, ['int32'], shapes=[()])
            self.file_id_enqueue =  self.file_id_queue.enqueue([self.id_file])

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'], shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])

        # accomodate in our embedding table.
        if self.gc_enabled:
            self.gc_category_cardinality = len(dirs)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)
    
    def dequeue_fid(self, num_elements):
        return self.file_id_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_audio(self.dirs, self.sample_rate, self.table_path)
            
            for src, fid, category_id in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                if self.silence_threshold is not None:
                    # Remove silence
                    src = trim_silence(src[:, 0], self.silence_threshold)
                    src = src.reshape(-1, 1)
                    if src.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(fid))
                        
                src = np.pad(src, [[self.receptive_field, 0], [0, 0]],'constant')

                if self.sample_size:
                    padded_sample_size = self.receptive_field + self.sample_size                    
                    if len(src) > padded_sample_size:
                        while len(src) > self.receptive_field:
                            piece = src[:padded_sample_size, :]
                            if len(piece) < padded_sample_size:
                                n_pad = padded_sample_size - len(piece)
                                piece = np.pad(piece, [[0, n_pad], [0, 0]],'constant')

                            sess.run(self.enqueue, feed_dict={self.sample_placeholder: piece})
                            src = src[self.sample_size:, :]

                            if self.fid_enabled:
                                sess.run(self.file_id_enqueue, feed_dict={self.id_file: fid})

                            if self.gc_enabled:
                                sess.run(self.gc_enqueue, feed_dict={self.id_placeholder: category_id})
                    else:
                        n_pad = padded_sample_size - len(src)
                        src = np.pad(src, [[0, n_pad], [0, 0]],'constant')     
                        sess.run(self.enqueue, feed_dict={self.sample_placeholder: src})

                        if self.fid_enabled:
                            sess.run(self.file_id_enqueue, feed_dict={self.id_file: fid})

                        if self.gc_enabled:
                            sess.run(self.gc_enqueue, feed_dict={self.id_placeholder: category_id})                        
                            
                else:
                    sess.run(self.enqueue, feed_dict={self.sample_placeholder: src})
                    
                    if self.fid_enabled:
                        sess.run(self.file_id_enqueue, feed_dict={self.id_file: fid})
                        
                    if self.gc_enabled:
                        sess.run(self.gc_enqueue, feed_dict={self.id_placeholder: category_id})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads