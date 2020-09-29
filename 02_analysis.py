import os
import numpy as np
from music21 import converter
import random as rd
from models.MuseGAN import MuseGAN
from utils.loaders import load_music
import tensorflow as tf

import matplotlib.pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("start!")
#===========
music_num = rd.randint(1,35)
music_num = str(music_num)
#===============
#run para
SECTION = 'musik'
RUN_ID = '4444'  #3333 : 64
DATA_NAME = 'npz'
part_name = '0928' #music_num
RUNNAME='musik'
FILENAME = '1-35_444' +'.npz'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, RUNNAME])

#================
###data load
BATCH_SIZE = 64#64:3333, 40:4444
n_bars = 10
n_steps_per_bar = 16
n_pitches = 84
n_tracks = 4

data_binary, data_ints, raw_data = load_music(DATA_NAME, FILENAME, n_bars, n_steps_per_bar)
data_binary = np.squeeze(data_binary)

gan = MuseGAN(input_dim = data_binary.shape[1:]
        , critic_learning_rate = 0.001
        , generator_learning_rate = 0.001
        , optimiser = 'adam'
        , grad_weight = 10
        , z_dim = 1024 #64:3333, 4444:512
        , batch_size = BATCH_SIZE
        , n_tracks = n_tracks
        , n_bars = n_bars
        , n_steps_per_bar = n_steps_per_bar
        , n_pitches = n_pitches
        )

gan.load_weights(RUN_FOLDER, None)

#gan.generator.summary()
#gan.critic.summary()

#=============
c_n = rd.random()
s_n = rd.random()
m_n = rd.random()
g_n = rd.random()

#======================================
### sample score
chords_noise = np.random.normal(0, c_n, (1, gan.z_dim))
style_noise = np.random.normal(0, s_n, (1, gan.z_dim))
melody_noise = np.random.normal(0, m_n, (1, gan.n_tracks, gan.z_dim))
groove_noise = np.random.normal(0, g_n, (1, gan.n_tracks, gan.z_dim))

gen_scores = gan.generator.predict([chords_noise, style_noise, melody_noise, groove_noise])

np.argmax(gen_scores[0,0,0:2:,2], axis = 1)
n_rd = rd.random()
print(n_rd)
gen_scores[0,0,0:2,60,1] = n_rd
print(c_n,s_n,m_n,g_n)
#score show
filename = part_name+'example'
gan.notes_to_midi(RUN_FOLDER, gen_scores, filename)
gen_score = converter.parse((RUN_FOLDER+'/samples/{}.midi'.format(filename)))
#gen_score.show()

gan.draw_score(gen_scores, 0)

#=============================
### find_closet score
def find_closest(data_binary, score):
    current_dist = 99999999
    current_i = -1
    for i, d in enumerate(data_binary):
        dist = np.sqrt(np.sum(pow((d - score), 2)))
        if dist < current_dist:
            current_i = i
            current_dist = dist

    return current_i

closest_idx = find_closest(data_binary, gen_scores[0])
closest_data = data_binary[[closest_idx]]
print(closest_idx)

filename = part_name+'closest'
gan.notes_to_midi(RUN_FOLDER, closest_data,filename)
closest_score = converter.parse(os.path.join(RUN_FOLDER, 'samples/{}.midi'.format(filename)))
print('생성된 악보')
#gen_score.show()
print('가장 가까운 악보')
closest_score.show('midi')

###chords
chords_noise_2 = 5 * np.ones((1, gan.z_dim))
chords_scores = gan.generator.predict([chords_noise_2, style_noise, melody_noise, groove_noise])
filename = part_name+'changing_chords'
gan.notes_to_midi(RUN_FOLDER, chords_scores, filename)
chords_score = converter.parse(os.path.join(RUN_FOLDER, 'samples/{}.midi'.format(filename)))
#chords_score.show('midi')
print('생성된 악보')
#gen_score.show()
print('화음 잡음 변경')
#chords_score.show()

###style
'''style_noise_2 = 5 * np.ones((1, gan.z_dim))
style_scores = gan.generator.predict([chords_noise, style_noise_2, melody_noise, groove_noise])
filename = part_name+'changing_style'
gan.notes_to_midi(RUN_FOLDER, style_scores, filename)
style_score = converter.parse(os.path.join(RUN_FOLDER, 'samples/{}.midi'.format(filename)))
print('생성된 악보')
#gen_score.show()
print('스타일 잡음 변경')
#style_score.show()

###groove
groove_noise_2 = np.copy(groove_noise)
groove_noise_2[0,1,:] = 5 * np.ones(gan.z_dim)
groove_scores = gan.generator.predict([chords_noise, style_noise, melody_noise, groove_noise_2])
filename = part_name+'changing_groove'
gan.notes_to_midi(RUN_FOLDER, groove_scores, filename)
groove_score = converter.parse(os.path.join(RUN_FOLDER, 'samples/{}.midi'.format(filename)))
print('생성된 악보')
#gen_score.show()
print('그루브 잡음 변경')'''
#groove_score.show()'''

###finish
print(FILENAME)
print("finish!!!")
