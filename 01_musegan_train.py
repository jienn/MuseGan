import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models.MuseGAN import MuseGAN
from utils.loaders import load_music


physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

### run params
SECTION = 'musik'
RUN_ID = '4444' #3333,4444
DATA_NAME = 'npz'
RUNNAME='musik'   #1-35-44.npz : 4중주.2526 제외. 심도 512
FILENAME = '3+256_444.npz' #20-39, 2-19(3333) 훈련 완료 #4444- 4중주 혹시 몰라서 훈련시키기~ 2-30
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, RUNNAME])

print(RUN_FOLDER)



mode =  'build' #build' # ' 'load' #

### 데이터 적재 (prarmeters 까다로워 뒤짐;)
BATCH_SIZE = 64 #64:3333, 40:4444
n_bars = 10 #45 #2 #can : 10~20 버거워함
n_steps_per_bar = 16 #16
n_pitches = 84 #84
n_tracks = 4 #3333:2, 4444:4
#madi > n_bars * n_steps_per_bar
#findlove maid : 800



data_binary, data_ints, raw_data = load_music(DATA_NAME, FILENAME, n_bars, n_steps_per_bar)
data_binary = np.squeeze(data_binary)

##gpu 사용


### 모델 만들기
gan = MuseGAN(input_dim = data_binary.shape[1:]
        , critic_learning_rate = 0.002
        , generator_learning_rate = 0.001
        , optimiser = 'adam'
        , grad_weight = 10
        , z_dim = 1600
        , batch_size = BATCH_SIZE
        , n_tracks = n_tracks
        , n_bars = n_bars
        , n_steps_per_bar = n_steps_per_bar
        , n_pitches = n_pitches
        )

if mode == 'build':
    gan.save(RUN_FOLDER)
else:
    gan.load_weights(RUN_FOLDER)

#gan.chords_tempNetwork.summary()
gan.barGen[0].summary()
gan.critic.summary()

###모델 훈련

EPOCHS = 10000
PRINT_EVERY_N_BATCHES = 100 #10

gan.epoch = 0

gan.train(
    data_binary
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    ,print_every_n_batches = PRINT_EVERY_N_BATCHES
    ,n_critic=4
)


fig = plt.figure()
plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.25)

plt.plot([x[1] for x in gan.d_losses], color='green', linewidth=0.25)
plt.plot([x[2] for x in gan.d_losses], color='red', linewidth=0.25)
plt.plot(gan.g_losses, color='orange', linewidth=0.25)

plt.xlabel('batch', fontsize=18)
plt.ylabel('loss', fontsize=16)

plt.xlim(0, len(gan.d_losses))
#plt.ylim(0, 2)

plt.savefig('fig_'+FILENAME+'.png',dpi=300)

print('finish!!!')

### finish

