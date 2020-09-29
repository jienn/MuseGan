import os
import pickle
import numpy
import tensorflow as tf
from music21 import note, chord,converter, corpus, instrument, midi, pitch, stream, duration
from models.RNNAttention import get_distinct, create_lookups, prepare_sequences, get_music_list, create_network, sample_with_temp
import random as rd
physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("data check")
#데이터 확인
dataset_name = 'trot'
run_name = 'musik'
filename = '3' #filename
file = "./data/{}/{}.mid".format(dataset_name,filename)
store_folder = "./data/{}/store".format(dataset_name)

original_score = converter.parse(file)

print("set parameter")
#===========
#파라미터 설정
# 실행 파라미터
section = 'musik'
run_id = '4444'
music_name = run_name

run_folder = 'run/{}/'.format(section)
run_folder += '_'.join([run_id, music_name])

store_folder = os.path.join(run_folder, 'store')
data_folder = os.path.join('data', dataset_name)

if not os.path.exists(run_folder):
    os.mkdir(run_folder)
    os.mkdir(os.path.join(run_folder, 'store'))
    os.mkdir(os.path.join(run_folder, 'output'))
    os.mkdir(os.path.join(run_folder, 'weights'))
    os.mkdir(os.path.join(run_folder, 'viz'))
    os.mkdir(os.path.join(run_folder, 'images'))
    os.mkdir(os.path.join(run_folder, 'samples'))


mode = 'build'  # 'load' #

# 데이터 파라미터
intervals = range(1)
seq_len = 256 #64


#==================

print("data extraction")
#데이터 추출
notes = ['', ]
durations = []

for element in original_score.flat:

    if isinstance(element, chord.Chord):
        notes.append(element.pitchNames[0])
        durations.append(element.duration.quarterLength)

    if isinstance(element, note.Note):
        if element.isRest:
            notes.append(str(element.name))
            durations.append(element.duration.quarterLength)
        else:
            notes.append(str(element.pitches[0]))
            durations.append(element.duration.quarterLength)
#=======================

#============
#악보 추출

if mode == 'build':

    music_list, parser = get_music_list(data_folder)
    print(len(music_list), 'files in total')
    notes = []
    durations = []

    for i, file in enumerate(music_list):
        print(i + 1, "Parsing %s" % file)
        original_score = parser.parse(file)

        for interval in intervals:

            score = original_score.transpose(interval)

            '''notes.extend(['START'] * seq_len)
            durations.extend([0] * seq_len)'''

            for element in original_score.flat:

                if isinstance(element, note.Note):
                    if element.isRest:
                        notes.append(str(element.name))
                        durations.append(element.duration.quarterLength)
                    else:
                        notes.append(str(element.nameWithOctave))
                        durations.append(element.duration.quarterLength)

                if isinstance(element, chord.Chord):
                    notes.append(str(element.pitches[0].nameWithOctave))  #('.'.join(n.nameWithOctave for n in element.pitches))
                    durations.append(element.duration.quarterLength)
    # ======= 화음 다 찢어버러기 단음으로
    for i in range(seq_len, len(notes)):
        cut = notes[i].split('.')
        if cut == -1:
            pass
        else:
            notes[i] = cut[rd.randint(0,len(cut)-1)]

    #==================

    with open(os.path.join(store_folder, 'notes'), 'wb') as f:
        pickle.dump(notes, f)  # ['G2', 'D3', 'B3', 'A3', 'B3', 'D3', 'B3', 'D3', 'G2',...]
    with open(os.path.join(store_folder, 'durations'), 'wb') as f:
        pickle.dump(durations, f)
else:
    with open(os.path.join(store_folder, 'notes'), 'rb') as f:
        notes = pickle.load(f)  # ['G2', 'D3', 'B3', 'A3', 'B3', 'D3', 'B3', 'D3', 'G2',...]
    with open(os.path.join(store_folder, 'durations'), 'rb') as f:
        durations = pickle.load(f)



#======================
#룩업 테이블 만들
# 고유한 음표와 박자 얻어오기
note_names, n_notes = get_distinct(notes)
duration_names, n_durations = get_distinct(durations)
distincts = [note_names, n_notes, duration_names, n_durations]

with open(os.path.join(store_folder, 'distincts'), 'wb') as f:
    pickle.dump(distincts, f)

# 음표와 박자 룩업 딕셔너리 만들고 저장하기
note_to_int, int_to_note = create_lookups(note_names)
duration_to_int, int_to_duration = create_lookups(duration_names)
lookups = [note_to_int, int_to_note, duration_to_int, int_to_duration]

with open(os.path.join(store_folder, 'lookups'), 'wb') as f:
    pickle.dump(lookups, f)

#midi 파일 읽어와서, note랑 duration 추출하고. 그 이후에 문자열들을 인덱스 매겨서 int로 전환하기
#note to int, duration_tp_int 까지 끝내면 1차 데이터 전처리 끝
#룩업 테이블 = [note_to_int,int_to_note, duration_to_int, int_to_duration]


#===============
#extend npy
npy = []

'''for n,d in zip(notes,durations):
        l=int(d/md)
        nums=float(lookups[0][n])
        npy.extend([nums]*l)'''


#print(npy)
for n,d in zip(notes,durations):
        nums=float(lookups[0][n])
        npy.append(nums)
npy2 = []
for a,b,c,d in zip(npy,npy,npy,npy):
    npy2.append([a,b,c,d])
'''for a,b in zip(npy,npy):
    npy2.append([a,b])'''



npy5=[]
counter=0
madi = 800 # madi > n_bars * n_step_per_bar 45 * 16 =720
for n in enumerate(npy2):
    if(counter % madi ==0):
        npy5.append(npy2[counter:counter+madi])
    counter+=1

npzz = numpy.array(npy5[0:-1])

npz_path='data/npz/'+filename+'+256_444.npz' #44 : 화음 첫음만, 444:화음 랜덤 음
numpy.savez(npz_path,train=npzz)
print("save! npz_path : ",npz_path)


print(lookups[0])
print(len(lookups[0]))