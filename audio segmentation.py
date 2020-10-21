import pydub
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
import os
import jiwer

# load the Audio
sound = AudioSegment.from_wav(file="converted.wav")
print("total duration:", sound.duration_seconds)
chunks = split_on_silence(sound, min_silence_len=200, silence_thresh=-34, keep_silence=True)
print(len(chunks))


durations_b_m = []
for chunk in chunks:
    durations_b_m.append(chunk.duration_seconds)
print("duration before merge: ", durations_b_m)


duration_minimun = 3
chunks_merged = []

currently_merging = False
merging_chunk = None

for j in range(0, len(chunks)):

    if not currently_merging:
        current_chunk = chunks[j]
    else:
        current_chunk = merging_chunk + chunks[j]

    if current_chunk.duration_seconds > duration_minimun:
         chunks_merged.append(current_chunk)
         merging_chunk = None
         currently_merging = False
    else:
        currently_merging = True
        merging_chunk = current_chunk

durations = []
for i, chunk in enumerate(chunks_merged):
    durations.append(chunk.duration_seconds)
    chunk.export("chunks/chunk{:02d}.wav".format(i), format="wav")

print("duration after merge: ", durations)


# transcribe the segmnets


files = os.listdir("chunks")
sorted_files = sorted(files)
sorted_files = sorted_files[1:]
print("files:", sorted_files)

# transcribe segments to texts


r = sr.Recognizer()

text_segments = []
for file in sorted_files:
    audio = sr.AudioFile("chunks"+"/"+file)
    with audio as source:
        audio_file = r.record(source)
        result = r.recognize_google(audio_file, language='de-DE')
        text_segments.append(result)
    #print(result)

with open('recognized_segments.txt', mode='w') as file:
    for item in text_segments:
        file.write("%s\n" % item)
# evaluation

# open text files
with open("transcribed.txt", "r") as test:
    refs = test.readlines()
with open("recognized.txt", "r") as pred:
    preds = pred.readlines()

reference = refs[0]
predicted = preds[1]

measures = jiwer.compute_measures(refs[0], preds[1])
wer = measures['wer']
mer = measures['mer']
wil = measures['wil']
print("word error rate is: ", wer*100)
print("match error rate is: ", mer*100)
print("Word information lost is:", wil*100)
















