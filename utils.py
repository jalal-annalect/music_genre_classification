import tensorflow as tf
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import requests
import tqdm
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# save model summary
def save_model_summary(model, filename):
    with open(filename, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'), show_trainable=True)

# standarize data
def standarize(matrix):
    return (matrix-matrix.mean(axis=0))/matrix.std(axis=0), matrix.mean(axis=0), matrix.std(axis=0)

# extract features from sound wave
def extract_features(root_dir, filename, out_dir):
    audio_path = f"{root_dir}/{filename}"
    sound_wave, sr = librosa.load(audio_path)
    #mfcc
    mfcc = librosa.feature.mfcc(sound_wave)
    mfcc_mean, mfcc_var = mfcc.mean(axis=1), mfcc.var(axis=1)

    # combining features 
    mfcc_features = np.concatenate((mfcc_mean, mfcc_var))

    # tempo
    tempo = librosa.beat.tempo(sound_wave)

    # rms
    rms = librosa.feature.rms(sound_wave)
    rms_mean, rms_var = rms.mean(axis=1), rms.var(axis=1)

    # spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(sound_wave)
    spectral_centroid_mean, spectral_centroid_var = spectral_centroid.mean(axis=1), spectral_centroid.var(axis=1)

    # spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(sound_wave)
    spectral_bandwidth_mean, spectral_bandwidth_var = spectral_bandwidth.mean(axis=1), spectral_bandwidth.var(axis=1)

    # spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(sound_wave)
    spectral_rolloff_mean, spectral_rolloff_var = spectral_rolloff.mean(axis=1), spectral_rolloff.var(axis=1)


    # chroma spectogram
    chroma_stft = librosa.feature.chroma_stft(sound_wave)
    chroma_stft_mean, chroma_stft_var = chroma_stft.mean(), chroma_stft.var()

    # derivative of sound wave
    zero_crossing_rate = librosa.feature.zero_crossing_rate(sound_wave)
    zero_crossing_rate_mean, zero_crossing_rate_var = zero_crossing_rate.mean(axis=1), zero_crossing_rate.var(axis=1)

    # harmonic effects
    harmonic_mean, harmonic_var = librosa.effects.harmonic(sound_wave).mean(), librosa.effects.harmonic(sound_wave).var()

    # percussive
    percussive_mean, percussive_var = librosa.effects.percussive(sound_wave).mean(), librosa.effects.percussive(sound_wave).var()

    features = np.concatenate((chroma_stft_mean.reshape(-1), chroma_stft_var.reshape(-1), rms_mean, rms_var, 
                           spectral_centroid_mean, spectral_centroid_var,
                           spectral_bandwidth_mean, spectral_bandwidth_var, 
                           spectral_rolloff_mean, spectral_rolloff_var,
                           harmonic_mean.reshape(-1), harmonic_var.reshape(-1),
                           mfcc_features, tempo, percussive_mean.reshape(-1), percussive_var.reshape(-1),
                           zero_crossing_rate_mean, zero_crossing_rate_var))

    with open(f"{out_dir}/{filename[:-4]}.npz", 'wb') as outfile:
        np.save(outfile, features)

# process batch features
def get_batch_features(root_dir, files, out_dir):
    pbar = tqdm.tqdm(total=len(files))
    for file in files:
        extract_features(root_dir=root_dir, filename=file, out_dir=out_dir)
        pbar.update(1)
    pbar.update(1)

# make model
def build_model_cnn(input_shape=(None, None, 3), classes=10):
  X_input = tf.keras.layers.Input(input_shape)
  X = tf.keras.layers.Resizing(288, 432)(X_input)
  X = tf.keras.layers.Rescaling(1.0/255)(X)
  X = tf.keras.layers.Conv2D(8,kernel_size=(3,3),strides=(1,1))(X)
  X = tf.keras.layers.BatchNormalization(axis=3)(X)
  X = tf.keras.layers.Activation('relu')(X)
  X = tf.keras.layers.MaxPooling2D((2,2))(X)
  
  X = tf.keras.layers.Conv2D(16,kernel_size=(3,3),strides = (1,1))(X)
  X = tf.keras.layers.BatchNormalization(axis=3)(X)
  X = tf.keras.layers.Activation('relu')(X)
  X = tf.keras.layers.MaxPooling2D((2,2))(X)
  
  X = tf.keras.layers.Conv2D(32,kernel_size=(3,3),strides = (1,1))(X)
  X = tf.keras.layers.BatchNormalization(axis=3)(X)
  X = tf.keras.layers.Activation('relu')(X)
  X = tf.keras.layers.MaxPooling2D((2,2))(X)

  X = tf.keras.layers.Conv2D(64,kernel_size=(3,3),strides=(1,1))(X)
  X = tf.keras.layers.BatchNormalization(axis=-1)(X)
  X = tf.keras.layers.Activation('relu')(X)
  X = tf.keras.layers.MaxPooling2D((2,2))(X)
  
  X = tf.keras.layers.Conv2D(128,kernel_size=(3,3),strides=(1,1))(X)
  X = tf.keras.layers.BatchNormalization(axis=-1)(X)
  X = tf.keras.layers.Activation('relu')(X)
  X = tf.keras.layers.MaxPooling2D((2,2))(X)

  
  X = tf.keras.layers.Flatten()(X)
  
  X = tf.keras.layers.Dropout(rate=0.3)(X)

  X = tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes))(X)

  model = tf.keras.Model(inputs=X_input,outputs=X,name='GenreModel')

  return model

# chunk audio
def chunk_audio(filename, output_dir, window=3):

    # First load the file
    sound_wave, sr = librosa.load(filename)

    # Get number of samples for 2 seconds; replace 2 by any number
    buffer = window*sr

    samples_total = len(sound_wave)
    samples_wrote = 0
    counter = 1

    while samples_wrote < samples_total:

        #check if the buffer is not exceeding total samples 
        if buffer > (samples_total - samples_wrote):
            buffer = samples_total - samples_wrote

        block = sound_wave[samples_wrote : (samples_wrote + buffer)]

        block_duration = librosa.get_duration(block)

        if block_duration>=window:
            out_filename = "split_" + str(counter) + "_" + filename.replace("\\", "_").replace("/", "_")
            out_filename = f"{output_dir}/{out_filename}"

            # Write 2 second segment
            sf.write(out_filename, block, sr)
            
        else:
            pass

        counter += 1
        samples_wrote += buffer

# get mel spectrograms
def get_melspectrogram(root_dir, filename, out_dir, padding=False):
    audio_path = os.path.join(root_dir, filename)
    # Loading demo track
    y, sr = librosa.load(audio_path)

    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    mels_dB = librosa.power_to_db(mels, ref=np.max)
    image = librosa.display.specshow(mels_dB)

    if not padding:
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{out_dir}/{filename[:-4]}.jpg")
    else:
        plt.savefig(f"{out_dir}/{filename[:-4]}.jpg")

# get mel spectrograms batches
def get_batch_spectrogram(input_dir, files, output_dir, padding=False):

    pbar = tqdm.tqdm(total=len(files))
    for file in files:
        get_melspectrogram(input_dir, filename=file, out_dir=output_dir, padding=padding)
        pbar.update(1)
    pbar.close()

# get optimizer
def get_optimizer(optimizer, learning_rate):

    if optimizer.lower()=='sgd':

        return tf.keras.optimizers.SGD(learning_rate=learning_rate)

    elif optimizer.lower()=='adam':

        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    elif optimizer.lower()=='rmsprop':

        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    else:
        raise Exception(f"Not supporting optimizer: {optimizer} at the moment.")

def build_model_mlp(num_classes=10):
    model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(512, activation="relu", input_shape=(57,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation="softmax")])

    return model

class LALALConnector:
    # intilize class
    def __init__(self, filepath):
        "DOCSTRING"
        self.filepath = filepath
        self.content = open(self.filepath, 'rb').read()
        self.license = "d5a093854bfb415f"
        self.filename = os.path.basename(filepath).split('/')[-1]
        self.id = None

    def upload(self):
        request = requests.post(url="https://www.lalal.ai/api/upload/", data=self.content, 
        headers={"Content-Disposition": f"attachment; filename={self.filename}", "Authorization": f"license {self.license}"})

        if request.status_code==200:
            data = request.json()
            self.id = data['id']
            print("Data uploaded successfully.")
        else:
            error_message = request.json()["error"]
            raise Exception(f"An error occured while uploading the data. Traceback: {error_message}.")
    
    def split(self, filter=2, stem="vocals"):
        headers = {"Authorization": f"license {self.license}"}
        params = {"id": self.id, "filter": filter, "stem": stem}
        request = requests.post(url="https://www.lalal.ai/api/split/", headers=headers, data=params)

        if request.status_code==200:
            print("Data split was successful")
        else:
            error_message = request.json()["error"]
            raise Exception(f"An error occured while splitting the data. Traceback: {error_message}.")

    def check(self):
        headers = {"Authorization": f"license {self.license}"}
        params = {"id": self.id}
        request = requests.post(url="https://www.lalal.ai/api/check/", headers=headers, data=params)

        if request.status_code==200:
            split = request.json()
            print("Waiting for response...")
            # wait until stemming is complete
            while not bool(split['result'][self.id]['split']):
                request = requests.post(url="https://www.lalal.ai/api/check/", headers=headers, data=params)
                split = request.json()
            
            # get results
            vocals = split['result'][self.id]['split']['stem_track']
            instrumental = split['result'][self.id]['split']['back_track']
            print("Data check was successful.")
            return {"vocal":vocals, "instrumental":instrumental}
        else:
            error_message = request.json()["error"]
            raise Exception(f"An error occured while checking the data. Traceback: {error_message}.")            

        
            
