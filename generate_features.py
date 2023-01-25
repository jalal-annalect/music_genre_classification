import multiprocessing
import os
import numpy as np
import random
import argparse
import utils

if __name__ == '__main__':  

    # command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("-audio_dir", "--audio_dir", help = "Input audio directory", required=True, type=str)
    parser.add_argument("-out_dir", "--out_dir", help = "Output out directory", required=True, type=str)
    parser.add_argument("-processes", "--processes", help = "Number of processes", required=True, type=int)
    args = parser.parse_args()

    # where to read audio from
    audio_root_dir = args.audio_dir

    # output dir
    out_dir = args.out_dir

    # n processes
    n_process = args.processes

    # audio files
    audio_files = [elem for elem in os.listdir(audio_root_dir) if elem.endswith(".wav")]

    # shuffling
    for _ in range(100):
        random.shuffle(audio_files)

    print(f"Spliting data into {n_process} equal parts...")
    # ranges for multiprocessing
    audio_files = np.array_split(audio_files, n_process)
    
    print(f"Allocating {n_process} parallel processes...")
    # allocating processes
    processes = [multiprocessing.Process(target=utils.get_batch_features, args=(audio_root_dir, audio_files[i], out_dir,)) for i, _ in enumerate(range(n_process))]

    # start processes
    for i, process in enumerate(processes):
        print(f"Process {i+1} started.")
        process.start()

    # wait until process is finished
    for i, process in enumerate(processes):
        process.join()
        print(f"Process {i+1} is finished.")


    # # creating processes
    # p1 = multiprocessing.Process(target=get_batch_spectrogram, args=(audio_root_dir, audio_files[0], out_dir,))
    # p2 = multiprocessing.Process(target=get_batch_spectrogram, args=(audio_root_dir, audio_files[1], out_dir,))
    # p3 = multiprocessing.Process(target=get_batch_spectrogram, args=(audio_root_dir, audio_files[2], out_dir,))
    # p4 = multiprocessing.Process(target=get_batch_spectrogram, args=(audio_root_dir, audio_files[3], out_dir,))
    # p5 = multiprocessing.Process(target=get_batch_spectrogram, args=(audio_root_dir, audio_files[4], out_dir,))
    # p6 = multiprocessing.Process(target=get_batch_spectrogram, args=(audio_root_dir, audio_files[5], out_dir,))
    # p7 = multiprocessing.Process(target=get_batch_spectrogram, args=(audio_root_dir, audio_files[6], out_dir,))
    # p8 = multiprocessing.Process(target=get_batch_spectrogram, args=(audio_root_dir, audio_files[7], out_dir,))

    # # starting process 1
    # p1.start()
    # # starting process 2
    # p2.start()
    # # starting process 3
    # p3.start()
    # # starting process 4
    # p4.start()
    # # starting process 5
    # p5.start()
    # # starting process 6
    # p6.start()
    # # starting process 7
    # p7.start()
    # # starting process 8
    # p8.start()

    # # wait until process 1 is finished
    # p1.join()
    # # wait until process 2 is finished
    # p2.join()
    # # wait until process 3 is finished
    # p3.join()
    # # wait until process 4 is finished
    # p4.join()
    # # wait until process 5 is finished
    # p5.join()
    # # wait until process 6 is finished
    # p6.join()
    # # wait until process 7 is finished
    # p7.join()
    # # wait until process 8 is finished
    # p8.join()

    # both processes finished
    print("Done!")








