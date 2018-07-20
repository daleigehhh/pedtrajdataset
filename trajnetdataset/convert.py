"""Create Trajnet data from original datasets."""

import subprocess

import pysparkling
import scipy.io
import trajnettools

from . import readers
from .scene import Scenes


def biwi(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.biwi)
            .cache())


def crowds(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .values()
            .flatMap(readers.crowds)
            .cache())


def mot(sc, input_file):
    """Was 7 frames per second in original recording."""
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.mot)
            .filter(lambda r: r.frame % 2 == 0)
            .cache())


def edinburgh(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .zipWithIndex()
            .flatMap(readers.edinburgh)
            .cache())


def syi(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .flatMap(readers.syi)
            .cache())


def dukemtmc(sc, input_file):
    print('processing ' + input_file)
    contents = scipy.io.loadmat(input_file)['trainData']
    return (sc
            .parallelize(readers.dukemtmc(contents))
            .cache())


def wildtrack(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .flatMap(readers.wildtrack)
            .cache())


def write(input_rows, output_file, train_fraction=0.6, val_fraction=0.2):
    frames = sorted(set(input_rows.map(lambda r: r.frame).toLocalIterator()))
    train_split_index = int(len(frames) * train_fraction)
    val_split_index = train_split_index + int(len(frames) * val_fraction)
    train_frames = set(frames[:train_split_index])
    val_frames = set(frames[train_split_index:val_split_index])
    test_frames = set(frames[val_split_index:])

    # train dataset
    train_rows = input_rows.filter(lambda r: r.frame in train_frames)
    train_output = output_file.format(split='train')
    train_scenes = Scenes().rows_to_file(train_rows, train_output)

    # validation dataset
    val_rows = input_rows.filter(lambda r: r.frame in val_frames)
    val_output = output_file.format(split='val')
    val_scenes = Scenes(start_scene_id=train_scenes.scene_id).rows_to_file(val_rows, val_output)

    # public test dataset
    test_rows = input_rows.filter(lambda r: r.frame in test_frames)
    test_output = output_file.format(split='test')
    test_scenes = Scenes(start_scene_id=val_scenes.scene_id, chunk_stride=21, visible_chunk=9)
    test_scenes.rows_to_file(test_rows, test_output)
    # private test dataset
    private_test_output = output_file.format(split='test_private')
    private_test_scenes = Scenes(start_scene_id=val_scenes.scene_id, chunk_stride=21)
    private_test_scenes.rows_to_file(test_rows, private_test_output)


def write_without_split(input_rows, output_file):
    Scenes().rows_to_file(input_rows, output_file)


def main():
    sc = pysparkling.Context()

    # new datasets
    write(wildtrack(sc, 'data/raw/wildtrack/Wildtrack_dataset/annotations_positions/*.json'),
          'output/{split}/wildtrack.ndjson')
    write(dukemtmc(sc, 'data/raw/duke/trainval.mat'),
          'output/{split}/dukemtmc.ndjson')
    write(syi(sc, 'data/raw/syi/0?????.txt'),
          'output/{split}/syi.ndjson')

    # originally train
    write(biwi(sc, 'data/raw/biwi/seq_hotel/obsmat.txt'),
          'output/{split}/biwi_hotel.ndjson')
    # write(crowds(sc, 'data/raw/crowds/arxiepiskopi1.vsp'),
    #       'output/{split}/crowds_arxiepiskopi1.ndjson')
    write(crowds(sc, 'data/raw/crowds/crowds_zara02.vsp'),
          'output/{split}/crowds_zara02.ndjson')
    write(crowds(sc, 'data/raw/crowds/crowds_zara03.vsp'),
          'output/{split}/crowds_zara03.ndjson')
    write(crowds(sc, 'data/raw/crowds/students001.vsp'),
          'output/{split}/crowds_students001.ndjson')
    write(crowds(sc, 'data/raw/crowds/students003.vsp'),
          'output/{split}/crowds_students003.ndjson')

    # originally test
    write_without_split(biwi(sc, 'data/raw/biwi/seq_eth/obsmat.txt'),
                        'output/test_holdout/biwi_eth.ndjson')
    write_without_split(crowds(sc, 'data/raw/crowds/crowds_zara01.vsp'),
                        'output/test_holdout/crowds_zara01.ndjson')
    write_without_split(crowds(sc, 'data/raw/crowds/uni_examples.vsp'),
                        'output/test_holdout/crowds_uni_examples.ndjson')

    # unused datasets
    write_without_split(edinburgh(sc, 'data/raw/edinburgh/tracks.*.zip'),
                        'output/unused/edinburgh.ndjson')
    write_without_split(mot(sc, 'data/raw/mot/pets2009_s2l1.txt'),
                        'output/unused/mot_pets2009_s2l1.ndjson')

    # compress the outputs
    subprocess.check_output(['tar', '-czf', 'output/train.tar.gz', 'output/train'])
    subprocess.check_output(['tar', '-czf', 'output/val.tar.gz', 'output/val'])
    subprocess.check_output(['tar', '-czf', 'output/test.tar.gz', 'output/test'])
    subprocess.check_output(['tar', '-czf', 'output/test_private.tar.gz', 'output/test_private'])
    subprocess.check_output(['tar', '-czf', 'output/unused.tar.gz', 'output/unused'])


if __name__ == '__main__':
    main()
