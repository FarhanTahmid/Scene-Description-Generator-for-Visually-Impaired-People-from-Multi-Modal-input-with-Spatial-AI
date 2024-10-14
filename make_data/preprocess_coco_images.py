from __future__ import print_function
import torch
import os
import sys
import argparse
import pickle
import PIL
import torch

# check if cuda is available at first
print(f"Is GPU available for pytorch? {torch.cuda.is_available()}")
print(f"Count of GPU: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name()}, Capability: {torch.cuda.get_device_capability()}")

BASE_DIR=os.environ['PROJECT_DIRECTORY']
sys.path.append(BASE_DIR+'software_utils/')
sys.path.append(BASE_DIR+'models/')

from create_transformer import create_transformer
from encoderCNN import EncoderCNN


def main(args):
    """Generate image embeddings using desired CNN from COCO images."""
    encoder = EncoderCNN(args.model)
    if torch.cuda.is_available():
        encoder.cuda()
    encoder.eval()

    transformer = create_transformer()

    sets = ['train2014', 'val2014', 'train2017', 'val2017']

    for i, st in enumerate(sets):
        print('Preprocessing ' + st + ' images with ' + args.model)
        # Get list of images
        images = [f.split('.')[0]
                  for f in os.listdir(args.dir + st) if '.jpg' in f]
        
        # Remove any images that have already been processed
        if args.continue_preprocessing:
            converted = [f.split('.')[0] for f in os.listdir(
                args.dir + st) if args.model + '.pkl' in f]
            images = [f for f in images if f not in converted]

        # Load, transform, encode, and then pickle each image
        for j, im_file in enumerate(images):
            if j % 100 == 0:
                print('[{}/{}] Progress: {}%\r'.format(
                    i, len(sets), round(j /
                                        float(len(images))*100.0), 2), end='')
            try:
                im = PIL.Image.open(
                    args.dir + st + '/' + im_file + '.jpg').convert('RGB')
                if torch.cuda.is_available():
                    im = transformer(im).cuda().unsqueeze(0)
                else:
                    im = transformer(im).unsqueeze(0)
                im = encoder(im)

                with open(args.dir + st + '/' + im_file + '_' + args.model
                          + '.pkl', 'wb') as f:
                    pickle.dump(im, f)
            except OSError as e:
                print('Could not process ' + args.dir +
                      st + '/' + im_file + '.jpg')

        sys.stdout.flush()
        print('Done preprocessing ' + st + ' images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Base CNN encoding model',
                        default='resnet152')
    parser.add_argument('--dir', type=str,
                        help='Directory containing images to encode',
                        default=os.environ['PROJECT_DIRECTORY'] + 'Dataset/coco/images/')
    parser.add_argument('--continue_preprocessing', type=bool,
                        help='Continue preprocessing or start from scratch',
                        default=False)
    args = parser.parse_args()
    main(args)