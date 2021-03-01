
import os
from flask import Flask, json, Response, request, render_template, send_file, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import requests
from flask_cors import CORS
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image
import shutil
from PIL import Image
torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print('-------------', device)


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    # img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    # (1, enc_image_size, enc_image_size, encoder_dim)
    encoder_out = encoder(image)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    # (1, num_pixels, encoder_dim)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    # (k, num_pixels, encoder_dim)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor(
        [[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(
        device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(
            k_prev_words).squeeze(1)  # (s, embed_dim)

        # (s, encoder_dim), (s, num_pixels)
        awe, alpha = decoder.attention(encoder_out, h)

        # (s, enc_image_size, enc_image_size)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)

        # gating scalar, (s, encoder_dim)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        h, c = decoder.decode_step(
            torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            # (s)
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat(
            [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(
            set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    # print('seq', seq)
    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    print('####################')
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]
    # print('word', words)
    result = ''
    for t in range(len(words)):
        if t == 0 or t == len(words) - 1:
            continue
        result += words[t]
        result += ' '
    #     if t > 50:
    #         break
    #     plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

    #     plt.text(0, 1, '%s' % (words[t]), color='black',
    #              backgroundcolor='white', fontsize=12)
    #     plt.imshow(image)
    #     current_alpha = alphas[t, :]
    #     if smooth:
    #         alpha = skimage.transform.pyramid_expand(
    #             current_alpha.numpy(), upscale=24, sigma=8)
    #     else:
    #         alpha = skimage.transform.resize(
    #             current_alpha.numpy(), [14 * 24, 14 * 24])
    #     if t == 0:
    #         plt.imshow(alpha, alpha=0)
    #     else:
    #         plt.imshow(alpha, alpha=0.8)
    #     plt.set_cmap(cm.Greys_r)
    #     plt.axis('off')
    # plt.show()
    # print('result', result)
    return result


app = Flask(__name__, static_folder='storage')


def init_app():
    CORS(app)
    cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
    if os.path.exists(os.path.join(os.getcwd(), 'storage')) == False:
        os.mkdir(os.path.join(os.getcwd(), 'storage'))

    app.config['storage'] = os.path.join(os.getcwd(), 'storage')
    app.config['file_allowed'] = ['image/png', 'image/jpeg']


init_app()
path_model = 'BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
path_word_map = 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
beam_size = 5
# Load model
checkpoint = torch.load(path_model, map_location=str(device))
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()
# Load word map (word2ix)
with open(path_word_map, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

VALID_IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
]


def valid_url_extension(url, extension_list=VALID_IMAGE_EXTENSIONS):
    # http://stackoverflow.com/a/10543969/396300
    return any([url.endswith(e) for e in extension_list])


def is_url_image(image_url):
    image_formats = ("image/png", "image/jpeg", "image/jpg")
    r = requests.head(image_url)
    if r.headers["content-type"] in image_formats:
        return True
    return False


def check_request_containt_image_file(request_file):
    if 'img' not in request_file:
        return 1
    file = request_file['img']
    if file.mimetype not in app.config['file_allowed']:
        return 2


def success_handle(code, error_message,  status, mimetype='application/json'):
    # return Response(json.dumps({"code": code, "message": error_message, "status": status}), mimetype=mimetype)
    return jsonify(code=code, message=error_message, status=status)


def error_handle(code, error_message,  status, mimetype='application/json'):
    return Response(json.dumps({"code": code, "message": error_message, "status": status}),  mimetype=mimetype)


def convert_png_to_jpg(path_image):
    path_raw = path_image.replace('jpg', 'png')
    im = Image.open(path_raw)
    rgb_im = im.convert('RGB')
    rgb_im.save(path_image)


@app.route('/api', methods=['GET'])
def homepage():
    print('ahihihihi', flush=True)
    return success_handle(1, "OK", "OK")


@app.route('/api/add_image', methods=['POST'])
def add_image():
    created1 = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(created1, flush=True)
    # user_id = str(request.form['user_id'])
    # if user_id == '':
    #     return error_handle(0, "CHƯA NHẬP TÊN USER", "INVALID")
    # print('user_id', user_id, flush=True)

    flag_check = check_request_containt_image_file(request.files)
    # print('-------')
    if(flag_check == 1):
        # print("Not file in request")
        return error_handle(0, "KHÔNG CÓ FILE TRONG REQUEST", "INVALID")
    if(flag_check == 2):
        # print("File extension is not allowed")
        return error_handle(0, "CHỈ ĐƯỢC UPLOAD FILE THEO DẠNG .JPG .PNG .JPEG", "INVALID")

    file1 = request.files['img']
    try:
        if file1.filename == "":
            print("No filename")
            return error_handle(0, "KHÔNG CÓ ẢNH ĐẦU VÀO", "INVALID")
        filename = secure_filename(file1.filename)
        path_image = os.path.join(app.config['storage'], filename)
        file1.save(path_image)
        if 'png' in path_image:
            path_image = path_image.replace('png', 'jpg')
            convert_png_to_jpg(path_image)
    except Exception as e:
        print(e)
        return error_handle(0, "LỖI THÊM ẢNH", "INVALID")
    # try:
    #     file_path = os.path.join(
    #         app.config['storage'], '{}.jpg'.format(created1))
    #     # print(file_path)
    #     file1.save(file_path)
    try:
        # Encode, decode with attention and beam search
        seq, alphas = caption_image_beam_search(
            encoder, decoder, path_image, word_map, beam_size)
        alphas = torch.FloatTensor(alphas)
        # print(alphas)
        # Visualize caption and attention of best sequence
        result = visualize_att(path_image, seq, alphas, rev_word_map, True)
        # print(result)
        return success_handle(1, result, "VALID")
    except Exception as e:
        print(e)
        return error_handle(0, "LỖI SERVER", "INVALID")


@app.route('/api/add_url_image', methods=['GET'])
def add_url_image():
    created1 = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(created1, flush=True)
    url = str(request.args.get('url_img'))
    print('url', len(url))
    if len(url) > 200:
        return error_handle(0, "URL QUÁ DÀI", "INVALID")
    try:
        flag_check = is_url_image(url)
        if flag_check == False:
            return error_handle(0, "URL KHÔNG CHỨA ẢNH", "INVALID")
        response = requests.get(url, stream=True)
        path_image = 'storage/{}.jpg'.format(created1)
        with open(path_image, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
    except:
        return error_handle(0, "LỖI URL", "INVALID")
    try:
        print(path_image)
        # Encode, decode with attention and beam search
        seq, alphas = caption_image_beam_search(
            encoder, decoder, path_image, word_map, beam_size)
        alphas = torch.FloatTensor(alphas)
        # print(alphas)
        # Visualize caption and attention of best sequence
        result = visualize_att(path_image, seq, alphas, rev_word_map, True)
        # print(result)
        return success_handle(1, result, "VALID")
    except:
        return error_handle(0, "LỖI SERVER", "INVALID")


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000
        #debug=False,
        #threaded=False
    )

