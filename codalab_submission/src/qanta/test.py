import elmo
import train as training
import guess as guessing
import util
from config import *
from flask import Flask, jsonify, request
import torch

'''
Test.py
Entire file is specifically for docker commands
Calls functions in the train.py and the guess.py
'''

def create_app(enable_batch=True):
    elmo_guesser = elmo.ElmoGuesser()
    elmo_guesser.load()
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guessing.guess(question)

        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True,
            'include_wiki_paragraphs': False
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in guessing.batch_guess(questions)
        ])
    return app

@click.group()
def cli():
    pass

@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)

@cli.command()
def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """
    Train the model calls methods from train.py
    """
    if TRAIN_TYPE == 'elmo':
        training.elmo_train(device)
    else:
        print('Configure TRAIN_TYPE in config.py')

    if BUZZ_TYPE == 'rnn':
        training.buzz_rnn_train(device)

@cli.command()
@click.option('--local-qanta-prefix', default='data/')
@click.option('--retrieve-paragraphs', default=False, is_flag=True)
def download(local_qanta_prefix, retrieve_paragraphs):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix, retrieve_paragraphs)


if __name__ == '__main__':
    cli()