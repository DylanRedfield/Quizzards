weights_file.json options_file.json squad.train.json qanta.train.json qanta.test.json qanta.dev.json:
	wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
	mv elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 data/weights_file.json
	wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
	mv elmo_2x4096_512_2048cnn_2xhighway_options.json data/options_file.json

	wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
	mv train-v2.0.json data/squad.train.json

	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json
	mv qanta.train.2018.04.18.json data/qanta.train.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.dev.2018.04.18.json
	mv qanta.dev.2018.04.18.json data/qanta.dev.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.test.2018.04.18.json
	mv qanta.test.2018.04.18.json data/qanta.test.json