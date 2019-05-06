# used to be necessary before we understood ELMo
#elmo_weights.hdf5 elmo_options.json:
#	wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
#	mv elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 elmo_weights.hdf5
#	wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
#	mv elmo_2x4096_512_2048cnn_2xhighway_options.json elmo_options.json

squad.train.json squad.dev.json:
	wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
	mv train-v2.0.json squad.train.json
	wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
	mv dev-v2.0.json squad.dev.json

qanta.train.json qanta.test.json qanta.dev.json:
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json
	mv qanta.train.2018.04.18.json qanta.train.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.dev.2018.04.18.json
	mv qanta.dev.2018.04.18.json qanta.dev.json
	wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.test.2018.04.18.json
	mv qanta.test.2018.04.18.json qanta.test.json
