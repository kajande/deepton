# python-deeplearning
python course through deep learning implementation from scratch

__Structure__

io/
  .csv
  .json
data/
  extractor.py
    ExtractCSV
    ExtractJSON
  transformer.py
    FloatTransform
    IntTransform
  loader.py
    cross_validation_split
model/
  network.py
  train.py
    Trainer(loss, optimizer)
      .train(network, dataset)
train/
  loss
  optimizer