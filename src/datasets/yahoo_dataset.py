import torchtext.data as data

class YahooDataset(data.Dataset):
    def __init__(self, text_field, path):
         fields = [('text', text_field)]
         examples = []
         for line in open(path):
            text = line.split()[:20]
            examples.append(data.Example.fromlist([text], fields))
         super(YahooDataset, self).__init__(examples, fields)
