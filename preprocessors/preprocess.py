
'''class Preprocesser:
    def __init__(self, data: DatasetDict):
        self.data = data
        self.stock_name_regex = r"\$[A-Z][A-Z0-9\.]{0,9}"
        self.url_regex = r"\b(https?:\/\/|www\.)\S+\b"
    

    def preprocess(self):
        self.data = self.data.map(self.remove_stock_name)
        self.data = self.data.map(self.remove_url)
        self.data = self.data.map(self.lower_case)
        self.data = self.data.map(self.remove_hyphen)
        self.data = self.data.map(self.strip)
        return self.data

    def strip(self, example):
        example['text'] = example['text'].strip()
        return example
    
    def remove_hyphen(self, example):
        text: str = example['text']
        cleaned = text.replace(" - ", "", 1)
        example['text'] = cleaned
        return example



    def lower_case(self, example):
        example['text'] = example['text'].lower() 
        return example

    def remove_url(self, example):
        example['text'] = re.sub(self.url_regex, "", example['text'])
        return example

    def remove_stock_name(self, example):
        example['text'] = re.sub(self.stock_name_regex, "", example['text'])
        return example

# preprocessor = Preprocesser(ds)
# ds = preprocessor.preprocess()'''