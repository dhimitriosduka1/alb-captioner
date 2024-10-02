from transformers import MarianMTModel, MarianTokenizer

class EnToSqTranslator:
    """
    Translator class to translate text from English to Albanian.
    """
    def __init__(self):
        MODEL_NAME     = "Helsinki-NLP/opus-mt-en-sq"
        self.model     = MarianMTModel.from_pretrained(MODEL_NAME)
        self.tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME) 

    def translate(self, sentence):
        """
        Translate a single sentence from English to Albanian.
        """
        tokens = self.tokenizer(sentence, return_tensors="pt", padding=True)
        translated = self.model.generate(**tokens)

        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

    def translate_batch(self, sentences):
        """
        Translate a batch of sentences from English to Albanian.
        """
        return [self.translate(sentence) for sentence in sentences]