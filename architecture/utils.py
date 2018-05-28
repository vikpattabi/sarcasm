
# Define a class containing our language - we want to use the top N common words...
class language:
    def __init__(self):
        self.word_indices = {}
        self.N = -1

    def process_sentence(self, sentence):
        pass

    def process_word(self, word):
        pass

# Words to indices...see example?
def words_to_indexes(language, sentence):
    sentence = sentence.split(' ')
    pass
