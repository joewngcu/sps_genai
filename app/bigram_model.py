import random
import re
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus):
        # Combine the sample sentences into one text.
        text = " ".join(corpus)

        # Convert to lowercase and extract words.
        words = re.findall(r"\b\w+\b", text.lower())

        # Count adjacent word pairs and individual words.
        bigram_counts = Counter(zip(words[:-1], words[1:]))
        unigram_counts = Counter(words)

        # Calculate bigram probabilities.
        self.bigram_probs = defaultdict(dict)

        for (word1, word2), count in bigram_counts.items():
            self.bigram_probs[word1][word2] = (
                count / unigram_counts[word1]
            )

    def generate_text(self, start_word, length):
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word)

            if not next_words:
                break

            current_word = random.choices(
                list(next_words.keys()),
                weights=list(next_words.values())
            )[0]

            generated_words.append(current_word)

        return " ".join(generated_words)