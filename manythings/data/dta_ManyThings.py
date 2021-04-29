"""
http://www.manythings.org/anki/ has a collection
of language pairs that can be used to train
seq2seq models for language translation
"""
import pathlib
import os
from manythings.data.util import download_data
import numpy as np

default_dir = pathlib.Path.cwd() / "manythings" / "datasets"
url = "http://www.manythings.org/anki/"  # web url for language pairs datasets


class ManyThings():
    def __init__(self, lang1_id: str = "fra", lang2_id: str = "eng"):
        self.lang1_id = lang1_id
        self.lang2_id = lang2_id
        

    def load_or_generate(self, download_dir=default_dir):
        pair_name = f"{self.lang1_id}-{self.lang2_id}"
        pair_zip = f"{pair_name}.zip"
        # data is should be in Datasets/pair_name
        # if not, it is downloaded to that directory
        data_parent_dir = os.path.join(default_dir, pair_name)
        if not os.path.exists(data_parent_dir):
            # dir = os.path.join(default_dir, pair_name)
            os.mkdir(data_parent_dir)
            download_data(data_parent_dir, pair_name, url + pair_zip)

            if os.path.exists(data_parent_dir / f"{self.lang1_id}.txt"):
                self.data_txt_file = data_parent_dir / f"{self.lang1_id}.txt"
            else:
                print("data not properly downloaded or extracted")
        else:
            data = os.path.join(data_parent_dir, f"{self.lang1_id}.txt")
            if os.path.exists(data):
                self.data_txt_file = data
            else:
                print("Data not found in parent directory")

    def preprocess(self, num_samples=10000):
        data_txt_file = self.data_txt_file

        # Vectorize the data.
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()

        with open(data_txt_file, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")

        for line in lines[:min(num_samples, len(lines) - 1)]:
            input_text, target_text, _ = line.split("\t")
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = "\t" + target_text + "\n"
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        self.input_characters = sorted(list(input_characters))
        self.target_characters = sorted(list(target_characters))
        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in target_texts])
        self.io_shapes = [self.num_encoder_tokens, self.num_decoder_tokens]

        print("Number of samples:", len(input_texts))
        print("Number of unique input tokens:", self.num_encoder_tokens)
        print("Number of unique output tokens:", self.num_decoder_tokens)
        print("Max sequence length for inputs:", self.max_encoder_seq_length)
        print("Max sequence length for outputs:", self.max_decoder_seq_length)

        input_token_index = dict([
            (char, i) for i, char in enumerate(self.input_characters)
        ])
        target_token_index = dict([
            (char, i) for i, char in enumerate(self.target_characters)
        ])

        self.encoder_input_data = np.zeros(
            (len(input_texts), self.max_encoder_seq_length,
             self.num_encoder_tokens),
            dtype="float32")
        self.decoder_input_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length,
             self.num_decoder_tokens),
            dtype="float32")
        self.decoder_target_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length,
             self.num_decoder_tokens),
            dtype="float32")

        for i, (input_text,
                target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, input_token_index[char]] = 1.0
            self.encoder_input_data[i, t + 1:, input_token_index[" "]] = 1.0

            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self.decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.decoder_target_data[i, t - 1,
                                             target_token_index[char]] = 1.0
            self.decoder_input_data[i, t + 1:, target_token_index[" "]] = 1.0
            self.decoder_target_data[i, t:, target_token_index[" "]] = 1.0

        return


if __name__ == "__main__":
    data = ManyThings()
    data.load_or_generate()
    data.preprocess()
    print("Done!")
