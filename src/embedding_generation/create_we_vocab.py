import sys
import pickle as p

PAD_token = '<PAD>'
SOS_token = '<SOS>'
EOS_token = '<EOS>'
GENERIC_token = '<GENERIC>'
SPECIFIC_token = '<SPECIFIC>'

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python create_we_vocab.py <word_vectors.txt> <output_we.p> <output_vocab.p>")
        sys.exit(0)
    word_vectors_file = open(sys.argv[1], 'r')
    word_embeddings = []
    vocab = {}
    vocab[PAD_token] = 0
    vocab[SOS_token] = 1
    vocab[EOS_token] = 2
    vocab[GENERIC_token] = 3
    vocab[SPECIFIC_token] = 4
    word_embeddings.append(None)
    word_embeddings.append(None)
    word_embeddings.append(None)
    word_embeddings.append(None)
    word_embeddings.append(None)
    
    i = 5
    for line in word_vectors_file.readlines():
        vals = line.rstrip().split(' ')
        vocab[vals[0]] = i
        word_embeddings.append(list(map(float, vals[1:])))
        i += 1
    word_embeddings[0] = [0] * len(word_embeddings[5])
    word_embeddings[1] = [0] * len(word_embeddings[5])
    word_embeddings[2] = [0] * len(word_embeddings[5])
    word_embeddings[3] = [0] * len(word_embeddings[5])
    word_embeddings[4] = [0] * len(word_embeddings[5])

    p.dump(word_embeddings, open(sys.argv[2], 'wb'))
    p.dump(vocab, open(sys.argv[3], 'wb'))

