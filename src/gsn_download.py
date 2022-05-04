import requests


def main():
    gsn_dir = '/home/jingyihe/scratch/inf_mix_chaining/gsn/'
    for i in range(99):
        node_id = str(i).rjust(2, '0')
        url = 'http://commondatastorage.googleapis.com/books/syntactic-ngrams/eng/verbargs.' \
              + node_id + '-of-99.gz'
        r = requests.get(url, stream=True)
        with open(gsn_dir + 'verb_args_{}.gz'.format(i), 'wb') as f:
            for chunk in r.raw.stream(1024, decode_content=False):
                if chunk:
                    f.write(chunk)


if __name__ == '__main__':
    main()

