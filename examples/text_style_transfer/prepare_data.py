#
"""Downloads data.
"""
import texar as tx

# pylint: disable=invalid-name

def prepare_data():
    """Downloads data.
    """
    tx.data.maybe_download(
        urls='https://drive.google.com/file/d/'
             '1HaUKEYDBEk6GlJGmXwqYteB-4rS9q8Lg/view?usp=sharing',
        path='./',
        filenames='yelp.zip',
        extract=True)

def main():
    """Entrypoint.
    """
    prepare_data()

if __name__ == '__main__':
    main()
