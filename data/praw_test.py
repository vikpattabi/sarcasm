import numpy as np
import praw
import bz2

reddit = praw.Reddit(client_id='9Hi6zn1RTsCTWA',
                     client_secret='iPo2wCmH5rmoXHNj0oqmrhbch-U',
                     password='Neurotic1',
                     user_agent='testscript by /u/vikpattabi',
                     username='vikpattabi')

# def get_sarc_data(filename):
#     with bz2.BZ2File(filename, 'r') as infile:
#
#         for line in infile:
#             print(line)


# get_sarc_data('sarc_09-12.csv.bz2')
