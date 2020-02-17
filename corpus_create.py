import lyricsgenius, statistics
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
PYTHONWARNINGS = "ignore"

def clean_in_pandas(corpus):
    ##deletes newline and punctuation, and returns everything lowercase
    corpus['Lyrics'] = corpus['Lyrics'].apply(lambda x: x.replace("\n", ". ").lower())
    return corpus

def fill_happy_sad(df):
    sid = SentimentIntensityAnalyzer()
    i = 0
    for index, row in df.iterrows():
        lyr = getattr(row, "Lyrics")
        ans = sid.polarity_scores(lyr)
        pos, neu, neg = ans["pos"], ans["neu"], ans["neg"]
        opp_neg = 1 - neg
        final = statistics.mean([pos, opp_neg])
        row['pos'], row['neu'], row['neg'], row['opp neg'], row['final score'] = pos, neu, neg, opp_neg, final
        i += 1
    return df

def import_artist_to_pandas(artist_name, max):
    genius = lyricsgenius.Genius("qaZCzxfq2h5IkUFydvbDNA5a74xhU-8paFi95Tc8ymKIPzcGlBi1aqYTPt8nVKai")
    genius.remove_section_headers = True
    artist = genius.search_artist(artist_name, max_songs=max, sort="popularity")
    df = pd.DataFrame(
        columns=['Artist', 'Title', 'Album', 'Year', 'Lyrics', 'Lyrics Link', 'pos', 'neu', 'neg', 'opp neg', 'final score'])
    for song in artist.songs:
        ##truncates date format "YYYY-MM-DD" to year only (if date isn't missing)
        if song.year is not None:
            year = song.year[:4]
        else:
            year = song.year
        df = df.append(
            {'Artist': song.artist, 'Title': song.title, 'Album': song.album, 'Year': year, "Lyrics": song.lyrics, "Lyrics Link": song.url},
            ignore_index=True)
    df = clean_in_pandas(df)
    df = fill_happy_sad(df)
    return df
