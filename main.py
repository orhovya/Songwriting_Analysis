import corpus_create as importer

def import_artist (artist_name, max):
    df = importer.import_artist_to_pandas(artist_name, max)
    df.to_csv(artist_name + ".csv", index=False)

if __name__ == '__main__':
    import_artist("Elvis Costello", 5)




